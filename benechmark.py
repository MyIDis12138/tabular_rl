import subprocess
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil
import sys
import multiprocessing
import time

from utils.utils import load_config as project_load_config
from utils.utils import deep_merge

# --- Add project root to Python path ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Configuration ---
# IMPORTANT: Files listed in ALGO_CONFIG_FILES should NO LONGER contain the 'imports:' key
#            for the environment.
ALGO_CONFIG_FILES = [
    "configs/qlearning.yaml",     # Assumes 'imports:' is removed
    "configs/sarsa.yaml",         # Assumes 'imports:' is removed
    "configs/rmax.yaml",          # Assumes 'imports:' is removed
    "configs/actor_critic.yaml",  # Assumes 'imports:' is removed
]
# List of environment config files defining different difficulties
ENV_CONFIG_FILES = [
    "configs/env_easy.yaml",
    "configs/env_medium.yaml",
    "configs/env_hard.yaml",
    "configs/env_veryhard.yaml",
]
SEEDS = [42, 123, 999, 1024, 2025] # Or your desired list of seeds
RESULTS_BASE_DIR = "results_benchmark_multi_env"
PYTHON_EXECUTABLE = "python" # Or "python3"

# --- Parallelism Configuration ---
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1) # Adjust as needed

# --- Plotting Configuration ---
SMOOTHING_WINDOW = 100
USE_STANDARD_ERROR = True

# --- Helper Functions ---

def save_config(config_data, config_path):
    """Saves a dictionary to a YAML config file."""
    try:
        # Ensure config_path is a Path object for consistency
        config_path = Path(config_path)
        os.makedirs(config_path.parent, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, Dumper=yaml.SafeDumper)
        return True
    except Exception as e:
        print(f"Error saving config {config_path}: {e}")
        return False

def run_experiment_worker(temp_config_path_str):
    """Worker function for multiprocessing. Returns (config_path_str, success_bool)."""
    # Ensure paths are handled correctly, convert string back to Path if needed
    config_path = Path(temp_config_path_str)
    command = [PYTHON_EXECUTABLE, "run.py", "--config", str(config_path)] # Pass path as string
    worker_pid = os.getpid()
    print(f"[Worker {worker_pid}] Running command: {' '.join(command)}")
    try:
        project_root = Path(__file__).resolve().parent
        result = subprocess.run(
            command,
            check=True,        # Raise error on non-zero exit code
            capture_output=True, # Capture stdout/stderr
            text=True,         # Decode stdout/stderr as text
            cwd=project_root   # Run from project root
        )
        print(f"[Worker {worker_pid}] Experiment {config_path.name} completed successfully.")
        return (temp_config_path_str, True)
    except subprocess.CalledProcessError as e:
        print(f"--- [Worker {worker_pid}] ERROR running experiment for {config_path.name} ---")
        print(f"Return code: {e.returncode}")
        # Print full logs on error
        print("--- Command stdout: ---")
        print(e.stdout)
        print("--- Command stderr: ---")
        print(e.stderr)
        return (temp_config_path_str, False)
    except Exception as e:
         # Catch other potential errors during subprocess execution
         print(f"--- [Worker {worker_pid}] UNEXPECTED ERROR for {config_path.name}: {e} ---")
         return (temp_config_path_str, False)

def parse_results(results_dir):
    """Parses training and evaluation results from YAML files."""
    results = {}
    results_dir = Path(results_dir) # Ensure it's a Path object
    try:
        eval_path = results_dir / "evaluation_results.yaml"
        if eval_path.is_file(): # Check if it's a file
            with open(eval_path, 'r') as f:
                eval_data = yaml.safe_load(f) or {} # Handle empty file
                results['eval_avg_reward'] = eval_data.get('avg_reward')
                results['eval_avg_steps'] = eval_data.get('avg_steps')
        else:
             # Reduce noise: print only if directory exists but file doesn't
             if results_dir.is_dir(): print(f"Warning: Eval results file not found: {eval_path}")
             results['eval_avg_reward'] = None
             results['eval_avg_steps'] = None

        train_path = results_dir / "training_stats.yaml"
        if train_path.is_file(): # Check if it's a file
             with open(train_path, 'r') as f:
                 train_data = yaml.safe_load(f) or {} # Handle empty file
                 results['train_episode_rewards'] = train_data.get('episode_rewards')
        else:
             if results_dir.is_dir(): print(f"Warning: Training stats file not found: {train_path}")
             results['train_episode_rewards'] = None

    except Exception as e:
        # More specific error message
        print(f"Error parsing result files in directory {results_dir}: {e}")
        # Ensure keys exist even if parsing fails mid-way
        results.setdefault('eval_avg_reward', None)
        results.setdefault('eval_avg_steps', None)
        results.setdefault('train_episode_rewards', None)
    return results


# --- Main Benchmark Logic ---

start_time_main = time.time()
all_results_data = []
experiment_details = []

benchmark_runs_dir = Path(RESULTS_BASE_DIR)
os.makedirs(benchmark_runs_dir, exist_ok=True)
temp_config_dir = benchmark_runs_dir / "_temp_configs"

# --- 1. Prepare all temporary configs and experiment details ---
print("--- Preparing all experiment configurations ---")
tasks_to_run = []
for algo_config_path_str in ALGO_CONFIG_FILES:
    algo_file = Path(algo_config_path_str) # Create Path object
    print(f"\nProcessing Algorithm Config: {algo_file.name}")
    if not algo_file.is_file():
        print(f"ERROR: Algorithm config file not found: {algo_file}. Skipping.")
        continue
    try:
        algo_base_config = project_load_config(algo_file) # Pass Path object
        if not algo_base_config:
            print(f"Warning: Loaded empty or invalid algo config from {algo_file}. Skipping.")
            continue
        if 'imports' in algo_base_config:
             print(f"WARNING: Algorithm config '{algo_file.name}' contains 'imports:'. It will be ignored by this script.")
             # del algo_base_config['imports'] # Optionally remove
    except Exception as e:
        print(f"ERROR loading algo config {algo_file}: {e}")
        continue # Skip this algo config if loading fails

    for env_config_path_str in ENV_CONFIG_FILES:
        env_file = Path(env_config_path_str) # Create Path object
        print(f"  Processing Env Config: {env_file.name}")
        if not env_file.is_file():
             print(f"ERROR: Environment config file not found: {env_file}. Skipping this environment for {algo_file.name}.")
             continue
        try:
            env_base_config = project_load_config(env_file) # Pass Path object
            # Check structure AFTER loading
            if not env_base_config or 'environment' not in env_base_config:
                print(f"Warning: Loaded env config from {env_file} is invalid (must contain top-level 'environment' key). Skipping.")
                continue
        except Exception as e:
            # Pass the specific file path to the error message
            print(f"ERROR loading env config {env_file}: {e}")
            continue # Skip this env config if loading fails

        # Merge the loaded configs using the imported deep_merge
        try:
            merged_config = deep_merge(algo_base_config, env_base_config)
        except Exception as e:
             print(f"ERROR merging {algo_file.name} and {env_file.name}: {e}")
             continue # Skip this combination if merging fails

        # Extract details for naming and tracking
        algo_name = merged_config.get('algorithm', {}).get('class_path', 'UnknownAgent').split('.')[-1]
        env_base_name = env_file.stem # Use env file name (e.g., "env_hard")
        # Use name from algo config 'experiment' section if present, else algo filename stem
        algo_exp_name_base = merged_config.get('experiment', {}).get('name', algo_file.stem)
        base_exp_name = f"{algo_exp_name_base}_{env_base_name}" # Combined name

        print(f"    Algo: {algo_name}, Env: {env_base_name}, BaseName: {base_exp_name}")

        # Ensure experiment dict exists
        exp_config = merged_config.setdefault('experiment', {})

        for seed in SEEDS:
            # Create a fresh config copy for this specific run
            # Using python's deepcopy is safest for complex nested dicts
            import copy
            current_config = copy.deepcopy(merged_config)

            exp_name = f"{base_exp_name}_seed{seed}"
            current_config['experiment']['name'] = exp_name
            current_config['experiment']['seed'] = seed
            current_config['experiment']['results_dir'] = str(benchmark_runs_dir)

            temp_config_path = temp_config_dir / f"{exp_name}_config.yaml"
            if save_config(current_config, temp_config_path):
                # Add the temporary config path (as string) to the list of tasks
                tasks_to_run.append(str(temp_config_path))
                # Store metadata for parsing results later
                experiment_details.append({
                    'temp_config_path': str(temp_config_path),
                    'exp_name': exp_name,
                    'algorithm': algo_name,
                    'env_name': env_base_name,
                    'seed': seed,
                    'algo_config_base': str(algo_file),
                    'env_config_base': str(env_file)
                })
            else:
                print(f"  Skipping run {exp_name} due to config save error.")

# --- 2. Run experiments in parallel ---
if not tasks_to_run:
     print("\nNo valid experiment configurations prepared. Exiting.")
     sys.exit(0)

print(f"\n--- Starting parallel execution with {NUM_WORKERS} workers for {len(tasks_to_run)} tasks ---")
start_time_parallel = time.time()
pool = None
parallel_results = [] # Initialize empty list
try:
    # Use context manager for the pool for cleaner handling
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        parallel_results = pool.map(run_experiment_worker, tasks_to_run)
except Exception as e:
     print(f"\n--- ERROR during parallel execution: {e} ---")
     # Pool cleanup is handled by context manager
finally:
    # Ensure pool resources are released (often handled by context manager, but explicit can be good)
    if pool: pool.close(); pool.join()

end_time_parallel = time.time()
if parallel_results: # Check if any results were actually returned
    print(f"--- Parallel execution finished in {end_time_parallel - start_time_parallel:.2f} seconds ---")
else:
     print("--- Parallel execution completed, but no results were returned (may indicate errors). ---")


# --- 3. Parse results sequentially after parallel runs ---
print("\n--- Parsing results ---")
details_map = {detail['temp_config_path']: detail for detail in experiment_details}

# Check if parallel_results has the expected structure
if not isinstance(parallel_results, list):
     print(f"ERROR: Unexpected format for parallel_results: {type(parallel_results)}. Cannot parse.")
     parallel_results = [] # Reset to prevent further errors

for result_item in parallel_results:
    # Add checks for result_item format
    if not isinstance(result_item, tuple) or len(result_item) != 2:
         print(f"Warning: Skipping invalid item from parallel results: {result_item}")
         continue

    temp_config_path_str, success = result_item
    detail = details_map.get(temp_config_path_str)
    if not detail:
        print(f"Warning: Could not find details for completed task {temp_config_path_str}. Skipping parsing.")
        continue

    if success:
        run_results_dir = benchmark_runs_dir / detail['exp_name']
        # Only attempt parsing if the results directory actually exists
        if run_results_dir.is_dir():
             parsed = parse_results(run_results_dir)
             parsed['algorithm'] = detail['algorithm']
             parsed['env_name'] = detail['env_name']
             parsed['seed'] = detail['seed']
             parsed['algo_config_base'] = detail['algo_config_base']
             parsed['env_config_base'] = detail['env_config_base']
             all_results_data.append(parsed)
        else:
             print(f"Warning: Results directory not found for successful run {detail['exp_name']}. Appending placeholder.")
             success = False # Treat as failure if dir is missing

    # Use 'if not success' to catch both explicit failures and cases where dir was missing
    if not success:
        # Append placeholder if run failed or results dir missing
        print(f"Appending placeholder results for: {detail['exp_name']}")
        all_results_data.append({
            'algorithm': detail['algorithm'],
            'env_name': detail['env_name'],
            'seed': detail['seed'],
            'algo_config_base': detail['algo_config_base'],
            'env_config_base': detail['env_config_base'],
            'eval_avg_reward': None,
            'eval_avg_steps': None,
            'train_episode_rewards': None
        })

# --- 4. Aggregate and Display Results ---
print("\n===== Benchmark Summary =====")

if not all_results_data:
    print("No results collected or parsed.")
else:
    df = pd.DataFrame(all_results_data)
    df['eval_avg_reward'] = pd.to_numeric(df['eval_avg_reward'], errors='coerce')
    df['eval_avg_steps'] = pd.to_numeric(df['eval_avg_steps'], errors='coerce')

    print("\n--- Evaluation Metrics (Averaged Across Seeds per Algo/Env) ---")
    # Use observed=True in groupby if pandas version supports it, helps with categorical data
    eval_summary = df.groupby(['algorithm', 'env_name'], observed=True).agg(
        avg_eval_reward=('eval_avg_reward', 'mean'),
        std_eval_reward=('eval_avg_reward', 'std'),
        avg_eval_steps=('eval_avg_steps', 'mean'),
        std_eval_steps=('eval_avg_steps', 'std'),
        runs=('seed', 'count'),
        successful_runs=('eval_avg_reward', 'count') # Counts non-NaN rewards
    ).reset_index()
    eval_summary = eval_summary.round({'avg_eval_reward': 3, 'std_eval_reward': 3, 'avg_eval_steps': 1, 'std_eval_steps': 1})
    eval_summary = eval_summary.sort_values(by=['env_name', 'algorithm'])
    print(eval_summary.to_string(index=False))

    # --- Plotting per environment ---
    print(f"\n--- Generating Learning Curve Plots per Environment (Smoothing: {SMOOTHING_WINDOW}, Bands: {'SEM' if USE_STANDARD_ERROR else 'STD'}) ---")
    unique_envs = df['env_name'].unique()

    for env_name in unique_envs:
        print(f"  Generating plot for environment: {env_name}")
        plt.figure(figsize=(12, 8)) # Create a new figure for each env
        env_df = df[df['env_name'] == env_name].copy() # Create a copy for filtering

        max_episodes = 0
        # Use .loc to avoid SettingWithCopyWarning if modifying env_df later
        valid_rewards_series = env_df.loc[env_df['train_episode_rewards'].notna(), 'train_episode_rewards']
        for rewards in valid_rewards_series:
            if rewards: max_episodes = max(max_episodes, len(rewards))

        if max_episodes == 0:
            print(f"    No valid training reward data found for env '{env_name}'.")
            plt.close() # Close the unused figure
            continue

        plot_has_data = False
        algorithms_in_env = env_df['algorithm'].unique()
        for algo in algorithms_in_env:
            # Filter for algo AND current env, ensuring rewards are not NaN
            algo_env_df = env_df[(env_df['algorithm'] == algo) & env_df['train_episode_rewards'].notna()]
            if algo_env_df.empty: continue

            rewards_padded = []
            run_count_for_algo = 0
            for r_list in algo_env_df['train_episode_rewards']:
                 if r_list and len(r_list) > 0: # Check list is not None and not empty
                     # Pad with NaN up to max_episodes for this environment
                     padded = np.pad(r_list, (0, max_episodes - len(r_list)), 'constant', constant_values=np.nan)
                     rewards_padded.append(padded)
                     run_count_for_algo += 1

            if run_count_for_algo > 0:
                plot_has_data = True
                rewards_array = np.array(rewards_padded)
                # Calculate statistics ignoring NaNs
                mean_rewards = np.nanmean(rewards_array, axis=0)
                std_rewards = np.nanstd(rewards_array, axis=0)
                std_rewards = np.nan_to_num(std_rewards) # Replace NaN std (e.g., N=1) with 0

                # Apply Smoothing using pandas on the calculated mean
                mean_rewards_series = pd.Series(mean_rewards)
                smoothed_mean_rewards = mean_rewards_series.rolling(
                    window=SMOOTHING_WINDOW, min_periods=1, center=False
                ).mean().to_numpy()

                # Calculate Error Band (SEM or STD)
                if USE_STANDARD_ERROR:
                    # Calculate SEM = std_dev / sqrt(N), handle N=0 case
                    sem = std_rewards / np.sqrt(run_count_for_algo) if run_count_for_algo > 0 else np.zeros_like(std_rewards)
                    error_band = sem
                else:
                    error_band = std_rewards

                # Smooth the error band
                error_band_series = pd.Series(error_band)
                smoothed_error_band = error_band_series.rolling(
                    window=SMOOTHING_WINDOW, min_periods=1, center=False
                ).mean().to_numpy()

                # Handle potential NaNs from smoothing at the start
                nan_mask = np.isnan(smoothed_mean_rewards) # Find where NaNs might be
                smoothed_mean_rewards[nan_mask] = 0.0 # Or some other placeholder if 0 is misleading
                smoothed_error_band[nan_mask] = 0.0

                episodes = np.arange(len(smoothed_mean_rewards)) + 1

                # Plot smoothed data
                line, = plt.plot(episodes, smoothed_mean_rewards, label=f"{algo} (N={run_count_for_algo})")
                plt.fill_between(
                    episodes,
                    smoothed_mean_rewards - smoothed_error_band,
                    smoothed_mean_rewards + smoothed_error_band,
                    alpha=0.2,
                    color=line.get_color()
                )

        # Configure and save the plot for the current environment if it has data
        if plot_has_data:
            plt.title(f'Learning Curves - Env: {env_name} (Smoothed over {SMOOTHING_WINDOW} episodes)')
            plt.xlabel('Episode')
            plt.ylabel(f'Smoothed Average Reward ({"SEM" if USE_STANDARD_ERROR else "STD"} bands)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout() # Adjust layout to prevent labels overlapping
            plot_filename = benchmark_runs_dir / f"learning_curves_env_{env_name}_smooth{SMOOTHING_WINDOW}_{'sem' if USE_STANDARD_ERROR else 'std'}.png"
            try:
                plt.savefig(plot_filename)
                print(f"    Learning curve plot saved to: {plot_filename}")
            except Exception as e:
                print(f"    Error saving plot for {env_name}: {e}")
            plt.show() # Display the plot for the current environment
        else:
            plt.close() # Close the figure if no data was plotted

# --- 5. Cleanup ---
try:
    # Check if the directory exists before attempting removal
    if temp_config_dir.is_dir():
        shutil.rmtree(temp_config_dir)
        print(f"Cleaned up temporary config directory: {temp_config_dir}")
except Exception as e:
    print(f"Warning: Could not remove temporary config directory {temp_config_dir}: {e}")

end_time_main = time.time()
print(f"\nBenchmark script finished. Total time: {end_time_main - start_time_main:.2f} seconds.")