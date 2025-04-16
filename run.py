import yaml
import argparse
import gymnasium as gym
import numpy as np
from pathlib import Path
import time
import traceback

from utils.utils import *

# ==============================================
# Core Logic Functions
# ==============================================

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run RL Algorithms defined by Configuration")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file for the experiment.")
    return parser.parse_args()

def create_environment(env_config, seed):
    """Creates the environment based on the config and seeds it."""
    print("\n--- Creating Environment ---")
    env_params = env_config.get('params', {})
    env_id = env_config.get('id')
    env_class_path = env_config.get('class_path')
    terminal_states = set()
    env = None

    try:
        if env_class_path:
            print(f"Creating custom environment: {env_class_path}")
            EnvClass = load_class(env_class_path)
            env = EnvClass(**env_params)
            print(f"Custom environment '{type(env).__name__}' created.")
            if hasattr(env, 'terminal_states'):
                terminal_states = env.terminal_states
                print(f"Fetched terminal states from custom env: {terminal_states if len(terminal_states) < 20 else str(len(terminal_states)) + ' states'}")
            else:
                 print(f"Warning: Custom environment does not have 'terminal_states' attribute.")
        elif env_id:
            print(f"Creating Gym environment: {env_id}")
            env = gym.make(env_id, **env_params)
            print(f"Gym environment '{env_id}' created.")
            if 'FrozenLake' in env_id:
                 terminal_states = get_frozenlake_terminal_states(env)
                 print(f"Identified FrozenLake Terminal States: {terminal_states}")
        else:
            raise ValueError("Environment config needs 'id' or 'class_path'")

        # Seed environment after creation
        if seed is not None:
            try:
                 # Try new seeding API first
                 env.reset(seed=seed)
                 print("Seeded environment using env.reset(seed=...).")
            except TypeError:
                 print(f"Warning: env.reset() for {type(env).__name__} might not support seeding. Trying space seeding.")
            # Always try to seed spaces
            try:
                 if hasattr(env, 'action_space') and hasattr(env.action_space, 'seed'):
                     env.action_space.seed(seed)
                     print(f"Seeded action space.")
            except Exception as e: print(f"Warning: Could not seed action space: {e}")
            try:
                 if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'seed'):
                     env.observation_space.seed(seed)
                     print(f"Seeded observation space.")
            except Exception as e: print(f"Warning: Could not seed observation space: {e}")

        return env, terminal_states

    except Exception as e:
        print(f"ERROR: Failed during environment creation: {e}")
        traceback.print_exc()
        if env: env.close() # Attempt cleanup if partially created
        raise # Re-raise the exception to stop execution

def create_agent(algo_config, env, terminal_states):
    """Loads and initializes the agent based on the config."""
    print("\n--- Initializing Agent ---")
    agent_class_path = algo_config.get('class_path')
    algo_params = algo_config.get('hyperparameters', {})
    if not agent_class_path:
         raise ValueError("Algorithm configuration must contain 'class_path'")

    try:
        AgentClass = load_class(agent_class_path)
        
        # Initialize agent with env and terminal_states
        # The new BaseAgent architecture handles state/action extraction from env
        agent = AgentClass(
            env=env, 
            terminal_states=terminal_states, 
            **algo_params
        )
        
        print(f"Initialized Agent: {AgentClass.__name__}")
        print(f"  with Hyperparameters: {algo_params}")
        return agent
    except Exception as e:
        print(f"ERROR: Failed to initialize agent {agent_class_path}: {e}")
        traceback.print_exc()
        raise # Re-raise the exception

def setup_results_dir(config, config_path):
    """Creates the results directory and saves the config."""
    results_dir = Path(config.get('experiment', {}).get('results_dir', 'results'))
    exp_name = config.get('experiment', {}).get('name', config_path.stem)
    run_dir = results_dir / exp_name
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved in: {run_dir.resolve()}")
        with open(run_dir / 'config_used.yaml', 'w') as f:
             yaml.dump(config, f, default_flow_style=False)
        return run_dir
    except Exception as e:
         print(f"Warning: Could not create results directory {run_dir} or save config: {e}")
         return Path(".") # Fallback

def run_training(agent, train_config, env, run_dir):
    """Runs the agent's training loop and saves stats."""
    print("\n--- Starting Training ---")
    try:
        total_episodes = train_config['total_episodes'] # Required
        max_steps = train_config.get('max_steps_per_episode', 200) # Default
        print(f"Training for {total_episodes} episodes (max {max_steps} steps/ep)...")

        # Use the standardized train method from BaseAgent
        train_stats = agent.train(
            total_episodes=total_episodes, 
            max_steps_per_episode=max_steps,
            env=env
        )
        
        print("Training finished successfully.")

        # Save training stats if returned
        if train_stats and isinstance(train_stats, dict):
             try:
                 with open(run_dir / 'training_stats.yaml', 'w') as f:
                     serializable_stats = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in train_stats.items()}
                     yaml.dump(serializable_stats, f, default_flow_style=False)
                 print("Training stats saved.")
             except Exception as e:
                 print(f"Warning: Could not save training stats: {e}")
        return True # Indicate success

    except KeyError as e:
        print(f"ERROR: Missing required experiment parameter in config: {e}")
    except Exception as e:
        print(f"ERROR: An error occurred during training: {e}")
        traceback.print_exc()
    return False # Indicate failure

def save_agent_state(agent, algo_config, run_dir):
    """Saves the agent's state/model."""
    save_filename = algo_config.get('save_model_filename', f"{type(agent).__name__}_final.agent")
    model_save_path = run_dir / save_filename
    if hasattr(agent, 'save'):
        try:
            agent.save(model_save_path)
            # Agent's save method should print confirmation
        except Exception as e:
            print(f"Warning: Failed to save agent state to {model_save_path}: {e}")
    else:
        print("Warning: Agent does not have a 'save' method. Skipping save.")

def run_evaluation(agent, eval_config, env, run_dir):
    """Runs the evaluation loop and saves results."""
    print("\n--- Evaluating Final Policy ---")
    if not hasattr(agent, 'choose_action'):
         print("Warning: Agent does not have 'choose_action' method. Skipping evaluation.")
         return

    eval_episodes = eval_config.get('eval_episodes', 100)
    eval_max_steps = eval_config.get('max_steps_per_episode', 200)
    eval_rewards = []
    eval_steps = []
    eval_success = []

    print(f"Evaluating for {eval_episodes} episodes (max {eval_max_steps} steps/ep)...")
    for i in range(eval_episodes):
        obs_result = env.reset() # Use seeded reset if possible
        if isinstance(obs_result, tuple):  # Gym v26+ returns (obs, info)
            state, info = obs_result
        else:  # Backwards compatibility
            state = obs_result
            info = {}
            
        episode_reward = 0
        done = False
        steps = 0
        for step in range(eval_max_steps):
            # Use evaluate=True to get greedy policy
            action = agent.choose_action(state, evaluate=True)
            if action is None: break

            try:
                result = env.step(action)
                if len(result) == 5:  # Gym v26+ returns (obs, reward, terminated, truncated, info)
                    next_state, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:  # Backwards compatibility
                    next_state, reward, done, info = result
                    
                episode_reward += reward
                state = next_state
                steps = step + 1
                if done:
                    if isinstance(info, dict) and 'success' in info: eval_success.append(info['success'])
                    break
            except Exception as e:
                 print(f"ERROR during eval step {step} (action {action}): {e}")
                 break # Stop this episode on error

        eval_rewards.append(episode_reward)
        eval_steps.append(steps)

    # Print & Save Evaluation Results
    avg_reward = np.mean(eval_rewards) if eval_rewards else 0
    std_reward = np.std(eval_rewards) if eval_rewards else 0
    avg_steps = np.mean(eval_steps) if eval_steps else 0
    print(f"Average Evaluation Reward ({eval_episodes} episodes): {avg_reward:.3f} +/- {std_reward:.3f}")
    print(f"Average Evaluation Steps ({eval_episodes} episodes): {avg_steps:.1f}")

    eval_results = {'num_episodes': eval_episodes, 'avg_reward': float(avg_reward),
                    'std_reward': float(std_reward), 'avg_steps': float(avg_steps),
                    'rewards_list': eval_rewards, 'steps_list': eval_steps}
    if eval_success:
        success_rate = np.mean(eval_success)
        print(f"Success Rate: {success_rate:.2%}")
        eval_results['success_rate'] = float(success_rate)
        eval_results['success_list'] = eval_success

    try:
         with open(run_dir / 'evaluation_results.yaml', 'w') as f:
             yaml.dump(eval_results, f, default_flow_style=False)
         print("Evaluation results saved.")
    except Exception as e: print(f"Warning: Could not save evaluation results: {e}")


# ==============================================
# Main Orchestrator Function
# ==============================================
def main():
    """Main function to orchestrate the RL experiment."""
    start_time = time.time()
    env = None # Initialize env to None for cleanup block
    try:
        # 1. Configuration & Setup
        args = parse_arguments()
        config = load_config(args.config)
        config_path = Path(args.config)
        seed_used = setup_seed(config.get('experiment', {}).get('seed', None))

        # 2. Environment Creation
        env, terminal_states = create_environment(config.get('environment', {}),
                                                  config.get('experiment', {}).get('seed', None))

        # 3. Agent Creation
        agent = create_agent(config.get('algorithm', {}), env, terminal_states)

        # 4. Results Directory Setup
        run_dir = setup_results_dir(config, config_path)

        # 5. Training
        training_successful = run_training(agent, config.get('experiment', {}), env, run_dir)

        # 6. Save Agent State (if training succeeded)
        if training_successful:
            save_agent_state(agent, config.get('algorithm', {}), run_dir)

        # 7. Evaluation (always run evaluation if agent exists)
        run_evaluation(agent, config.get('experiment', {}), env, run_dir)

    except (FileNotFoundError, ValueError, ImportError, AttributeError, KeyError, Exception) as e:
        # Catch specific expected errors and general exceptions
        print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: A critical error occurred: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Print traceback for unexpected errors
        if not isinstance(e, (FileNotFoundError, ValueError, ImportError, AttributeError, KeyError)):
             traceback.print_exc()
    finally:
        # 8. Cleanup
        end_time = time.time()
        print("\n--- Run Finished ---")
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
        if env is not None:
            try:
                env.close()
                print("Environment closed.")
            except Exception as e:
                print(f"Warning: Error closing environment: {e}")

# ==============================================
# Entry Point Guard
# ==============================================
if __name__ == "__main__":
    main()