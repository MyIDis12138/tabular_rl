import gymnasium as gym
from gymnasium import spaces
import numpy as np
import io
from contextlib import closing
import time # Added for human rendering delay

class CustomGridWorld(gym.Env):
    """
    Custom Grid World Environment for Reinforcement Learning.

    Follows the Gymnasium API.
    Allows specifying grid size, start/goal/hole locations, rewards,
    stochasticity (slipperiness), and multi-step actions.

    Action Space: Discrete(4 * max_step_size)
        - Represents flattened (direction, step_size) pairs.
        - Mapping: action = (step_size - 1) * 4 + direction
          - direction: 0: Up, 1: Down, 2: Left, 3: Right
          - step_size: 1 to max_step_size
        - Example: max_step_size=3 -> actions 0-3 are step 1 (U,D,L,R), 4-7 step 2, 8-11 step 3.
    Observation Space: Discrete(rows * cols)
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, rows=5, cols=5, start_coords=(0, 0), goal_coords=None,
                 hole_coords=None, reward_map=None, is_slippery=False,
                 max_step_size=1, # Max steps per action (>= 1)
                 render_mode=None):
        """
        Initializes the Custom Grid World.

        Args:
            rows (int): Number of rows.
            cols (int): Number of columns.
            start_coords (tuple): (row, col) of the starting position.
            goal_coords (tuple): (row, col) of the goal position. Defaults to bottom-right.
            hole_coords (set): Set of (row, col) tuples for holes. Defaults to empty.
            reward_map (dict): Maps 'goal', 'hole', 'step' to rewards.
            is_slippery (bool): If True, transitions are stochastic.
            max_step_size (int): Maximum number of unit steps per action (must be >= 1).
            render_mode (str | None): Rendering mode ('human', 'ansi', or None).
        """
        super().__init__()

        assert max_step_size >= 1, "max_step_size must be at least 1"

        self.rows = rows
        self.cols = cols
        self.size = rows * cols
        self.is_slippery = is_slippery
        self.max_step_size = max_step_size
        self.render_mode = render_mode

        # Default goal and holes if not provided
        if goal_coords is None: goal_coords = (rows - 1, cols - 1)
        if hole_coords is None: hole_coords = set()
        if reward_map is None: reward_map = {'goal': 1.0, 'hole': -1.0, 'step': -0.01}

        self.start_coords = tuple(start_coords)
        self.goal_coords = tuple(goal_coords)
        self.hole_coords = set(set([tuple(t) for t in hole_coords]))
        self.reward_map = reward_map

        # --- Input Validation ---
        assert 0 <= self.start_coords[0] < self.rows and 0 <= self.start_coords[1] < self.cols
        assert 0 <= self.goal_coords[0] < self.rows and 0 <= self.goal_coords[1] < self.cols
        assert self.start_coords != self.goal_coords
        assert self.start_coords not in self.hole_coords
        assert self.goal_coords not in self.hole_coords
        for r, c in self.hole_coords: assert 0 <= r < self.rows and 0 <= c < self.cols
        # --- End Validation ---

        # --- Define spaces ---
        self.observation_space = spaces.Discrete(self.size)
        # Action space is flattened: 4 directions * max_step_size possible steps
        self.action_space = spaces.Discrete(4 * self.max_step_size)
        self.num_directions = 4 # Store for decoding action

        # Action mapping (row_delta, col_delta)
        self._action_to_delta = { 0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1) }
        self._perpendicular_actions = { 0: [2, 3], 1: [2, 3], 2: [0, 1], 3: [0, 1] }

        # Create state/coordinate mappings
        self._state_to_coords = {i: (r, c) for i, (r, c) in enumerate(np.ndindex(rows, cols))}
        self._coords_to_state = {v: k for k, v in self._state_to_coords.items()}

        self.start_state = self._coords_to_state[self.start_coords]
        self.goal_state = self._coords_to_state[self.goal_coords]
        self.hole_states = {self._coords_to_state[hc] for hc in self.hole_coords}
        self.terminal_states = self.hole_states.union({self.goal_state})

        # Internal state
        self._agent_coords = None
        self.np_random = None # Will be initialized in reset()

    def _decode_action(self, flat_action):
        """ Decodes a flattened action integer into direction and step_size. """
        if not (0 <= flat_action < self.action_space.n):
             raise ValueError(f"Invalid flat_action: {flat_action}. Must be 0 <= action < {self.action_space.n}")
        direction = flat_action % self.num_directions
        step_size_index = flat_action // self.num_directions
        step_size = step_size_index + 1 # Map 0..max-1 to 1..max
        return direction, step_size

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_coords = self.start_coords
        observation = self.start_state
        info = self._get_info()
        if self.render_mode == "human": self._render_frame()
        return observation, info

    def step(self, action):
        """ Executes potentially multiple steps based on the flattened action integer. """
        flat_action = action # Action is now an integer

        # Check if already in a terminal state before starting
        start_state_idx = self._coords_to_state[self._agent_coords]
        if start_state_idx in self.terminal_states:
             return start_state_idx, 0.0, True, False, self._get_info()

        # Decode the flat action
        intended_direction, step_size = self._decode_action(flat_action)

        current_coords = self._agent_coords
        terminated = False
        final_reward = self.reward_map['step'] # Assume step cost unless goal/hole hit

        for _ in range(step_size):
            # Determine actual move direction for this unit step
            if self.is_slippery:
                p = self.np_random.uniform()
                if p < 1/3.: actual_direction = intended_direction
                elif p < 2/3.: actual_direction = self._perpendicular_actions[intended_direction][0]
                else: actual_direction = self._perpendicular_actions[intended_direction][1]
                actual_delta = self._action_to_delta[actual_direction]
            else:
                actual_delta = self._action_to_delta[intended_direction]

            # Calculate potential next coordinates for this unit step
            next_row = current_coords[0] + actual_delta[0]
            next_col = current_coords[1] + actual_delta[1]

            # Check boundaries - stay in place if hitting wall
            next_row = max(0, min(next_row, self.rows - 1))
            next_col = max(0, min(next_col, self.cols - 1))
            next_coords = (next_row, next_col)

            # Update current coords for next iteration or final state
            current_coords = next_coords
            current_state_idx = self._coords_to_state[current_coords]

            # Check for immediate termination within the multi-step
            if current_state_idx == self.goal_state:
                 final_reward = self.reward_map['goal']
                 terminated = True
                 break # Stop multi-step immediately
            elif current_state_idx in self.hole_states:
                 final_reward = self.reward_map['hole']
                 terminated = True
                 break # Stop multi-step immediately

        # Multi-step move finished (or broken early)
        self._agent_coords = current_coords
        observation = self._coords_to_state[self._agent_coords]

        # Assign final reward (overwrites step cost if terminated at goal/hole)
        reward = final_reward
        truncated = False # No truncation in this version
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _get_info(self):
        return {"agent_coords": self._agent_coords}

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        elif self.render_mode == "human":
            self._render_frame()
        # else: return None implicitly

    def _render_text(self):
        outfile = io.StringIO()
        for r in range(self.rows):
            for c in range(self.cols):
                coords = (r, c)
                if coords == self._agent_coords: char = "A"
                elif coords == self.goal_coords: char = "G"
                elif coords in self.hole_coords: char = "H"
                else: char = "."
                outfile.write(char + " ")
            outfile.write("\n")
        outfile.write("\n")
        with closing(outfile): return outfile.getvalue()

    def _render_frame(self):
        if self.render_mode == "human":
             print(self._render_text())
             time.sleep(1.0 / self.metadata["render_fps"])

    def close(self):
        pass

# ##############################################
# Example Usage and Registration (Optional)
# ##############################################

if __name__ == '__main__':
    from gymnasium.envs.registration import register

    # Helper to encode action for testing
    def encode_action(direction, step_size, num_directions=4):
        step_size_index = step_size - 1
        return step_size_index * num_directions + direction

    try:
        register(
             id='CustomGridWorld-v2', # Changed version to v2 due to action space change
             entry_point='__main__:CustomGridWorld',
             max_episode_steps=200,
             kwargs={'rows': 5, 'cols': 6, 'max_step_size': 2}
        )
        print("Registered CustomGridWorld-v2")
    except Exception as e:
        print(f"Could not register CustomGridWorld-v2 (may already be registered): {e}")

    # --- Test Direct Instantiation ---
    print("\nTesting Direct Instantiation with Flattened Multi-Step Actions:")
    env_direct = CustomGridWorld(rows=5, cols=5, is_slippery=False, render_mode='ansi',
                                 start_coords=(0,0), goal_coords=(4,4),
                                 hole_coords={(1,1), (2,3), (3,1)},
                                 max_step_size=3) # Allow up to 3 steps

    print(f"Action space: {env_direct.action_space}") # Should be Discrete(12)

    obs, info = env_direct.reset(seed=42)
    print("Initial State:", obs)
    print(env_direct.render())

    # Take a few multi-step actions using the flattened encoding
    # Direction: 0: Up, 1: Down, 2: Left, 3: Right
    # Step size: 1 to 3 (index 0 to 2)
    # flat_action = (step_size - 1) * 4 + direction
    actions = [
        encode_action(direction=1, step_size=2), # Down, 2 steps -> (2-1)*4 + 1 = 5
        encode_action(direction=3, step_size=3), # Right, 3 steps -> (3-1)*4 + 3 = 11
        encode_action(direction=1, step_size=1), # Down, 1 step -> (1-1)*4 + 1 = 1
    ]
    total_reward = 0
    for i, action in enumerate(actions):
        direction, step_size = env_direct._decode_action(action) # Decode for printing
        print(f"--- Step {i+1}, Action: {action} (Dir: {direction}, Steps: {step_size}) ---")
        obs, reward, terminated, truncated, info = env_direct.step(action)
        total_reward += reward
        print(env_direct.render())
        print(f"State: {obs}, Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
        if terminated or truncated:
            print("Episode Finished.")
            break
    print(f"Total Reward: {total_reward:.2f}")
    env_direct.close()

    # --- Test gym.make() if registered ---
    print("\nTesting gym.make() with Flattened Multi-Step Actions:")
    try:
        # Uses default max_step_size=2 from registration kwargs -> Discrete(8)
        env_gym = gym.make('CustomGridWorld-v2', render_mode='ansi', rows=6, cols=4, is_slippery=True)
        print(f"Created env via make. Action space: {env_gym.action_space}")
        obs, info = env_gym.reset(seed=123)
        print("Initial State (gym.make):", obs)
        print(env_gym.render())
        # Take one random multi-step action
        action = env_gym.action_space.sample() # Sample flat action
        direction, step_size = env_gym._decode_action(action) # Decode for printing
        print(f"--- Step 1, Random Action: {action} (Dir: {direction}, Steps: {step_size}) ---")
        obs, reward, terminated, truncated, info = env_gym.step(action)
        print(env_gym.render())
        print(f"State: {obs}, Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
        env_gym.close()
    except Exception as e:
        print(f"Could not create environment using gym.make(): {e}")
