import gymnasium as gym
from gymnasium import spaces
import numpy as np
import io
from contextlib import closing
import time
import random

class CustomGridWorld(gym.Env):
    """
    Custom Grid World Environment for Reinforcement Learning.

    Follows the Gymnasium API.
    Can be initialized either via a map layout (list of strings) or by
    specifying dimensions and coordinates explicitly.

    Allows specifying grid size, start/goal/hole/wall locations, rewards,
    stochasticity (slipperiness), action failure probability, and multi-step actions.

    Action Space: Discrete(4 * max_step_size)
    Observation Space: Discrete(rows * cols)
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    # Define characters used in map layout
    MAP_CHARS = {
        'START': 'A',
        'GOAL': 'G',
        'HOLE': 'H',
        'WALL': '#',
        'EMPTY': '.'
    }

    def __init__(self, 
                 map_layout=None,
                 rows=None, 
                 cols=None,
                 start_coords=None, 
                 goal_coords=None,
                 hole_coords=None, 
                 wall_coords=None,
                 reward_map=None, 
                 is_slippery=False,
                 action_failure_prob=0.0,
                 action_failure_outcomes=None,
                 max_step_size=1,
                 render_mode=None):
        """
        Initializes the Custom Grid World.

        Prioritizes `map_layout` if provided. Otherwise, requires `rows`, `cols`,
        `start_coords`, and `goal_coords`.

        Args:
            map_layout (list[str], optional): A list of strings defining the grid.
                                             Characters: 'A' (Start), 'G' (Goal),
                                             'H' (Hole), '#' (Wall), '.' (Empty).
                                             Must be rectangular and contain exactly
                                             one 'A' and one 'G'. Defaults to None.
            rows (int, optional): Number of rows. Required if `map_layout` is None.
            cols (int, optional): Number of columns. Required if `map_layout` is None.
            start_coords (tuple, optional): (row, col) of start. Required if `map_layout` is None.
            goal_coords (tuple, optional): (row, col) of goal. Required if `map_layout` is None.
                                          Defaults to bottom-right if `map_layout` is None and not specified.
            hole_coords (set | list, optional): Set/List of (row, col) tuples for holes.
                                               Defaults to empty if `map_layout` is None and not specified.
            wall_coords (set | list, optional): Set/List of (row, col) tuples for walls.
                                               Defaults to empty if `map_layout` is None and not specified.
            reward_map (dict, optional): Maps 'goal', 'hole', 'step' to rewards. Defaults provided.
            is_slippery (bool): If True, successful non-failure actions have stochastic transitions.
            action_failure_prob (float): Probability (0.0 to 1.0) that the intended action fails.
            action_failure_outcomes (dict, optional): Probabilities for outcomes ('S', 'L', 'R', 'B')
                                                      when action fails. Defaults provided if failure_prob > 0.
            max_step_size (int): Maximum number of unit steps per action (must be >= 1).
            render_mode (str | None): Rendering mode ('human', 'ansi', or None).
        """
        super().__init__()

        # --- Basic Parameter Setup ---
        assert max_step_size >= 1, "max_step_size must be at least 1"
        assert 0.0 <= action_failure_prob <= 1.0, "action_failure_prob must be between 0.0 and 1.0"
        self.is_slippery = is_slippery
        self.action_failure_prob = action_failure_prob
        self.max_step_size = max_step_size
        self.render_mode = render_mode
        self.action_failure_outcomes = action_failure_outcomes # Will be validated later

        # --- Determine Grid Configuration (Map Layout or Explicit Params) ---
        if map_layout is not None:
            self._init_from_map(map_layout)
            # Optionally warn if other coordinate params were passed but ignored
            if any(p is not None for p in [rows, cols, start_coords, goal_coords, hole_coords, wall_coords]):
                print("Warning: map_layout provided. Explicit coordinate/size parameters (rows, cols, start_coords, etc.) are ignored.")
        else:
            self._init_from_params(rows, cols, start_coords, goal_coords, hole_coords, wall_coords)

        # --- Common Initialization Logic ---
        self.size = self.rows * self.cols

        # Default reward map
        if reward_map is None:
            self.reward_map = {'goal': 1.0, 'hole': -1.0, 'step': -0.01}
        else:
            self.reward_map = reward_map

        # Perform validation on the final configuration
        self._validate_configuration()

        # Setup action failure outcomes properly now that prob is known
        self._setup_action_failure()

        # Define spaces, state mappings, and helper dictionaries
        self._initialize_spaces_and_mappings()

        self._agent_coords = None # Set in reset
        self.np_random = None    # Initialized in reset

    def _init_from_map(self, map_layout):
        """ Helper to initialize grid dimensions and coords from map_layout. """
        if not isinstance(map_layout, list) or not all(isinstance(row, str) for row in map_layout):
            raise ValueError("map_layout must be a list of strings.")
        if not map_layout: raise ValueError("map_layout cannot be empty.")

        parsed_rows = len(map_layout)
        if parsed_rows == 0: raise ValueError("map_layout cannot be empty.")
        parsed_cols = len(map_layout[0])
        if parsed_cols == 0: raise ValueError("map_layout rows cannot be empty.")
        if not all(len(row) == parsed_cols for row in map_layout):
            raise ValueError("All rows in map_layout must have the same length.")

        self.rows = parsed_rows
        self.cols = parsed_cols

        parsed_start = None
        parsed_goal = None
        parsed_holes = set()
        parsed_walls = set()

        allowed_chars = set(self.MAP_CHARS.values())

        for r, row_str in enumerate(map_layout):
            for c, char in enumerate(row_str):
                coords = (r, c)
                if char == self.MAP_CHARS['START']:
                    if parsed_start is not None: raise ValueError("map_layout contains multiple Start locations.")
                    parsed_start = coords
                elif char == self.MAP_CHARS['GOAL']:
                    if parsed_goal is not None: raise ValueError("map_layout contains multiple Goal locations.")
                    parsed_goal = coords
                elif char == self.MAP_CHARS['HOLE']:
                    parsed_holes.add(coords)
                elif char == self.MAP_CHARS['WALL']:
                    parsed_walls.add(coords)
                elif char == self.MAP_CHARS['EMPTY']:
                    pass # Valid empty cell
                else:
                    raise ValueError(f"Unrecognized character '{char}' at ({r},{c}) in map_layout. Allowed: {allowed_chars}")

        if parsed_start is None: raise ValueError("map_layout must contain exactly one Start ('A') location.")
        if parsed_goal is None: raise ValueError("map_layout must contain exactly one Goal ('G') location.")

        self.start_coords = parsed_start
        self.goal_coords = parsed_goal
        self.hole_coords = parsed_holes
        self.wall_coords = parsed_walls

    def _init_from_params(self, rows, cols, start_coords, goal_coords, hole_coords, wall_coords):
         """ Helper to initialize grid dimensions and coords from explicit parameters. """
         if rows is None or cols is None or start_coords is None:
              raise ValueError("If map_layout is not provided, rows, cols, and start_coords must be specified.")

         self.rows = rows
         self.cols = cols
         self.start_coords = tuple(start_coords)

         # Default goal if not provided
         if goal_coords is None:
             self.goal_coords = (rows - 1, cols - 1)
             print(f"Warning: goal_coords not specified. Defaulting to bottom-right: {self.goal_coords}")
         else:
              self.goal_coords = tuple(goal_coords)

         # Default holes/walls if not provided
         self.hole_coords = set(tuple(t) for t in hole_coords) if hole_coords else set()
         self.wall_coords = set(tuple(t) for t in wall_coords) if wall_coords else set()


    def _validate_configuration(self):
         """ Helper method to run validation checks after coords are set. """
         assert self.rows > 0 and self.cols > 0, "Grid dimensions must be positive."
         assert 0 <= self.start_coords[0] < self.rows and 0 <= self.start_coords[1] < self.cols, f"Start coords {self.start_coords} out of bounds for grid ({self.rows}x{self.cols})"
         assert 0 <= self.goal_coords[0] < self.rows and 0 <= self.goal_coords[1] < self.cols, f"Goal coords {self.goal_coords} out of bounds for grid ({self.rows}x{self.cols})"
         assert self.start_coords != self.goal_coords, "Start and Goal cannot be the same"
         assert self.start_coords not in self.hole_coords, f"Start location {self.start_coords} cannot be a Hole"
         assert self.goal_coords not in self.hole_coords, f"Goal location {self.goal_coords} cannot be a Hole"
         assert self.start_coords not in self.wall_coords, f"Start location {self.start_coords} cannot be a Wall"
         assert self.goal_coords not in self.wall_coords, f"Goal location {self.goal_coords} cannot be a Wall"

         for r, c in self.hole_coords: assert 0 <= r < self.rows and 0 <= c < self.cols, f"Hole coord {(r,c)} out of bounds"
         for r, c in self.wall_coords: assert 0 <= r < self.rows and 0 <= c < self.cols, f"Wall coord {(r,c)} out of bounds"
         assert self.hole_coords.isdisjoint(self.wall_coords), f"Holes and Walls cannot overlap. Overlap: {self.hole_coords.intersection(self.wall_coords)}"


    def _setup_action_failure(self):
        """ Setup and validate action failure parameters. """
        if self.action_failure_prob > 0.0:
            if self.action_failure_outcomes is None:
                # Default failure outcome: just stay put
                self.action_failure_outcomes = {'S': 1.0, 'L': 0.0, 'R': 0.0, 'B': 0.0}
                print("Warning: action_failure_prob > 0 but action_failure_outcomes not specified. Defaulting to {'S': 1.0}")
            else:
                # Validate provided outcomes
                allowed_keys = {'S', 'L', 'R', 'B'}
                if not set(self.action_failure_outcomes.keys()).issubset(allowed_keys):
                    raise ValueError(f"Invalid keys in action_failure_outcomes. Allowed keys: {allowed_keys}")
                prob_sum = sum(self.action_failure_outcomes.values())
                if abs(prob_sum - 1.0) > 1e-6:
                     raise ValueError(f"Probabilities in action_failure_outcomes must sum to 1.0. Got: {prob_sum}")
            # Precompute for faster selection during step
            self._failure_outcomes_keys = list(self.action_failure_outcomes.keys())
            self._failure_outcomes_probs = list(self.action_failure_outcomes.values())
        else:
            # Ensure outcomes is None if no failure prob (cleaner state)
            self.action_failure_outcomes = None


    def _initialize_spaces_and_mappings(self):
        """ Helper method to initialize spaces, dictionaries and derived state sets. """
        # --- Define spaces ---
        self.observation_space = spaces.Discrete(self.size)
        self.action_space = spaces.Discrete(4 * self.max_step_size)
        self.num_directions = 4

        # --- Action/Delta Mappings ---
        # Mapping: 0: Up (-1, 0), 1: Down (1, 0), 2: Left (0, -1), 3: Right (0, 1)
        self._action_to_delta = { 0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1) }
        self._delta_to_action = {v: k for k, v in self._action_to_delta.items()} # For relative moves
        # Perpendicular actions for slipperiness
        self._perpendicular_actions = { 0: [2, 3], 1: [2, 3], 2: [0, 1], 3: [0, 1] }

        # --- State Mappings ---
        self._state_to_coords = {i: (r, c) for i, (r, c) in enumerate(np.ndindex(self.rows, self.cols))}
        self._coords_to_state = {v: k for k, v in self._state_to_coords.items()}

        self.start_state = self._coords_to_state[self.start_coords]
        self.goal_state = self._coords_to_state[self.goal_coords]
        self.hole_states = {self._coords_to_state[hc] for hc in self.hole_coords}
        self.wall_states = {self._coords_to_state[wc] for wc in self.wall_coords}
        self.terminal_states = self.hole_states.union({self.goal_state})

    # ===========================================================================
    # Methods below (step, reset, render, etc.) remain largely the same as before
    # No changes needed unless they directly relied on how __init__ structured things
    # ===========================================================================

    def _decode_action(self, flat_action):
        """ Decodes a flattened action integer into direction and step_size. """
        if not (0 <= flat_action < self.action_space.n):
             raise ValueError(f"Invalid flat_action: {flat_action}. Must be 0 <= action < {self.action_space.n}")
        direction = flat_action % self.num_directions
        step_size = flat_action // self.num_directions + 1
        return direction, step_size

    def _get_relative_delta(self, intended_direction, relative_outcome):
        """ Calculates the actual delta based on intended direction and relative outcome. """
        intended_delta = self._action_to_delta[intended_direction]
        dr, dc = intended_delta

        if relative_outcome == 'S': return (0, 0)
        elif relative_outcome == 'B': return (-dr, -dc)
        elif relative_outcome == 'L': return (-dc, dr) # Rotate left
        elif relative_outcome == 'R': return (dc, -dr) # Rotate right
        else: raise ValueError(f"Unknown relative outcome: {relative_outcome}") # Should not happen

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Initializes self.np_random
        self._agent_coords = self.start_coords
        observation = self.start_state
        info = self._get_info()
        if self.render_mode == "human": self._render_frame()
        return observation, info

    def step(self, action):
        """ Executes potentially multiple unit steps based on the flattened action integer,
            considering action failure, slipperiness, and walls. """
        flat_action = action
        start_state_idx = self._coords_to_state[self._agent_coords]
        if start_state_idx in self.terminal_states:
            return start_state_idx, 0.0, True, False, self._get_info()

        intended_direction, step_size = self._decode_action(flat_action)

        current_coords = self._agent_coords
        terminated = False
        final_reward = 0.0
        n_steps_taken = 0 # In case needed later

        for _ in range(step_size):
            n_steps_taken += 1
            actual_delta = (0, 0)
            action_failed = False

            # 1. Action Failure
            if self.action_failure_prob > 0 and self.np_random.uniform() < self.action_failure_prob:
                action_failed = True
                relative_outcome = random.choices( # Use stdlib random
                    self._failure_outcomes_keys, weights=self._failure_outcomes_probs, k=1
                )[0]
                actual_delta = self._get_relative_delta(intended_direction, relative_outcome)

            # 2. Slipperiness (if action didn't fail)
            if not action_failed:
                if self.is_slippery:
                    p_slip = self.np_random.uniform()
                    if p_slip < 1/3.: actual_delta = self._action_to_delta[intended_direction]
                    elif p_slip < 2/3.: actual_delta = self._action_to_delta[self._perpendicular_actions[intended_direction][0]]
                    else: actual_delta = self._action_to_delta[self._perpendicular_actions[intended_direction][1]]
                else: # Deterministic
                    actual_delta = self._action_to_delta[intended_direction]

            # 3. Calculate Potential Next Coords
            next_row_potential = current_coords[0] + actual_delta[0]
            next_col_potential = current_coords[1] + actual_delta[1]

            # 4. Clamp to Grid Boundaries
            next_row_clamped = max(0, min(next_row_potential, self.rows - 1))
            next_col_clamped = max(0, min(next_col_potential, self.cols - 1))
            next_coords_clamped = (next_row_clamped, next_col_clamped)

            # 5. Check Walls & Boundary Collisions
            if next_coords_clamped in self.wall_coords:
                next_coords = current_coords # Hit wall, stay put
            elif next_row_potential != next_row_clamped or next_col_potential != next_col_clamped:
                 next_coords = current_coords # Hit grid boundary edge, stay put
            else:
                next_coords = next_coords_clamped # Valid move

            # Update position for next iteration/final state
            current_coords = next_coords
            current_state_idx = self._coords_to_state[current_coords]

        # 6. Check Terminal State (Goal/Hole)
        step_reward = self.reward_map['step']
        if current_state_idx == self.goal_state:
            final_reward = self.reward_map['goal']
            terminated = True
        elif current_state_idx in self.hole_states:
            final_reward = self.reward_map['hole']
            terminated = True
        else:
            # Overwrite with step cost for this non-terminal step
            final_reward = step_reward

        # End of multi-step loop
        self._agent_coords = current_coords
        observation = self._coords_to_state[self._agent_coords]
        reward = final_reward
        truncated = False
        info = self._get_info()

        if self.render_mode == "human": self._render_frame()
        return observation, reward, terminated, truncated, info

    def _get_info(self):
        return {"agent_coords": self._agent_coords}

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_text(self):
        outfile = io.StringIO()
        # Use MAP_CHARS for consistency
        chars = {v: k for k, v in self.MAP_CHARS.items()} # Reverse lookup needed? No.

        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                coords = (r, c)
                if coords == self._agent_coords: char = self.MAP_CHARS['START']
                elif coords == self.goal_coords: char = self.MAP_CHARS['GOAL']
                elif coords in self.hole_coords: char = self.MAP_CHARS['HOLE']
                elif coords in self.wall_coords: char = self.MAP_CHARS['WALL']
                else: char = self.MAP_CHARS['EMPTY']
                row_str += char + " "
            outfile.write(row_str.rstrip() + "\n") # Use rstrip for cleaner end-of-line
        # outfile.write("\n") # Remove extra blank line often added

        with closing(outfile): return outfile.getvalue()

    def _render_frame(self):
        if self.render_mode == "human":
            print(self._render_text())
            time.sleep(1.0 / self.metadata["render_fps"])

    def close(self):
        pass