import gymnasium as gym
from gymnasium import spaces
import numpy as np
import io
from contextlib import closing
import pygame # Import pygame

class CustomGridWorld(gym.Env):
    """
    Custom Grid World Environment for Reinforcement Learning.

    Follows the Gymnasium API.
    Includes separate mechanisms for internal noise and action failure:
    - Internal Noise: Gaussian noise (if enabled via apply_noise) is added
      to the intended movement delta *before* checking for failure.
    - Action Failure: With probability action_failure_prob, the noisy intended
      action is overridden. The actual outcome (Stay, Left, Right, Back) is
      determined by action_failure_outcomes, resulting in a single unit step.

    Includes Pygame-based rendering for 'human' mode with cell margins.
    Step logic uses single-action multi-unit moves with path checking.

    Action Space: Discrete(4 * max_step_size)
    Observation Space: Discrete(rows * cols)
    """
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 10}

    MAP_CHARS = {
        'START': 'A',
        'GOAL': 'G',
        'HOLE': 'H',
        'WALL': '#',
        'EMPTY': '.'
    }

    DEFAULT_COLORS = {
        'A': (0, 0, 255),    # Agent (Blue)
        'G': (0, 255, 0),    # Goal (Green)
        'H': (0, 0, 0),      # Hole (Black) - Note: Same as default margin background
        '#': (188, 74,  60), # Wall (Brownish)
        '.': (41,  38,  38)  # Empty Floor (Dark Gray)
    }
    BACKGROUND_COLOR = (0, 0, 0) # Black background for margins

    def __init__(self,
                 map_layout=None,
                 rows=None,
                 cols=None,
                 start_coords=None,
                 goal_coords=None,
                 hole_coords=None,
                 wall_coords=None,
                 reward_map=None,
                 action_failure_prob=0.0,    # Probability of failure
                 action_failure_outcomes=None,# Dict for L/R/B/S outcomes on failure
                 apply_noise=True,           # Control internal noise application
                 noise_stddev=1.0,           # Std dev for internal Gaussian noise
                 max_step_size=1,
                 render_mode=None,
                 cell_size=50,
                 cell_margin_ratio=0.1):
        """
        Initializes the Custom Grid World.

        Args:
            map_layout (list[str], optional): Map definition.
            rows (int, optional): Number of rows.
            cols (int, optional): Number of columns.
            start_coords (tuple, optional): Start coordinates (row, col).
            goal_coords (tuple, optional): Goal coordinates (row, col).
            hole_coords (set | list, optional): Set/List of hole coordinates.
            wall_coords (set | list, optional): Set/List of wall coordinates.
            reward_map (dict, optional): Rewards for 'goal', 'hole', 'step'.
            action_failure_prob (float): Probability (0.0 to 1.0) that the action fails.
            action_failure_outcomes (dict, optional): Probabilities for outcomes
                                         ('S', 'L', 'R', 'B') relative to the
                                         *intended* direction when action fails.
                                         Defaults provided if failure_prob > 0.
                                         Resulting move is only 1 unit step.
            apply_noise (bool): If True, Gaussian noise (noise_stddev) is added
                                to the *intended* delta before failure check.
            noise_stddev (float): Standard deviation of Gaussian noise added to
                                 each component (row, col) of the intended unit delta.
            max_step_size (int): Maximum number of unit steps per successful action.
            render_mode (str | None): Rendering mode ('human', 'ansi', 'rgb_array').
            cell_size (int): Size of each cell in pixels for rendering.
            cell_margin_ratio (float): Ratio of cell_size for margin (0.0 to <0.5).
        """
        super().__init__()

        # --- Basic Parameter Setup ---
        assert max_step_size >= 1, "max_step_size must be at least 1"
        assert 0.0 <= action_failure_prob <= 1.0, "action_failure_prob must be between 0.0 and 1.0"
        assert 0.0 <= cell_margin_ratio < 0.5, "cell_margin_ratio must be between 0.0 and <0.5"
        assert noise_stddev >= 0.0, "noise_stddev cannot be negative"

        self.action_failure_prob = action_failure_prob
        self.action_failure_outcomes = action_failure_outcomes # Reintroduced
        self.apply_noise = apply_noise
        self.noise_stddev = noise_stddev
        self.max_step_size = max_step_size
        self.render_mode = render_mode

        # --- Determine Grid Configuration ---
        if map_layout is not None:
            self._init_from_map(map_layout)
            if any(p is not None for p in [rows, cols, start_coords, goal_coords, hole_coords, wall_coords]):
                 print("Warning: map_layout provided. Explicit coordinate/size parameters are ignored.")
        else:
            self._init_from_params(rows, cols, start_coords, goal_coords, hole_coords, wall_coords)

        # --- Common Initialization Logic ---
        self.size = self.rows * self.cols
        if reward_map is None:
            self.reward_map = {'goal': 1.0, 'hole': -1.0, 'step': -0.01}
        else:
            self.reward_map = reward_map

        self._validate_configuration()
        self._setup_action_failure() # Added back
        self._initialize_spaces_and_mappings()

        self._agent_coords = None
        self.np_random = None # Initialized in reset()

        # --- Pygame Specific Initialization ---
        self.cell_size = cell_size
        self.margin_pixels = 0
        if cell_margin_ratio > 0:
             self.margin_pixels = max(1, int(self.cell_size * cell_margin_ratio))
        if self.cell_size - 2 * self.margin_pixels <= 0:
             print(f"Warning: Calculated margin ({self.margin_pixels}px) is too large for cell size ({self.cell_size}px). Setting margin to 0.")
             self.margin_pixels = 0
        self.window_width = self.cols * self.cell_size
        self.window_height = self.rows * self.cell_size
        self.screen = None
        self.clock = None

    # --- Initialization Helpers ---
    # _init_from_map, _init_from_params, _validate_configuration unchanged
    # ... [Copy _init_from_map here] ...
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

    # ... [Copy _init_from_params here] ...
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

    # ... [Copy _validate_configuration here] ...
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


    # _setup_action_failure (Added Back)
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
            self._failure_outcomes_keys = None
            self._failure_outcomes_probs = None

    # _initialize_spaces_and_mappings unchanged
    # ... [Copy _initialize_spaces_and_mappings here] ...
    def _initialize_spaces_and_mappings(self):
        """ Helper method to initialize spaces, dictionaries and derived state sets. """
        self.observation_space = spaces.Discrete(self.size)
        self.action_space = spaces.Discrete(4 * self.max_step_size)
        self.num_directions = 4

        # Mapping: 0: Up (-1, 0), 1: Down (1, 0), 2: Left (0, -1), 3: Right (0, 1)
        self._action_to_delta = { 0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1) }

        self._state_to_coords = {i: (r, c) for i, (r, c) in enumerate(np.ndindex(self.rows, self.cols))}
        self._coords_to_state = {v: k for k, v in self._state_to_coords.items()}

        self.start_state = self._coords_to_state[self.start_coords]
        self.goal_state = self._coords_to_state[self.goal_coords]
        self.hole_states = {self._coords_to_state[hc] for hc in self.hole_coords}
        self.wall_states = {self._coords_to_state[wc] for wc in self.wall_coords}
        self.terminal_states = self.hole_states.union({self.goal_state})

    # --- Graphical Rendering Method ---
    # render_grid_graphically remains unchanged
    # ... [Copy render_grid_graphically here] ...
    def render_grid_graphically(self, screen, cell_size):
        """ Renders the grid onto a given Pygame screen surface, including margins. """
        if self._agent_coords is None: return

        for r in range(self.rows):
            for c in range(self.cols):
                coords = (r, c)
                # Calculate the position and size of the inner rectangle (cell content)
                inner_rect_x = c * self.cell_size + self.margin_pixels
                inner_rect_y = r * self.cell_size + self.margin_pixels
                inner_rect_width = self.cell_size - 2 * self.margin_pixels
                inner_rect_height = self.cell_size - 2 * self.margin_pixels

                # Ensure width/height are not negative
                inner_rect_width = max(0, inner_rect_width)
                inner_rect_height = max(0, inner_rect_height)

                inner_rect = pygame.Rect(inner_rect_x, inner_rect_y, inner_rect_width, inner_rect_height)

                char = '.' # Default character
                if coords == self.goal_coords: char = 'G'
                elif coords in self.hole_coords: char = 'H'
                elif coords in self.wall_coords: char = '#'
                if coords == self._agent_coords: char = 'A'

                color = self.DEFAULT_COLORS.get(char, (255, 255, 255)) # Default white

                # Draw the inner rectangle with the determined color
                pygame.draw.rect(screen, color, inner_rect)

    # --- Core Gym Methods ---
    # reset remains unchanged
    # ... [Copy reset here] ...
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Initializes self.np_random
        self._agent_coords = self.start_coords
        observation = self.start_state
        info = self._get_info()

        # Render the initial state if requested
        if self.render_mode == "human":
            self.render()

        return observation, info

    # step (MODIFIED for Noise and Failure Outcomes)
    def step(self, action):
        """
        Executes a single action. Internal noise (if enabled) is applied to the
        intended delta first. Then, action failure is checked. If failed, uses
        action_failure_outcomes for a single step. If successful, uses the noisy
        intended delta for the original step_size. Stops at walls/boundaries.
        """
        flat_action = action
        if self._agent_coords is None:
            raise RuntimeError("Agent coordinates are not set. Call reset() before step().")

        start_state_idx = self._coords_to_state[self._agent_coords]
        if start_state_idx in self.terminal_states:
            return start_state_idx, 0.0, True, False, self._get_info()

        intended_direction, step_size = self._decode_action(flat_action)
        current_coords = self._agent_coords

        # --- Calculate Noisy Intended Delta ---
        base_unit_delta = self._action_to_delta[intended_direction]
        noisy_unit_delta = base_unit_delta # Start with clean delta

        if self.apply_noise and self.noise_stddev > 0:
            if self.np_random is None: self.reset() # Safety check
            noise_r = self.np_random.normal(0, self.noise_stddev)
            noise_c = self.np_random.normal(0, self.noise_stddev)
            noisy_unit_delta = (base_unit_delta[0] + noise_r, base_unit_delta[1] + noise_c)

        # --- Check for Action Failure ---
        effective_delta = noisy_unit_delta # Assume success initially
        effective_step_size = step_size
        action_failed = False

        if self.action_failure_prob > 0 and self.np_random.uniform() < self.action_failure_prob:
            action_failed = True
            # Determine failure outcome (L/R/B/S)
            relative_outcome = self.np_random.choice(
                self._failure_outcomes_keys, p=self._failure_outcomes_probs
            )
            # Get the *clean* delta for the failure outcome (L/R/B/S)
            # Note: Uses original intended_direction for reference
            effective_delta = self._get_relative_delta(intended_direction, relative_outcome)
            effective_step_size = 1 # Failure outcome is always a single step attempt

        # --- Find Endpoint ---
        # Uses noisy delta & original steps on success
        # Uses L/R/B/S delta & 1 step on failure
        next_coords = self._find_path_endpoint(current_coords, effective_delta, effective_step_size)

        # --- Update State and Calculate Reward ---
        self._agent_coords = next_coords
        observation = self._coords_to_state[self._agent_coords]
        terminated = False
        reward = 0.0

        if observation == self.goal_state:
            reward = self.reward_map['goal']
            terminated = True
        elif observation in self.hole_states:
            reward = self.reward_map['hole']
            terminated = True
        elif next_coords != current_coords: # Moved and didn't terminate
             reward = self.reward_map['step'] # Apply step cost only if moved
        # else: stayed put -> reward 0

        truncated = False
        info = self._get_info()
        info['action_failed'] = action_failed # Add failure status

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    # --- Action Decoding and Path Helpers ---
    # _decode_action remains unchanged
    # ... [Copy _decode_action here] ...
    def _decode_action(self, flat_action):
        """ Decodes a flattened action integer into direction and step_size. """
        if not (0 <= flat_action < self.action_space.n):
                raise ValueError(f"Invalid flat_action: {flat_action}. Must be 0 <= action < {self.action_space.n}")
        direction = flat_action % self.num_directions
        step_size = flat_action // self.num_directions + 1
        return direction, step_size

    # _get_relative_delta (Added Back)
    def _get_relative_delta(self, intended_direction, relative_outcome):
        """
        Calculates the single-step delta for a relative outcome ('S', 'L', 'R', 'B')
        based on the original intended direction. Used for action failure outcomes.
        """
        intended_delta = self._action_to_delta[intended_direction]
        dr, dc = intended_delta

        if relative_outcome == 'S': return (0, 0)      # Stay
        elif relative_outcome == 'B': return (-dr, -dc) # Back
        elif relative_outcome == 'L': return (-dc, dr)  # Rotate left 90 deg
        elif relative_outcome == 'R': return (dc, -dr)  # Rotate right 90 deg
        else: raise ValueError(f"Unknown relative outcome: {relative_outcome}")

    # _find_path_endpoint remains unchanged
    # ... [Copy _find_path_endpoint here] ...
    def _find_path_endpoint(self, start_coords, unit_delta, max_steps):
        """
        Finds the final reachable grid cell by attempting to move along the
        (potentially float) unit_delta direction for max_steps, rounding
        at each step and stopping at walls or boundaries.

        Args:
            start_coords (tuple): The starting (row, col).
            unit_delta (tuple): The potentially float change in (row, col)
                                for one conceptual step.
            max_steps (int): The maximum number of steps to attempt.

        Returns:
            tuple: The final reachable (row, col) integer coordinate.
        """
        current_valid_coords = start_coords

        # If delta is effectively zero, no need to check path
        # Use a small threshold for float comparison
        if abs(unit_delta[0]) < 1e-9 and abs(unit_delta[1]) < 1e-9:
             return start_coords

        for step_num in range(max_steps):
            # Calculate the float target coordinates for *this* step relative to start
            # Add a small epsilon to handle rounding issues near 0.5? Maybe not needed yet.
            target_r_float = start_coords[0] + unit_delta[0] * (step_num + 1)
            target_c_float = start_coords[1] + unit_delta[1] * (step_num + 1)

            # Round to the nearest grid cell for checking
            target_cell_r = int(round(target_r_float))
            target_cell_c = int(round(target_c_float))
            target_cell_coords = (target_cell_r, target_cell_c)

            # Check if the target cell is the same as the last one due to small delta/rounding
            if target_cell_coords == current_valid_coords:
                 continue # Continue loop in case later steps cross boundary

            # Check boundaries for the target cell
            if not (0 <= target_cell_r < self.rows and 0 <= target_cell_c < self.cols):
                return current_valid_coords # Stop before hitting boundary

            # Check walls for the target cell
            if target_cell_coords in self.wall_coords:
                return current_valid_coords # Stop before hitting wall

            # If the target cell is valid, update the last known valid position
            current_valid_coords = target_cell_coords

        # If loop completes, the full path was clear
        return current_valid_coords


    # --- Info Getter ---
    # _get_info remains unchanged
    # ... [Copy _get_info here] ...
    def _get_info(self):
        return {"agent_coords": self._agent_coords}

    # --- Rendering Methods ---
    # render, _render_text, _init_pygame, _render_frame, _render_rgb_array
    # remain unchanged from the previous version.
    # ... [Copy render here] ...
    def render(self):
        """ Renders the environment based on the specified render_mode. """
        if self.render_mode == "ansi":
            return self._render_text()
        elif self.render_mode == "human":
            return self._render_frame()
        elif self.render_mode == "rgb_array":
             return self._render_rgb_array() # Added for compatibility
    # ... [Copy _render_text here] ...
    def _render_text(self):
        """ Renders the grid as text to an ANSI string. """
        # Ensure agent coords are valid before rendering
        if self._agent_coords is None: return "Agent position not set. Call reset()."

        outfile = io.StringIO()
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                coords = (r, c)
                char = self.MAP_CHARS['EMPTY'] # Default
                if coords == self.goal_coords: char = self.MAP_CHARS['GOAL']
                elif coords in self.hole_coords: char = self.MAP_CHARS['HOLE']
                elif coords in self.wall_coords: char = self.MAP_CHARS['WALL']
                # Agent drawn last
                if coords == self._agent_coords: char = self.MAP_CHARS['START']

                row_str += char + " "
            outfile.write(row_str.rstrip() + "\n")

        with closing(outfile): return outfile.getvalue()
        
    def _init_pygame(self):
        """ Initializes Pygame display if not already done. """
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Custom Grid World")
            if self.clock is None:
                self.clock = pygame.time.Clock()

    def _render_frame(self):
        """ Renders the current frame using Pygame for 'human' mode. """
        if self.render_mode != "human":
            print(f"Warning: _render_frame called with render_mode='{self.render_mode}'. Expected 'human'.")
            return

        self._init_pygame() # Ensure Pygame is initialized

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return # Stop rendering if closing

        # --- Drawing ---
        # Fill the background (this will be the color of the margins)
        self.screen.fill(self.BACKGROUND_COLOR)

        # Draw the grid cells (now with margins)
        self.render_grid_graphically(self.screen, self.cell_size)

        # --- Update Display ---
        pygame.display.flip()

        # --- Control Frame Rate ---
        self.clock.tick(self.metadata["render_fps"])
    # ... [Copy _render_rgb_array here] ...
    def _render_rgb_array(self):
        """ Renders the current state as an RGB array. """
        self._init_pygame() # Ensure Pygame is initialized

        # Fill background and draw grid
        self.screen.fill(self.BACKGROUND_COLOR)
        self.render_grid_graphically(self.screen, self.cell_size)

        # Get pixel data
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    # --- Cleanup Method ---
    # close remains unchanged
    # ... [Copy close here] ...
    def close(self):
        """ Cleans up resources, including quitting Pygame if initialized. """
        if self.screen is not None:
            print("Closing Pygame display.")
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

# Example Usage (Optional - for testing the environment directly)
if __name__ == '__main__':
    custom_map = [
      "A...#.....",
      ".##..H.#..",
      "H.#...#.H.",
      ".#.#H##...",
      ".##..#GH#H",
      "....#....."
    ]
    # Example failure outcomes: higher chance to go left/right on failure
    failure_outcomes = {'S': 0.1, 'L': 0.4, 'R': 0.4, 'B': 0.1}

    env = CustomGridWorld(map_layout=custom_map,
                          render_mode='human',
                          cell_size=60,
                          max_step_size=2,
                          action_failure_prob=0.15, # Example failure prob
                          action_failure_outcomes=failure_outcomes, # Use custom outcomes
                          apply_noise=True,        # Apply internal noise
                          noise_stddev=0.1         # Internal noise level
                         )
    observation, info = env.reset()

    total_reward = 0
    steps = 0
    max_steps = 100

    key_to_action = {
        pygame.K_UP: 0, pygame.K_DOWN: 1, pygame.K_LEFT: 2, pygame.K_RIGHT: 3, # Step size 1
        pygame.K_KP8: 4, pygame.K_KP2: 5, pygame.K_KP4: 6, pygame.K_KP6: 7, # Step size 2 (Numpad)
        pygame.K_w: 0, pygame.K_s: 1, pygame.K_a: 2, pygame.K_d: 3, # WASD size 1
    }
    print("Manual Control:")
    print("  Arrows/WASD: Move 1 step (with noise/failure)")
    print("  Numpad 8/2/4/6: Move 2 steps (with noise/failure, if max_step_size >= 2)")
    print("  ESC: Quit")

    pygame_running = True
    while pygame_running and steps < max_steps:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame_running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame_running = False
                elif event.key in key_to_action:
                     potential_action = key_to_action[event.key]
                     _dir, step_s = env._decode_action(potential_action)
                     if step_s <= env.max_step_size:
                          action = potential_action
                     else:
                          print(f"Action {potential_action} (step size {step_s}) exceeds max_step_size ({env.max_step_size})")

        if action is not None and pygame_running:
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            _dir, step_s = env._decode_action(action)
            fail_status = "Failed" if info.get('action_failed', False) else "Success"
            print(f"Step: {steps}, Action: {action} (Dir: {_dir}, Size: {step_s}), Status: {fail_status}, Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}, Coords: {info['agent_coords']}")


            if terminated or truncated:
                print(f"Episode finished after {steps} steps. Total reward: {total_reward:.2f}")
                pygame_running = False # Stop after one episode

        if env.screen is None:
             pygame_running = False

    env.close()
    print("Environment closed.")

