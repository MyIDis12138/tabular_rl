import random
import pickle
import collections
import numpy as np
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for tabular reinforcement learning agents.
    
    This class defines the common interface that all agent implementations
    should follow to ensure consistency across the codebase.
    """
    
    def __init__(self, 
                 states, 
                 actions, 
                 terminal_states, 
                 gamma,
                 **kwargs):
        """
        Initialize the base agent with common parameters.
        
        Args:
            states: List of all possible states in the environment
            actions: List of all possible actions in the environment
            terminal_states: Set of terminal states where episodes end
            gamma: Discount factor for future rewards
            **kwargs: Additional algorithm-specific parameters
        """
        self.states = list(states)
        self.actions = list(actions)
        self.terminal_states = terminal_states
        self.gamma = gamma
        
        # Common attributes that may be overridden by subclasses
        self.Q = collections.defaultdict(float)  # Q-values (state, action) -> value
        self._steps_in_last_episode = 0  # Track steps per episode for reporting
    
    @abstractmethod
    def choose_action(self, state, evaluate=False):
        """
        Select an action for the given state.
        
        Args:
            state: The current environment state
            evaluate: If True, act greedily (evaluation mode). If False, may explore
            
        Returns:
            The selected action or None if in a terminal state
        """
        pass
    
    @abstractmethod
    def learn_step(self, state, action, reward, next_state, done, **kwargs):
        """
        Update agent's knowledge from a single environment step.
        
        Args:
            state: The current state
            action: The action taken
            reward: The reward received
            next_state: The state transitioned to
            done: Whether this transition resulted in a terminal state
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            None
        """
        pass
    
    def learn_episode(self, max_steps=100, env_step_func=None, env_reset_func=None):
        """
        Run a complete episode, learning from each step.
        
        Args:
            max_steps: Maximum number of steps per episode
            env_step_func: Function to step in environment: (action) -> (next_state, reward, done, info)
            env_reset_func: Function to reset environment: () -> initial_state
            
        Returns:
            Total episode reward
        """
        if not env_step_func or not env_reset_func:
            raise ValueError(f"{self.__class__.__name__} needs environment functions passed to learn_episode")
        
        current_state = env_reset_func()
        total_reward = 0
        steps_taken = 0
        
        for step in range(max_steps):
            action = self.choose_action(current_state)
            if action is None:  # Terminal state
                break
                
            # Use the wrapper function which should return (next_state, reward, done, info)
            next_state, reward, done, info = env_step_func(action)
            total_reward += reward
            steps_taken += 1
            
            # Learn from this step
            self.learn_step(current_state, action, reward, next_state, done)
            
            current_state = next_state
            if done:
                break
        
        # Store steps for reporting
        self._steps_in_last_episode = steps_taken
                
        return total_reward
    
    def train(self, total_episodes, max_steps_per_episode, env=None):
        """
        Train the agent for a specified number of episodes.
        
        Args:
            total_episodes: Number of episodes to train for
            max_steps_per_episode: Maximum steps per episode
            env: The environment to train in
            
        Returns:
            Dictionary of training statistics
        """
        if env is None:
            raise ValueError("Environment must be provided for training")
            
        # Create environment step and reset functions
        def env_step_func(action):
            result = env.step(action)
            if len(result) == 5:  # Gym v26+ returns (obs, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
                return next_state, reward, done, info
            else:  # Backwards compatibility
                next_state, reward, done, info = result
                return next_state, reward, done, info
                
        def env_reset_func():
            result = env.reset()
            if isinstance(result, tuple):  # Gym v26+ returns (obs, info)
                return result[0]
            else:  # Backwards compatibility
                return result
        
        episode_rewards = []
        all_steps = 0
        
        print(f"Starting {self.__class__.__name__} training for {total_episodes} episodes...")
        progress_interval = max(1, total_episodes // 10)  # Report progress at 10% intervals
        
        for episode in range(total_episodes):
            # Run one learning episode
            episode_reward = self.learn_episode(
                max_steps=max_steps_per_episode,
                env_step_func=env_step_func,
                env_reset_func=env_reset_func
            )
            
            episode_rewards.append(episode_reward)
            all_steps += min(max_steps_per_episode, self._last_episode_steps)
            
            # Print progress at intervals
            if (episode + 1) % progress_interval == 0:
                avg_reward = np.mean(episode_rewards[-progress_interval:])
                print(f"Episode {episode + 1}/{total_episodes} | "
                      f"Avg Reward (last {progress_interval}): {avg_reward:.3f} | "
                      f"Steps: {self._last_episode_steps}")
        
        print(f"--- {self.__class__.__name__} Training Complete ---")
        return {"episode_rewards": episode_rewards, "total_steps": all_steps}
    
    def save(self, filepath):
        """
        Save the agent's state to a file.
        
        Args:
            filepath: Path to save the agent state to
            
        Returns:
            None
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.get_save_data(), f)
        print(f"Agent state saved to {filepath}")
    
    def load(self, filepath):
        """
        Load the agent's state from a file.
        
        Args:
            filepath: Path to load the agent state from
            
        Returns:
            None
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.set_load_data(data)
            print(f"Agent state loaded from {filepath}")
        except FileNotFoundError:
            print(f"Warning: Could not find agent state file {filepath}. Starting fresh.")
        except Exception as e:
            print(f"Error loading agent state from {filepath}: {e}. Starting fresh.")
    
    def get_save_data(self):
        """
        Get data to save when saving agent.
        Override in subclasses if needed.
        
        Returns:
            Dictionary of data to save
        """
        return {"Q": dict(self.Q)}
    
    def set_load_data(self, data):
        """
        Set agent state from loaded data.
        Override in subclasses if needed.
        
        Args:
            data: Dictionary of loaded data
            
        Returns:
            None
        """
        self.Q = collections.defaultdict(float, data.get("Q", {}))
    
    @property
    def _last_episode_steps(self):
        """
        Helper method to store and retrieve the number of steps in the last episode.
        This is used for reporting during training.
        
        Returns:
            Number of steps in the last episode, or 0 if not tracked
        """
        if not hasattr(self, "_steps_in_last_episode"):
            return 0
        return self._steps_in_last_episode