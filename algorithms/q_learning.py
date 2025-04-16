# algorithms/q_learning.py

import numpy as np
import random
from .base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    """ Tabular Q-Learning Agent """
    def __init__(
            self, 
            env=None,
            states=None, 
            actions=None, 
            terminal_states=None, 
            gamma=0.99, 
            alpha=0.1, 
            epsilon_start=1.0, 
            epsilon_min=0.05, 
            epsilon_decay=0.999, 
            **kwargs
        ):
        # Extract state/action space info from env if provided directly
        if env is not None and (states is None or actions is None):
            try:
                states = list(range(env.observation_space.n))
                actions = list(range(env.action_space.n))
            except AttributeError:
                raise TypeError("QLearningAgent requires Discrete observation and action spaces.")
        
        # Initialize the base class
        super().__init__(states, actions, terminal_states, gamma, **kwargs)
        
        # Q-Learning specific parameters
        self.alpha = alpha
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._steps_in_last_episode = 0
        
        # Store environment if provided
        self.env = env

    def choose_action(self, state, evaluate=False):
        """ Choose action using epsilon-greedy policy """
        if state in self.terminal_states: 
            return None
        
        # During evaluation or if epsilon is zero, act greedily
        if evaluate or random.random() >= self.epsilon:
            # Exploit: choose the best action from Q-table
            q_values = [self.Q.get((state, a), 0.0) for a in self.actions]
            max_q = max(q_values) if q_values else 0
            
            # Break ties randomly
            best_actions = [a for a, q in zip(self.actions, q_values) if np.isclose(q, max_q)]
            return random.choice(best_actions) if best_actions else random.choice(self.actions)
        else:
            # Explore: choose a random action
            return random.choice(self.actions)

    def learn_step(self, state, action, reward, next_state, done, **kwargs):
        """ Update Q-value based on one step """
        if state in self.terminal_states:
            return
        
        # Get maximum Q-value for next state
        max_q_next = 0.0
        if not done and next_state not in self.terminal_states:
            q_values_next = [self.Q.get((next_state, next_a), 0.0) for next_a in self.actions]
            if q_values_next:
                max_q_next = max(q_values_next)
        
        # Q-Learning update rule
        current_q = self.Q.get((state, action), 0.0)
        td_target = reward + self.gamma * max_q_next
        td_error = td_target - current_q
        self.Q[(state, action)] = current_q + self.alpha * td_error

    def learn_episode(self, max_steps=100, env_step_func=None, env_reset_func=None):
        """ Run one episode and learn """
        # Call the base class implementation
        total_reward = super().learn_episode(max_steps, env_step_func, env_reset_func)
        
        # Decay epsilon after each episode
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return total_reward
    
    def get_save_data(self):
        """ Get data to save when saving agent """
        data = super().get_save_data()
        data.update({
            "epsilon": self.epsilon,
            "alpha": self.alpha
        })
        return data
    
    def set_load_data(self, data):
        """ Set agent state from loaded data """
        super().set_load_data(data)
        if "epsilon" in data:
            self.epsilon = data["epsilon"]
        if "alpha" in data:
            self.alpha = data["alpha"]