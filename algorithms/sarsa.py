# algorithms/sarsa.py

import numpy as np
import random
from .base_agent import BaseAgent

class SarsaAgent(BaseAgent):
    """ Tabular SARSA Agent """
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
                raise TypeError("SarsaAgent requires Discrete observation and action spaces.")
        
        # Initialize the base class
        super().__init__(states, actions, terminal_states, gamma, **kwargs)
        
        # SARSA specific parameters
        self.alpha = alpha
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._steps_in_last_episode = 0
        
        # Store environment if provided
        self.env = env
        
        # SARSA needs to remember the next action
        self._next_action = None

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

    def learn_step(self, state, action, reward, next_state, done, next_action=None, **kwargs):
        """ Update Q-value based on SARSA tuple """
        if state in self.terminal_states:
            return
        
        # SARSA needs to know the next action
        if next_action is None:
            # If not provided, need to choose next action
            next_action = self.choose_action(next_state) if not done else None
            # Store the next action for learn_episode
            self._next_action = next_action
        
        # Get Q value for the *next* state-action pair
        q_next = 0.0
        if not done and next_state not in self.terminal_states and next_action is not None:
            q_next = self.Q.get((next_state, next_action), 0.0)
        
        # SARSA update rule
        current_q = self.Q.get((state, action), 0.0)
        td_target = reward + self.gamma * q_next
        td_error = td_target - current_q
        self.Q[(state, action)] = current_q + self.alpha * td_error

    def learn_episode(self, max_steps=100, env_step_func=None, env_reset_func=None):
        """ Run one episode and learn (SARSA needs special implementation) """
        if not env_step_func or not env_reset_func:
            raise ValueError("SARSA agent needs environment functions passed to learn_episode")
        
        current_state = env_reset_func()
        # Choose first action
        action = self.choose_action(current_state)
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            if action is None:  # Started in terminal state
                break
                
            try:
                # Use the wrapper function which should handle all environment variations
                next_state, reward, done, info = env_step_func(action)
                total_reward += reward
                
                # Choose the *next* action for SARSA update
                next_action = self.choose_action(next_state) if not done and next_state not in self.terminal_states else None
                
                # Learn from this step (using S, A, R, S', A')
                self.learn_step(current_state, action, reward, next_state, done, next_action=next_action)
                
                current_state = next_state
                action = next_action  # Use the next action for the next iteration
                steps += 1
            except Exception as e:
                print(f"Error in SARSA learn_episode step {step}: {e}")
                print(f"Action: {action}, Type: {type(action)}")
                break
            
            if done:
                break
                
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._steps_in_last_episode = steps
        
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