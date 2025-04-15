# algorithms/q_learning.py

import numpy as np
import collections
import random

class QLearningAgent: # Could inherit from BaseAlgorithm
    """ Tabular Q-Learning Agent (Modified Interface) """
    def __init__(
            self, 
            env, 
            terminal_states, 
            gamma, 
            alpha, 
            epsilon_start, 
            epsilon_min, 
            epsilon_decay, **kwargs
        ):
        self.env = env
        # Extract state/action space info from env
        # Assuming Discrete spaces for tabular methods
        try:
            self.n_states = env.observation_space.n
            self.n_actions = env.action_space.n
            self.states = list(range(self.n_states))
            self.actions = list(range(self.n_actions))
        except AttributeError:
             raise TypeError("Q-Learning Agent requires Discrete observation and action spaces.")

        self.terminal_states = terminal_states # Passed explicitly
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = collections.defaultdict(float)

    def choose_action(self, state, evaluate=False):
        """ Choose action using epsilon-greedy policy """
        if state in self.terminal_states: return None # Or a default action if needed

        # During evaluation or if epsilon is zero, act greedily
        if evaluate or random.random() >= self.epsilon:
             # Exploit: choose the best action from Q-table
            q_values = [self.Q.get((state, a), 0.0) for a in self.actions]
            max_q = -np.inf
            if q_values: # Handle empty action space? Unlikely for discrete.
                 max_q = np.max(q_values)

            # Break ties randomly
            best_actions = [a for a, q in zip(self.actions, q_values) if np.isclose(q, max_q)]
            if not best_actions: # Should not happen if actions exist
                 return self.env.action_space.sample() # Fallback to random sample
            return random.choice(best_actions)
        else:
            # Explore: choose a random action
             return self.env.action_space.sample()


    def _learn_step(self, s, a, r, s_prime, done):
        """ Update Q-value based on one step """
        if s in self.terminal_states: return # Cannot learn from terminal state

        max_q_next = 0.0
        if not done and s_prime not in self.terminal_states:
            q_values_next = [self.Q.get((s_prime, next_a), 0.0) for next_a in self.actions]
            if q_values_next:
                 max_q_next = np.max(q_values_next)

        td_target = r + self.gamma * max_q_next
        td_error = td_target - self.Q.get((s, a), 0.0)
        self.Q[(s, a)] = self.Q.get((s, a), 0.0) + self.alpha * td_error

    def train(self, total_episodes, max_steps_per_episode):
        """ Main training loop """
        episode_rewards = []
        all_steps = 0

        print(f"Starting Q-Learning Training for {total_episodes} episodes...")
        progress_interval = total_episodes // 10 if total_episodes >= 10 else 1

        for episode in range(total_episodes):
            current_state, info = self.env.reset()
            episode_reward = 0
            episode_steps = 0

            for step in range(max_steps_per_episode):
                action = self.choose_action(current_state)
                if action is None: # Should only happen if starting in terminal state
                    print(f"Warning: Starting state {current_state} might be terminal or invalid action.")
                    break

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

                # Learn from this step
                self._learn_step(current_state, action, reward, next_state, done)

                current_state = next_state
                episode_steps = step + 1
                all_steps += 1
                if done:
                    break

            # Decay epsilon after each episode
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_rewards.append(episode_reward)

            # Print progress
            if (episode + 1) % progress_interval == 0:
                avg_reward = np.mean(episode_rewards[-progress_interval:])
                print(f"Episode {episode + 1}/{total_episodes} | Avg Reward (last {progress_interval}): {avg_reward:.3f} | Steps: {episode_steps} | Epsilon: {self.epsilon:.4f}")

        print("--- Q-Learning Training Complete ---")
        return {"episode_rewards": episode_rewards, "total_steps": all_steps} # Return training stats

    def save(self, filepath):
        # Example using pickle (can be adapted for other formats)
        import pickle
        with open(filepath, 'wb') as f:
             pickle.dump({"Q": dict(self.Q), "epsilon": self.epsilon}, f)
        print(f"Agent state saved to {filepath}")

    def load(self, filepath):
        import pickle
        try:
             with open(filepath, 'rb') as f:
                 data = pickle.load(f)
                 self.Q = collections.defaultdict(float, data.get("Q", {}))
                 self.epsilon = data.get("epsilon", self.epsilon_min) # Load saved epsilon or use min
             print(f"Agent state loaded from {filepath}")
        except FileNotFoundError:
             print(f"Warning: Could not find agent state file {filepath}. Starting fresh.")
        except Exception as e:
             print(f"Error loading agent state from {filepath}: {e}. Starting fresh.")