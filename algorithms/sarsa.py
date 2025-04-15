import numpy as np
import collections
import random

class SarsaAgent:
    """ Tabular SARSA Agent """
    def __init__(self, states, actions, terminal_states, gamma, alpha, epsilon_start, epsilon_min, epsilon_decay, **kwargs):
        self.states = list(states)
        self.actions = list(actions)
        self.terminal_states = terminal_states
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = collections.defaultdict(float)

    def choose_action(self, state):
        """ Choose action using epsilon-greedy policy """
        if state in self.terminal_states: return None
        if random.random() < self.epsilon:
            return random.choice(self.actions) if self.actions else None
        else:
            q_values = [self.Q.get((state, a), 0.0) for a in self.actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            return random.choice(best_actions) if best_actions else random.choice(self.actions)

    def _learn_step(self, s, a, r, s_prime, a_prime, done):
        """ Update Q-value based on SARSA tuple """
        if s in self.terminal_states: return

        # Get Q value for the *next* state-action pair (a_prime)
        q_next = 0.0
        if not done and s_prime not in self.terminal_states and a_prime is not None:
            q_next = self.Q.get((s_prime, a_prime), 0.0)

        # SARSA update rule
        td_target = r + self.gamma * q_next
        td_error = td_target - self.Q.get((s, a), 0.0)
        self.Q[(s, a)] = self.Q.get((s, a), 0.0) + self.alpha * td_error

    def learn_episode(self, max_steps=100, env_step_func=None, env_reset_func=None):
        """ Run one episode and learn """
        if not env_step_func or not env_reset_func:
            raise ValueError("SARSA agent needs environment functions passed to learn_episode")

        current_state = env_reset_func()
        action = self.choose_action(current_state) # Choose first action
        total_reward = 0

        for step in range(max_steps):
            if action is None: break # Started or landed in terminal state

            next_state, reward, done, info = env_step_func(action)
            total_reward += reward

            # Choose the *next* action based on the *next* state
            next_action = self.choose_action(next_state)

            # Learn from this step (using S, A, R, S', A')
            self._learn_step(current_state, action, reward, next_state, next_action, done)

            current_state = next_state
            action = next_action # Prepare for next iteration

            if done: break

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # print(f"SARSA Episode ended. Reward={total_reward:.2f}, Epsilon={self.epsilon:.3f}")
        return total_reward
