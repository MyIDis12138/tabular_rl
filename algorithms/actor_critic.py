import numpy as np
import collections
import random
import math

class ActorCriticAgent:
    """ Simple Tabular Actor-Critic Agent """
    def __init__(self, states, actions, terminal_states, gamma, alpha_actor, alpha_critic, **kwargs):
        self.states = list(states)
        self.actions = list(actions)
        self.terminal_states = terminal_states
        self.gamma = gamma
        self.alpha_actor = alpha_actor # Learning rate for actor (policy)
        self.alpha_critic = alpha_critic # Learning rate for critic (value function)

        # Critic: State value function V(s)
        self.V = collections.defaultdict(float)
        # Actor: Policy parameters H(s, a) (preferences, often called logits)
        self.H = collections.defaultdict(float)

    def _get_policy_probs(self, state):
        """ Calculate action probabilities using softmax from preferences H(s, a) """
        if state in self.terminal_states: return {}

        prefs = [self.H.get((state, a), 0.0) for a in self.actions]
        max_pref = max(prefs) if prefs else 0 # For numerical stability
        exp_prefs = [math.exp(p - max_pref) for p in prefs]
        sum_exp_prefs = sum(exp_prefs)

        if sum_exp_prefs == 0: # Avoid division by zero, return uniform if all prefs are -inf or list empty
            return {a: 1.0 / len(self.actions) for a in self.actions} if self.actions else {}

        probs = {a: exp_p / sum_exp_prefs for a, exp_p in zip(self.actions, exp_prefs)}
        return probs

    def choose_action(self, state):
        """ Choose action by sampling from the current policy """
        if state in self.terminal_states: return None

        probs_dict = self._get_policy_probs(state)
        if not probs_dict: return random.choice(self.actions) if self.actions else None # Fallback

        actions = list(probs_dict.keys())
        probs = list(probs_dict.values())
        # Ensure probabilities sum to 1 for numpy choice
        probs_sum = sum(probs)
        if not np.isclose(probs_sum, 1.0):
            print(f"Warning: AC Probs for state {state} sum to {probs_sum}. Renormalizing.")
            if probs_sum > 0: probs = [p / probs_sum for p in probs]
            else: probs = [1.0 / len(actions)] * len(actions) # Uniform if sum is zero

        try:
            return np.random.choice(actions, p=probs)
        except ValueError as e:
             print(f"Error sampling action for state {state}: {e}")
             print(f"Actions: {actions}, Probs: {probs}")
             return random.choice(self.actions) if self.actions else None # Fallback


    def _learn_step(self, s, a, r, s_prime, done):
        """ Update Actor and Critic based on one step """
        if s in self.terminal_states: return

        # Calculate TD Error (Critic's job)
        v_s = self.V.get(s, 0.0)
        v_s_prime = 0.0
        if not done and s_prime not in self.terminal_states:
            v_s_prime = self.V.get(s_prime, 0.0) # Get V(s') estimate

        td_target = r + self.gamma * v_s_prime
        td_error = td_target - v_s # Advantage estimate A(s,a) ~ delta

        # Update Critic (Value function V)
        self.V[s] = v_s + self.alpha_critic * td_error

        # Update Actor (Policy parameters H)
        probs_dict = self._get_policy_probs(s)
        pi_s_a = probs_dict.get(a, 0.0) # Probability of the chosen action

        # Update using log-likelihood gradient * advantage (TD error)
        # Grad log pi(a|s; H) for H(s,a') is 1{a'=a} - pi(a'|s)
        # Update for the chosen action H(s, a)
        # delta_H_sa = self.alpha_actor * td_error * (1 - pi_s_a) # Less common/stable
        # Simpler/more common update: just use td_error as importance weight
        delta_H_sa = self.alpha_actor * td_error
        self.H[(s, a)] = self.H.get((s, a), 0.0) + delta_H_sa

        # Optionally update H for other actions a' != a (less common in simple implementations)
        # for other_a in self.actions:
        #     if other_a != a:
        #         pi_s_other_a = probs_dict.get(other_a, 0.0)
        #         delta_H_s_other_a = self.alpha_actor * td_error * (- pi_s_other_a)
        #         self.H[(s, other_a)] = self.H.get((s, other_a), 0.0) + delta_H_s_other_a


    def learn_episode(self, max_steps=100, env_step_func=None, env_reset_func=None):
        """ Run one episode and learn """
        if not env_step_func or not env_reset_func:
            raise ValueError("Actor-Critic agent needs environment functions passed to learn_episode")

        current_state = env_reset_func()
        total_reward = 0
        for step in range(max_steps):
            action = self.choose_action(current_state)
            if action is None: break # Terminal state

            next_state, reward, done, info = env_step_func(action)
            total_reward += reward

            # Learn from this step
            self._learn_step(current_state, action, reward, next_state, done)

            current_state = next_state
            if done: break

        # print(f"ActorCritic Episode ended. Reward={total_reward:.2f}")
        return total_reward
