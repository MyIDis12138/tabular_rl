# algorithms/actor_critic.py

import numpy as np
import collections
import random
import math
from .base_agent import BaseAgent

class ActorCriticAgent(BaseAgent):
    """ Tabular Actor-Critic Agent """
    def __init__(
            self, 
            env=None,
            states=None, 
            actions=None, 
            terminal_states=None, 
            gamma=0.99, 
            alpha_actor=0.01, 
            alpha_critic=0.1,
            **kwargs
        ):
        # Extract state/action space info from env if provided directly
        if env is not None and (states is None or actions is None):
            try:
                states = list(range(env.observation_space.n))
                actions = list(range(env.action_space.n))
            except AttributeError:
                raise TypeError("ActorCriticAgent requires Discrete observation and action spaces.")
        
        # Initialize the base class
        super().__init__(states, actions, terminal_states, gamma, **kwargs)
        
        # Actor-Critic specific parameters
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self._steps_in_last_episode = 0
        
        # Store environment if provided
        self.env = env
        
        # Critic: State value function V(s)
        self.V = collections.defaultdict(float)
        
        # Actor: Policy parameters H(s, a) (preferences, often called logits)
        self.H = collections.defaultdict(float)

    def _get_policy_probs(self, state):
        """ Calculate action probabilities using softmax from preferences H(s, a) """
        if state in self.terminal_states: 
            return {}

        prefs = [self.H.get((state, a), 0.0) for a in self.actions]
        max_pref = max(prefs) if prefs else 0  # For numerical stability
        exp_prefs = [math.exp(p - max_pref) for p in prefs]
        sum_exp_prefs = sum(exp_prefs)

        if sum_exp_prefs == 0:  # Avoid division by zero
            # Return uniform if all prefs are -inf or list empty
            return {a: 1.0 / len(self.actions) for a in self.actions} if self.actions else {}

        probs = {a: exp_p / sum_exp_prefs for a, exp_p in zip(self.actions, exp_prefs)}
        return probs

    def choose_action(self, state, evaluate=False):
        """ Choose action by sampling from the current policy """
        if state in self.terminal_states:
            return None

        probs_dict = self._get_policy_probs(state)
        if not probs_dict:
            # Fallback to random choice if no probabilities
            return random.choice(self.actions) if self.actions else None

        # If evaluating, choose most probable action (with ties broken randomly)
        if evaluate:
            max_prob = max(probs_dict.values())
            best_actions = [a for a, p in probs_dict.items() if np.isclose(p, max_prob)]
            return random.choice(best_actions)

        # Sample from the probability distribution
        actions = list(probs_dict.keys())
        probs = list(probs_dict.values())
        
        # Ensure probabilities sum to 1 for numpy choice
        probs_sum = sum(probs)
        if not np.isclose(probs_sum, 1.0):
            if probs_sum > 0:
                probs = [p / probs_sum for p in probs]
            else:
                # Uniform if sum is zero
                probs = [1.0 / len(actions)] * len(actions)

        try:
            return np.random.choice(actions, p=probs)
        except ValueError as e:
            print(f"Error sampling action for state {state}: {e}")
            print(f"Actions: {actions}, Probs: {probs}")
            # Fallback to random
            return random.choice(self.actions) if self.actions else None

    def learn_step(self, state, action, reward, next_state, done, **kwargs):
        """ Update Actor and Critic based on one step """
        if state in self.terminal_states:
            return

        # Calculate TD Error (Critic's job)
        v_s = self.V.get(state, 0.0)
        v_s_prime = 0.0
        if not done and next_state not in self.terminal_states:
            v_s_prime = self.V.get(next_state, 0.0)  # Get V(s') estimate

        td_target = reward + self.gamma * v_s_prime
        td_error = td_target - v_s  # Advantage estimate A(s,a) ~ delta

        # Update Critic (Value function V)
        self.V[state] = v_s + self.alpha_critic * td_error

        # Update Actor (Policy parameters H)
        probs_dict = self._get_policy_probs(state)
        pi_s_a = probs_dict.get(action, 0.0)  # Probability of the chosen action

        # Simpler/more common update: just use td_error as importance weight
        delta_H_sa = self.alpha_actor * td_error
        self.H[(state, action)] = self.H.get((state, action), 0.0) + delta_H_sa

    def get_save_data(self):
        """ Get data to save when saving agent """
        data = super().get_save_data()
        data.update({
            "V": dict(self.V),
            "H": dict(self.H),
            "alpha_actor": self.alpha_actor,
            "alpha_critic": self.alpha_critic
        })
        return data
    
    def set_load_data(self, data):
        """ Set agent state from loaded data """
        super().set_load_data(data)
        
        if "V" in data:
            self.V = collections.defaultdict(float, data["V"])
        if "H" in data:
            self.H = collections.defaultdict(float, data["H"])
        if "alpha_actor" in data:
            self.alpha_actor = data["alpha_actor"]
        if "alpha_critic" in data:
            self.alpha_critic = data["alpha_critic"]