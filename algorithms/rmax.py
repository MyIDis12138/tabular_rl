# algorithms/rmax.py

import numpy as np
import collections
import random
from .base_agent import BaseAgent

class RMaxAgent(BaseAgent):
    """ Tabular R-max Agent """
    def __init__(
            self, 
            env=None,
            states=None, 
            actions=None, 
            terminal_states=None, 
            gamma=0.99, 
            m=10,  # Threshold for known state-actions
            R_max=1.0,  # Optimistic reward for unknown state-actions
            planning_iterations=50,
            planning_tolerance=1e-3,
            **kwargs
        ):
        # Extract state/action space info from env if provided directly
        if env is not None and (states is None or actions is None):
            try:
                states = list(range(env.observation_space.n))
                actions = list(range(env.action_space.n))
            except AttributeError:
                raise TypeError("RMaxAgent requires Discrete observation and action spaces.")
        
        # Initialize the base class
        super().__init__(states, actions, terminal_states, gamma, **kwargs)
        
        # RMax specific parameters
        self.m = m
        self.R_max = R_max
        self.planning_iterations = planning_iterations
        self.planning_tolerance = planning_tolerance
        self._steps_in_last_episode = 0
        
        # Store environment if provided
        self.env = env
        
        # RMax model components
        self.n_sa = collections.defaultdict(int)  # Count of (s,a) occurrences
        self.n_sas = collections.defaultdict(int)  # Count of (s,a,s') occurrences
        self.r_sum_sa = collections.defaultdict(float)  # Sum of rewards for (s,a)
        self.known_sa = set()  # Set of known (s,a) pairs
        self.T = collections.defaultdict(lambda: collections.defaultdict(float))  # Transition model
        self.R = collections.defaultdict(lambda: self.R_max)  # Reward model
        
        # Policy derived from planning
        self.policy = collections.defaultdict(lambda: random.choice(self.actions) if self.actions else None)
        
        # Initialize Q-values optimistically
        optimistic_q_val = self.R_max / (1 - self.gamma) if self.gamma < 1 else self.R_max
        for s in self.states:
            if s in self.terminal_states:
                continue
            for a in self.actions:
                self.Q[(s, a)] = optimistic_q_val

    def _update_model(self, state, action, reward, next_state):
        """ Update the model based on experience """
        if state in self.terminal_states:
            return False
        
        sa_pair = (state, action)
        sas_triple = (state, action, next_state)
        
        # Update counts and reward sum
        self.n_sa[sa_pair] += 1
        self.n_sas[sas_triple] += 1
        self.r_sum_sa[sa_pair] += reward
        
        # Check if this (s,a) pair just became known
        if sa_pair not in self.known_sa and self.n_sa[sa_pair] >= self.m:
            # Update the reward model
            self.R[sa_pair] = self.r_sum_sa[sa_pair] / self.n_sa[sa_pair]
            
            # Update the transition model
            total_transitions = self.n_sa[sa_pair]
            observed_next_states = [
                s_p for (s_obs, a_obs, s_p), count in self.n_sas.items() 
                if s_obs == state and a_obs == action and count > 0
            ]
            
            # Calculate transition probabilities
            current_sas_probs = collections.defaultdict(float)
            prob_sum_check = 0.0
            
            for next_s in observed_next_states:
                prob = self.n_sas[(state, action, next_s)] / total_transitions
                current_sas_probs[next_s] = prob
                prob_sum_check += prob
                
            # Normalize probabilities if needed
            if not np.isclose(prob_sum_check, 1.0):
                if prob_sum_check > 0:
                    scale = 1.0 / prob_sum_check
                    for next_s in current_sas_probs:
                        current_sas_probs[next_s] *= scale
            
            # Update transition model
            self.T[sa_pair] = current_sas_probs
            self.known_sa.add(sa_pair)
            
            return True  # Model was updated
            
        return False  # Model unchanged

    def plan(self):
        """ Run value iteration on the current model """
        Q_new = self.Q.copy()
        optimistic_q_val = self.R_max / (1 - self.gamma) if self.gamma < 1 else self.R_max
        
        for i in range(self.planning_iterations):
            Q_old = Q_new.copy()
            max_diff = 0
            
            for s in self.states:
                if s in self.terminal_states:
                    continue
                    
                for a in self.actions:
                    sa_pair = (s, a)
                    q_val = 0
                    
                    if sa_pair in self.known_sa:
                        # For known state-actions, use learned models
                        reward = self.R[sa_pair]
                        transitions = self.T[sa_pair]
                        
                        # Calculate expected next state value
                        expected_next_val = 0
                        if transitions:
                            for s_prime, prob in transitions.items():
                                if prob > 0:
                                    max_q_s_prime = 0
                                    if s_prime not in self.terminal_states:
                                        max_q_s_prime = max(
                                            Q_old.get((s_prime, next_a), 0.0) 
                                            for next_a in self.actions
                                        ) if self.actions else 0
                                    expected_next_val += prob * max_q_s_prime
                        
                        q_val = reward + self.gamma * expected_next_val
                    else:
                        # For unknown state-actions, use optimistic values
                        q_val = self.R_max + self.gamma * optimistic_q_val
                    
                    Q_new[sa_pair] = q_val
                    max_diff = max(max_diff, abs(Q_new[sa_pair] - Q_old.get(sa_pair, 0.0)))
            
            # Check for convergence
            if max_diff < self.planning_tolerance:
                break
                
        # Update Q-values and policy
        self.Q = Q_new
        
        # Derive policy from Q-values
        for s in self.states:
            if s in self.terminal_states:
                continue
                
            # Find best action for state
            best_action = None
            max_q = float('-inf')
            
            # Shuffle actions to break ties randomly
            shuffled_actions = list(self.actions)
            random.shuffle(shuffled_actions)
            
            for a in shuffled_actions:
                q_s_a = self.Q.get((s, a), float('-inf'))
                if q_s_a > max_q:
                    max_q = q_s_a
                    best_action = a
            
            # Update policy
            self.policy[s] = best_action if best_action is not None else (
                random.choice(self.actions) if self.actions else None
            )

    def choose_action(self, state, evaluate=False):
        """ Choose action according to current policy """
        if state in self.terminal_states:
            return None
            
        action = self.policy.get(state)
        if action is None and self.actions:
            # Fallback to random if policy not defined for this state
            action = random.choice(self.actions)
            
        return action

    def learn_step(self, state, action, reward, next_state, done, **kwargs):
        """ Update model and potentially replan """
        if state in self.terminal_states:
            return
            
        # Update model with new experience
        model_updated = self._update_model(state, action, reward, next_state)
        
        # Replan if model was updated
        if model_updated:
            self.plan()

    def learn_episode(self, max_steps=100, env_step_func=None, env_reset_func=None):
        """ Run one episode and learn """
        if not env_step_func or not env_reset_func:
            raise ValueError("RMax agent needs environment functions passed to learn_episode")
        
        current_state = env_reset_func()
        needs_planning = True  # Plan at the start of each episode
        total_reward = 0
        steps = 0
        
        # Initial planning if needed
        if needs_planning:
            self.plan()
            needs_planning = False
        
        for step in range(max_steps):
            if current_state in self.terminal_states:
                break
                
            action = self.choose_action(current_state)
            if action is None:
                break
            
            try:    
                # Use the wrapper function which should handle all environment variations
                next_state, reward, done, info = env_step_func(action)
                total_reward += reward
                
                # Learn from this experience
                if current_state not in self.terminal_states:
                    model_updated = self._update_model(current_state, action, reward, next_state)
                    if model_updated:
                        self.plan()
                
                current_state = next_state
                steps += 1
                
                if done:
                    break
            except Exception as e:
                print(f"Error in RMax learn_episode step {step}: {e}")
                print(f"Action: {action}, Type: {type(action)}")
                break
            
            if done:
                break
                
        self._steps_in_last_episode = steps
        return total_reward
    
    def get_save_data(self):
        """ Get data to save when saving agent """
        data = super().get_save_data()
        data.update({
            "n_sa": dict(self.n_sa),
            "n_sas": dict(self.n_sas),
            "r_sum_sa": dict(self.r_sum_sa),
            "known_sa": list(self.known_sa),
            "T": {str(k): dict(v) for k, v in self.T.items()},
            "R": dict(self.R),
            "policy": dict(self.policy)
        })
        return data
    
    def set_load_data(self, data):
        """ Set agent state from loaded data """
        super().set_load_data(data)
        
        if "n_sa" in data:
            self.n_sa = collections.defaultdict(int, data["n_sa"])
        if "n_sas" in data:
            self.n_sas = collections.defaultdict(int, data["n_sas"])
        if "r_sum_sa" in data:
            self.r_sum_sa = collections.defaultdict(float, data["r_sum_sa"])
        if "known_sa" in data:
            self.known_sa = set(data["known_sa"])
        if "T" in data:
            # Need to convert string keys back to tuples
            self.T = collections.defaultdict(lambda: collections.defaultdict(float))
            for k_str, v in data["T"].items():
                try:
                    # Parse string representation of tuple
                    k = eval(k_str)
                    self.T[k] = collections.defaultdict(float, v)
                except:
                    print(f"Warning: Could not parse transition key {k_str}")
        if "R" in data:
            self.R = collections.defaultdict(lambda: self.R_max, data["R"])
        if "policy" in data:
            self.policy = collections.defaultdict(
                lambda: random.choice(self.actions) if self.actions else None,
                data["policy"]
            )
