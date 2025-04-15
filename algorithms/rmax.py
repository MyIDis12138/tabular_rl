import numpy as np
import collections
import random

class RMaxAgent:
    """ Tabular R-max Agent """
    def __init__(
            self, 
            states, 
            actions, 
            terminal_states, 
            gamma, 
            m, 
            R_max, 
            env_step_func, 
            env_reset_func, 
            planning_iterations=50, 
            planning_tolerance=1e-3, 
            **kwargs
    ):
        self.states = list(states)
        self.actions = list(actions)
        self.terminal_states = terminal_states
        self.gamma = gamma
        self.m = m
        self.R_max = R_max
        self.env_step = env_step_func
        self.env_reset = env_reset_func
        self.planning_iterations = planning_iterations
        self.planning_tolerance = planning_tolerance

        self.n_sa = collections.defaultdict(int)
        self.n_sas = collections.defaultdict(int)
        self.r_sum_sa = collections.defaultdict(float)
        self.known_sa = set()
        self.T = collections.defaultdict(lambda: collections.defaultdict(float))
        self.R = collections.defaultdict(float)
        self.Q = collections.defaultdict(float)
        self.policy = collections.defaultdict(lambda: random.choice(self.actions) if self.actions else None)

        optimistic_q_val = self.R_max / (1 - self.gamma) if self.gamma < 1 else self.R_max
        for s in self.states:
            if s in self.terminal_states: continue
            for a in self.actions:
                self.R[(s, a)] = self.R_max
                self.Q[(s, a)] = optimistic_q_val

    def _update_model(self, s, a, r, s_prime):
        if s in self.terminal_states: return False
        sa_pair = (s, a)
        sas_triple = (s, a, s_prime)
        self.n_sa[sa_pair] += 1
        self.n_sas[sas_triple] += 1
        self.r_sum_sa[sa_pair] += r

        if sa_pair not in self.known_sa and self.n_sa[sa_pair] >= self.m:
            print(f"INFO: RMAX: State-action pair {sa_pair} became known at count {self.n_sa[sa_pair]}.")
            self.known_sa.add(sa_pair)
            self.R[sa_pair] = self.r_sum_sa[sa_pair] / self.n_sa[sa_pair]
            total_transitions = self.n_sa[sa_pair]
            observed_next_states = [s_p for (s_obs, a_obs, s_p), count in self.n_sas.items() if s_obs == s and a_obs == a and count > 0]
            current_sas_probs = collections.defaultdict(float)
            prob_sum_check = 0.0
            for next_s in observed_next_states:
                 prob = self.n_sas[(s, a, next_s)] / total_transitions
                 current_sas_probs[next_s] = prob
                 prob_sum_check += prob
            if not np.isclose(prob_sum_check, 1.0):
                 print(f"Warning: RMAX: Probabilities for ({s},{a}) sum to {prob_sum_check}. Normalizing.")
                 if prob_sum_check > 0: scale = 1.0 / prob_sum_check
                 else: scale = 0 # Avoid division by zero, though sum shouldn't be 0
                 for next_s in current_sas_probs: current_sas_probs[next_s] *= scale
            self.T[sa_pair] = current_sas_probs
            return True
        return False

    def plan(self):
        print("INFO: RMAX: Planning...")
        Q_new = self.Q.copy()
        optimistic_q_val = self.R_max / (1 - self.gamma) if self.gamma < 1 else self.R_max

        for i in range(self.planning_iterations):
            Q_old = Q_new.copy()
            max_diff = 0
            for s in self.states:
                if s in self.terminal_states: continue
                for a in self.actions:
                    sa_pair = (s, a)
                    q_val = 0
                    if sa_pair in self.known_sa:
                        reward = self.R[sa_pair]
                        transitions = self.T[sa_pair]
                        expected_next_val = 0
                        if transitions:
                            for s_prime, prob in transitions.items():
                                if prob > 0:
                                    max_q_s_prime = 0
                                    if s_prime not in self.terminal_states:
                                         max_q_s_prime = max(Q_old.get((s_prime, next_a), 0.0) for next_a in self.actions) if self.actions else 0
                                    expected_next_val += prob * max_q_s_prime
                        q_val = reward + self.gamma * expected_next_val
                    else:
                        q_val = self.R_max + self.gamma * optimistic_q_val
                    Q_new[sa_pair] = q_val
                    max_diff = max(max_diff, abs(Q_new[sa_pair] - Q_old.get(sa_pair, 0.0)))
            if max_diff < self.planning_tolerance:
                print(f"INFO: RMAX: Planning converged after {i+1} iterations.")
                break
        else: print(f"INFO: RMAX: Planning finished after max {self.planning_iterations} iterations.")
        self.Q = Q_new
        for s in self.states:
            if s not in self.terminal_states:
                best_action = None; max_q = -np.inf
                shuffled_actions = list(self.actions); random.shuffle(shuffled_actions)
                for a in shuffled_actions:
                    q_s_a = self.Q.get((s, a), -np.inf)
                    if q_s_a > max_q: max_q = q_s_a; best_action = a
                self.policy[s] = best_action if best_action is not None else (random.choice(self.actions) if self.actions else None)

    def choose_action(self, state):
        if state in self.terminal_states: return None
        action = self.policy.get(state)
        if action is None and self.actions:
             print(f"Warning: RMAX: Policy not found for non-terminal state {state}. Choosing random action.")
             action = random.choice(self.actions)
        return action

    def learn_episode(self, max_steps=100):
        current_state = self.env_reset()
        needs_planning = True
        steps = 0; total_reward = 0
        for step in range(max_steps):
            if current_state in self.terminal_states: break
            if needs_planning: self.plan(); needs_planning = False
            action = self.choose_action(current_state)
            if action is None: break
            next_state, reward, done, info = self.env_step(action)
            total_reward += reward
            if current_state not in self.terminal_states:
                 model_updated = self._update_model(current_state, action, reward, next_state)
                 if model_updated: needs_planning = True
            current_state = next_state
            steps = step + 1
            if done: break
        # print(f"RMAX Episode ended. Steps={steps}, Reward={total_reward:.2f}")
        return total_reward
