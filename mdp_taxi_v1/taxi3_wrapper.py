import gym
import numpy as np
class TaxiMDPWrapper:
    def __init__(self, env_name='Taxi-v3'):
        self.env = gym.make(env_name)
        self.env.seed(42)
        self.states = list(range(self.env.observation_space.n))
        self.actions = list(range(self.env.action_space.n))

        self.transitions = self.env.P

    def get_all_states(self):
        return [state for state in self.states]

    def get_possible_actions(self, state):
        actions = [i for i in range(len(self.actions)) if self.env.action_mask(state)[i] > 0]
        return actions

    def get_next_states(self, state, action):
        p, state_next, _, _ = self.transitions[state][action][0]
        return {state_next: 1.0}

    def get_reward(self, state, action, state_next):
        _, _, reward, _ = self.transitions[state][action][0]
        return reward

    def get_transition_prob(self, state, action, state_next):
        p, next_state, _, _ = self.transitions[state][action][0]
        return p

    def is_terminal(self, state):
       done = self.transitions[state][0][0][3]
       return done

