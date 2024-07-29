
import numpy as np


class OnPolicyMonteCarloRaceTrack():

    def __init__(self, environment_states):
        self.random_generator = np.random.default_rng(seed=None)
        self.epsilon = 0.2
        self.discount = 1
        self.max_velocity = 5
        self.actions = self.generate_actions_permutations()
        self.q = self.generate_empty_q_state_space(
            environment_states, self.max_velocity, self.actions)
        self.returns = {}  # (row,col,action) -> []

    def every_visit_update(self, episode):

        G = 0

        for step in episode[::-1]:
            (state, action, reward) = step
            G = self.discount*G + reward
            self.update_returns(state, action, G)
            self.update_q(state, action)

    def get_action_index(self, action):
        for index, a in enumerate(self.actions):
            if a == action:
                return index
        raise IndexError()

    def update_returns(self, state, action, G):

        if self.returns.get((state, action)) is None:
            self.returns[(state, action)] = (0, 0)  # (avg, count)

        (avg, count) = self.returns[(state, action)]
        self.returns[(state, action)] = (
            avg + (1/(count+1))*(G - avg), count + 1)

    def update_q(self, state, action):
        index = (state) + (self.get_action_index(action),)
        average = self.returns[(state, action)][0]
        self.q[index] = average

    def policy(self, state, explore=True):
        if self.random_generator.random() < self.epsilon and explore:
            return self.random_generator.choice(len(self.actions)), self.actions[self.random_generator.choice(len(self.actions))]
        else:
            max_q_value = np.max(self.q[state])
            max_q_values_indexes = np.where(self.q[state] == max_q_value)[0]
            best_action_index = self.random_generator.choice(
                max_q_values_indexes).item()
            return best_action_index, self.actions[best_action_index]

    def generate_actions_permutations(self):

        horizontal = [-1, 0, 1]
        vertical = [-1, 0, 1]

        actions = []

        for h in horizontal:
            for v in vertical:
                actions.append((v, h))

        return actions

    def generate_empty_q_state_space(self, environment_states: list[list[object]], max_velocity: int, actions: list[tuple]):
        total_state_space = (len(environment_states), len(
            environment_states[0]), max_velocity + 1, max_velocity + 1,  len(actions))
        return np.zeros(total_state_space)
