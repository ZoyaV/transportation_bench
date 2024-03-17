from itertools import combinations

class TransportationMDP:
    """
    A class to model the Markov Decision Process for a passenger transportation environment.
    """
    def __init__(self, config):
        """
        Initializes the MDP with the given configuration.

        :param config: A dictionary containing the environment's configuration including driver's start point,
                       destinations, passengers, and locations.
        """
        self.all_states = self._generate_all_states()
        self.action_space = 4  # Total number of actions available
        self.driver_start_point = config['driver_start_point']
        self.destinations = config['destinations']
        self.passengers = config['passengers']
        self.locations = config['locations']

    def _generate_all_states(self):
        """
        Generates all possible states for the environment based on the rules defined.

        :return: A list of all possible states.
        """
        # Initial setup for state generation
        all_states = [tuple([0] * 4)]  # Include the initial state with all zeros
        # Generate states with different combinations of 0, 1, and 2
        for i in range(4):
            for num_ones in range(0, 4):
                for ones_positions in combinations(set(range(4)) - {i}, num_ones):  # Positions for '1'
                    state = [0] * 4
                    state[i] = 2  # Set '2' for the current position
                    for op in ones_positions:
                        state[op] = 1
                    all_states.append(tuple(state))
        return all_states

    def state_to_index(self, state):
        """
        Converts a state to its corresponding index in the list of all states.

        :param state: The state to convert.
        :return: The index of the state in the list of all states.
        """
        return self.all_states.index(tuple(state))

    def get_all_states(self):
        """
        Returns a list of indexes for all possible states.

        :return: A list of state indexes.
        """
        return [i for i in range(len(self.all_states))]

    def get_possible_actions(self, state):
        """
        Determines the possible actions from a given state.

        :param state: The current state index or state tuple.
        :return: A tuple of possible action indexes.
        """
        state = self.index_to_state(state)
        return tuple([i+1 for i, s in enumerate(state) if s == 0])

    def get_next_states(self, state, action):
        """
        Determines the next state given a current state and an action.

        :param state: The current state index or state tuple.
        :param action: The action taken.
        :return: A dictionary with the next state index as the key and 1 as the value (indicating deterministic transition).
        """
        state = self.index_to_state(state)
        next_state = list(state)
        for i, s in enumerate(state):
            if s > 0:
                next_state[i] = 1
        next_state[action - 1] = 2
        return {self.state_to_index(tuple(next_state)): 1}

    def is_terminal(self, state):
        """
        Checks if a state is terminal (no more actions possible).

        :param state: The state index or state tuple.
        :return: True if terminal, False otherwise.
        """
        state = self.index_to_state(state)
        return 0 not in state

    def car_position(self, state):
        """
        Determines the car's position given the current state.

        :param state: The state index or state tuple.
        :return: The car's position as a tuple (x, y).
        """
        state = self.index_to_state(state)
        if 2 in state:
            last_location = state.index(2)
            last_location_letter = self.destinations[last_location][-1].lower()
            return self.locations[last_location_letter]
        return self.driver_start_point

    def index_to_state(self, state):
        """
        Converts a state index to its corresponding state tuple.

        :param state: The state index.
        :return: The corresponding state tuple.
        """
        if isinstance(state, int):
            return self.all_states[state]
        return state

    def get_reward(self, state, action, next_state):
        """
        Calculates the reward for a transition from a state to a next state by an action.

        :param state: The current state index or state tuple.
        :param action: The action taken.
        :param next_state: The next state index or state tuple.
        :return: The calculated reward.
        """
        # Ensure state and next_state are in tuple format
        state, next_state = self.index_to_state(state), self.index_to_state(next_state)
        if not self.check_transportation_possibility(state, next_state):
            return -1  # Penalty for impossible transitions

        # Calculate distances for pickup and dropoff
        driver_position = self.car_position(state)
        passenger_letter = self.destinations[action-1][0]
        destination_letter = self.destinations[action-1][1].lower()
        passenger_position = self.passengers[passenger_letter]
        destination_position = self.locations[destination_letter]
        pickup_distance = self._calculate_distance(driver_position, passenger_position)
        dropoff_distance = self._calculate_distance(passenger_position, destination_position)

        return -(pickup_distance + dropoff_distance)  # Reward is negative of total distance

    def check_transportation_possibility(self, state, next_state):
        """
        Checks if transitioning from a given state to a next state is possible.

        :param state: The current state index or state tuple.
        :param next_state: The next state index or state tuple.
        :return: True if transition is possible, False otherwise.
        """
        state, next_state = self.index_to_state(state), self.index_to_state(next_state)
        for action in self.get_possible_actions(state):
            potential_next_state = list(self.get_next_states(state, action).keys())[0]
            if potential_next_state == self.state_to_index(next_state):
                return True
        return False

    def _calculate_distance(self, point1, point2):
        """
        Calculates the Manhattan distance between two points.

        :param point1: The first point as a tuple (x, y).
        :param point2: The second point as a tuple (x, y).
        :return: distance:The Manhattan distance between the two points.
        """
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
