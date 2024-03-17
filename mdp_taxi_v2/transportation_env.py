import numpy as np
import gym
from gym import spaces


class TransportationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super(TransportationEnv, self).__init__()

        self.driver_start_point = config['driver_start_point']
        self.destinations = config['destinations']
        self.passengers = config['passengers']
        self.locations = config['locations']
        self.state = [0] * 4  # Initial state
        self.action_space = spaces.Discrete(4)  # 4 possible actions
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])  # State space
        self.driver_position = self.driver_start_point
        self.driver_path = [self.driver_start_point]  # Store driver's path

    def step(self, action):
        action -= 1  # Adjust for zero indexing
        reward = 0
        done = False

        order = self.destinations[action]
        passenger, destination = order
        passenger_position = self.passengers[passenger]
        destination_position = self.locations[destination.lower()]

        # Calculate distance and update reward
        pickup_distance = self._calculate_distance(self.driver_position, passenger_position)
        dropoff_distance = self._calculate_distance(passenger_position, destination_position)
        reward = -(pickup_distance + dropoff_distance)  # Negative reward for distance

        # Update state and path
        for i, s in enumerate(self.state):
            if s > 0:
                self.state[i] = 1
        self.state[action] = 2
        self._update_path(self.driver_position, passenger_position)
        self._update_path(passenger_position, destination_position)
        self.driver_position = destination_position

        if not 0 in self.state:
            done = True

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = [0] * 4
        self.driver_position = self.driver_start_point
        self.driver_path = [self.driver_start_point]
        return np.array(self.state)

    def render(self, mode='human'):
        grid_size = 7
        env_grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]

        # Mark destinations
        for key, value in self.locations.items():
            env_grid[value[0]][value[1]] = key

        # Mark passengers
        for key, value in self.passengers.items():
            if env_grid[value[0]][value[1]] == ' ':
                env_grid[value[0]][value[1]] = key
            else:
                env_grid[value[0]][value[1]] += key

        # Initialize the path grid to track the driver's route
        path_grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]

        for x, y in self.driver_path:
            path_grid[x][y] = '.'

        # Overlap the path_grid with the env_grid, except for the driver's current position and key locations
        for i in range(grid_size):
            for j in range(grid_size):
                if path_grid[i][j] == '.' and env_grid[i][j] == ' ':
                    env_grid[i][j] = path_grid[i][j]

        # Mark driver's position (on top of the path if necessary)
        env_grid[self.driver_position[0]][self.driver_position[1]] = 'x'

        # Build the grid string
        grid_str = "+" + "---+" * grid_size + "\n"
        for row in env_grid:
            grid_str += "| " + " | ".join(row) + " |\n"
            grid_str += "+" + "---+" * grid_size + "\n"

        if mode == 'human':
            print(grid_str)
        else:
            return grid_str


    def _calculate_distance(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def _update_path(self, start, end):
        # This method updates the driver's path from start to end
        # For simplicity, it assumes Manhattan distance and straight paths
        x_move = np.sign(end[0] - start[0])
        y_move = np.sign(end[1] - start[1])

        current = start
        while current != end:
            if current[0] != end[0]:
                current = (current[0] + x_move, current[1])
            elif current[1] != end[1]:
                current = (current[0], current[1] + y_move)
            self.driver_path.append(current)
