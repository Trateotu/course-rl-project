from constants import *

"""
This class is the environment used to train the agent. The included functions are:
- step
- reset

"""
class Simulator:

    def __init__(self, grid, swap_goal, idx):
        """
        The initialization includes setting to POS_VAL the initial position of the grid (actually fixed to (1,1))
        and to GOAL_VAL the position of the goal (fixed to (20,20)).
        The idx parameter has been used to swap initial position and goal every even idx to increase
        the difficulty of the task while training multiple MDPs continuously.


        :param grid: Initial maze that has to be solved
        :type grid: nympy array Size(22,22)
        :param idx: number of the maze of the dataset we generated
        :type idx: int
        """
        self.init_pos = (1, 1)              # FIXED by the dataset
        self.grid = grid                    # FIXED by the dataset
        self.goal = (self.grid.shape[0] - 2, self.grid.shape[1] - 2)     # bcs mazes have the borders as 1 so we need a further step


        # Swap goal and initial position (used in exp3)
        if swap_goal and idx % 2 != 0:
            self.init_pos = (self.grid.shape[0] - 2, self.grid.shape[1] - 2)
            self.goal = (1, 1)

        self.grid[self.init_pos] = POS_VAL
        self.grid[self.goal] = GOAL_VAL

        self.actual_pos_x = self.init_pos[0]
        self.actual_pos_y = self.init_pos[1]

    def get_state(self):
        return np.expand_dims(np.expand_dims(self.grid.copy(), 0), 0)

    def step(self, a):
        """
        Step function that updates the grid given an action. Fistly updates the position on the grid based on the action
        taken and checks it.
        If the action brings to a position where there's an obstacle, then the agents stays in the previous position.
        The episodes end if the agent reaches the goal (or the maximum number of steps, but the last condition is
        evaluated during the training phase, not here)

        :param a: action to take
        :type a: int, either 0: down, 1: up, 2: right, 3: left
        :return r: reward after taking the action that is -l1_norm(position - goal)
        :rtype: float
        :return end: if the episode reached the end or not
        :rtype: bool
        """
        # Remove initial position
        self.grid[self.actual_pos_x, self.actual_pos_y] = 0
        old_pos_x = self.actual_pos_x
        old_pos_y = self.actual_pos_y

        if a == 0:  # 0: forward
            self.actual_pos_x += 1
        elif a == 1:  # 1: backward
            self.actual_pos_x -= 1
        elif a == 2:  # 2: right
            self.actual_pos_y += 1
        elif a == 3:  # 3: left
            self.actual_pos_y -= 1

        # If hit obstacle: stay where you are
        if self.grid[self.actual_pos_x, self.actual_pos_y] == OBSTACLE_VAL:
            # reset initial position without moving
            self.actual_pos_x = old_pos_x
            self.actual_pos_y = old_pos_y

        r = - (np.abs(self.actual_pos_x - self.goal[0]) + np.abs(
            self.actual_pos_y - self.goal[1]))  # todo: unnormalized and l1 norm
        # r = - (np.abs(self.actual_pos_x - self.goal[0])**2 + np.abs(self.actual_pos_y - self.goal[1])**2)

        # Check if goal reached
        if self.grid[self.actual_pos_x, self.actual_pos_y] == GOAL_VAL:
            return r, True

        # Set new position
        self.grid[self.actual_pos_x, self.actual_pos_y] = POS_VAL
        return r, False

    def reset(self):
        """
        Reset the environment to the initial state
        """
        self.grid[self.actual_pos_x, self.actual_pos_y] = 0

        self.grid[self.init_pos] = POS_VAL
        self.grid[self.goal] = GOAL_VAL
        self.actual_pos_x = self.init_pos[0]
        self.actual_pos_y = self.init_pos[1]
