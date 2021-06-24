import numpy as np
import time
import tqdm


def step(x, y, direction, grid):
    grid_shape = grid.shape
    neighbor = (x + direction[0], y + direction[1])
    valid = 0 <= neighbor[0] < grid_shape[0] and 0 <= neighbor[1] < grid_shape[1]
    no_wall = valid and grid[neighbor] == 0

    return no_wall, neighbor


def BFS(queue, grid, visited, count, goal):
    if len(queue) == 0:
        return False, count

    next_queue = []
    for node in queue:
        directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        for direction in directions:
            valid, neighbor = step(node[0], node[1], direction, grid)

            if valid and not visited[neighbor]:
                if neighbor == goal:
                    return True, count + 1

                visited[neighbor] = True
                next_queue.append(neighbor)

    return BFS(next_queue, grid, visited, count + 1, goal)


class Maze_Gen:

    def get_maze(self, size):
        while True:
            temp = np.random.binomial(1, 0.2, (size, size))
            grid = np.ones((temp.shape[0] + 2, temp.shape[0] + 2))
            grid[1:-1, 1:-1] = temp

            start = (1, 1)
            goal = (grid.shape[0] - 2, grid.shape[1] - 2)
            grid[start] = grid[goal] = 0

            queue = []
            queue.append(start)

            visited = np.full(grid.shape, False, dtype=bool)
            visited[start] = True
            count = 0
            BFS_res, BFS_count = BFS(queue, grid, visited, count, goal)

            if BFS_res:
                return grid.copy(), BFS_count


