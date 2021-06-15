from constants import *
from env import Simulator
from agent import Net


def main():
    # load datasets
    mazes = np.load('datasets/mazes.npy')
    paths_length = np.load('datasets/paths_length.npy')

    agent = Net().to(device)
    # sim = Simulator(mazes[0])
    # sim = Simulator(mazes[314])
    # T = int(paths_length[0]*3)
    # T = int(paths_length[0] * 2)
    # frq = np.zeros((GRID_SIZE, GRID_SIZE))

    Q_values_mazes = np.zeros((mazes.shape[0], 66, 66))

    # train over multiple MDPs
    for i, maze in enumerate(tqdm(mazes)):
        # define simulator, horizon (maximum number of steps per training episode) and set epsilon parameter to its initial value every new MDP
        sim = Simulator(maze, i)
        T = int(paths_length[i] * 3)
        frq = np.zeros((GRID_SIZE, GRID_SIZE))                  # This grid is used to visualize which regions of the maze the agent visits the most
        agent.epsilon = agent.epsilon0

        # start training of the maze
        for e in range(EPISODES):
            sim.reset()
            st = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)
            # st = np.expand_dims(np.reshape(sim.grid.copy(), -1), 0)
            tot_reward = 0
            final_r = 0

            # move in the maze for at most T steps following the exploration strategy (epsilon greedy) and push to the memory buffer each step
            for t in range(T):
                frq[sim.actual_pos_x][sim.actual_pos_y] += 1
                a = agent.get_action(st)
                r, done = sim.step(a)

                st1 = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)
                # st1 = np.expand_dims(np.reshape(sim.grid.copy(), -1), 0)
                tot_reward += r
                final_r = r

                agent.push_memory(st, a, r, (not done), st1)

                if done:
                    break

                st = st1

            # Update the networks
            agent.update_Q()
            agent.update_target(e, sim.grid.copy())
            agent.write_reward(tot_reward, final_r)

            # perform a test of the policy where there is no exploration
            if e % 1000 == 999:
                sim.reset()
                st = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)
                # st = np.expand_dims(np.reshape(sim.grid.copy(), -1), 0)
                tot_reward = 0
                grid_frq = -10 * sim.grid.copy()

                for t in range(T):

                    a = agent.get_action(st, test=True)
                    r, done = sim.step(a)

                    st1 = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)
                    # st1 = np.expand_dims(np.reshape(sim.grid.copy(), -1), 0)
                    tot_reward += r

                    if done:
                        break
                    grid_frq[sim.actual_pos_x, sim.actual_pos_y] += 1
                    st = st1

                agent.writer.add_scalar("Test reward", tot_reward, int(i * 50 + e / 1000))

                fig1 = plt.figure()
                plt.imshow(frq)
                agent.writer.add_figure('Exploration frq', fig1, int(i * 50 + e / 1000))

                fig2 = plt.figure()
                plt.imshow(grid_frq)
                agent.writer.add_figure("Test path", fig2, int(i * 50 + e / 1000))

        # Once trained in a new maze, test the perfrormance in the previous mazes.
        if i != 0:
            tot_reward = 0
            for x, temp_maze in enumerate(mazes[:i]):

                sim = Simulator(temp_maze, x)
                T = int(paths_length[x] * 3)

                # sim.reset()
                st = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)
                # st = np.expand_dims(np.reshape(sim.grid.copy(), -1), 0)
                tmp_reward = 0

                for t in range(T):

                    a = agent.get_action(st, test=True)
                    r, done = sim.step(a)

                    st1 = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)
                    # st1 = np.expand_dims(np.reshape(sim.grid.copy(), -1), 0)
                    tot_reward += r
                    tmp_reward += r

                    if done:
                        break
                    st = st1

                print('maze: ' + str(x) + ' reward: ' + str(tmp_reward), end=' ')
            print()

            agent.writer.add_scalar("Previous mazes average reward", tot_reward / i, int(i))

        Q_values_mazes[i] = agent.get_Q_grid(maze)
        np.save('Q_values_retraining_NO_eps_decay.npy', Q_values_mazes)

    print()


if __name__ == '__main__':
    main()
