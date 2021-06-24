import numpy as np

from constants import *
from env import Simulator
from meta_agent import Net
from maze_gen import Maze_Gen


def main(swap_goal, grid_size):

    #LOG_DIR = './logs/exp6_mbs='+str(mbs)+"_swap_goal="+str(swap_goal)
    LOG_DIR = "./logs_grid/MAML_swap_goal="+str(swap_goal)+"_grid_size="+str(grid_size)+"_mbs=1_model_type=0"

    # # load datasets
    # mazes = np.load('datasets/mazes.npy')
    # paths_length = np.load('datasets/paths_length.npy')

    mazes = None
    paths_length = []
    maze_gen = Maze_Gen()

    agent = Net(LOG_DIR, grid_size).to(device)

    # train over multiple MDPs batches
    for epoch in tqdm(range(EPOCHS)):

        maze, path_len = maze_gen.get_maze(grid_size)
        paths_length.append(path_len)
        if mazes is None:
            mazes = np.expand_dims(maze, 0)
        else:
            mazes = np.concatenate((mazes, np.expand_dims(maze, 0)), 0)

        sim = Simulator(maze, swap_goal, epoch)
        T = paths_length[-1] * HORIZON_MULTIPLIER
        frqs = np.zeros((1, grid_size+2, grid_size+2))

        agent.epsilon = agent.epsilon0

        agent.D.reset()

        ''' ADAPTATION STEP '''

        sim.reset()
        s0 = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)
        a0 = agent.get_action(s0)
        r0, done = sim.step(a0)
        s1 = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)
        theta_i = agent.adapt(s0, a0, r0, (not done), s1, train=True)

        ''' EVALUATION STEP '''

        # start training of the maze
        for e in range(EPISODES):

            tot_reward = 0
            final_r = 0

            sim.reset()
            st = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)


            for t in range(T):
                frqs[0, sim.actual_pos_x, sim.actual_pos_y] += 1
                a = agent.get_action(st, theta=theta_i)
                r, done = sim.step(a)

                st1 = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)

                tot_reward += r
                final_r = r

                agent.push_memory(st, a, r, (not done), st1)

                if done:
                    break

                st = st1

            # TODO: decrease epsilon for multiple trajectories evaluations ???
            agent.epsilon *= agent.epsilon_decay

        # Update the networks
        agent.update_Q(theta_i)
        agent.write_reward(tot_reward, final_r)

        ''' TEST NEW POLICY ON CURRENT MAZE '''

        sim.reset()
        s0 = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)
        a0 = agent.get_action(s0)
        r0, done = sim.step(a0)
        s1 = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)
        theta_i = agent.adapt(s0, a0, r0, (not done), s1)

        tot_reward = 0

        sim.reset()
        st = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)

        for t in range(T):
            a = agent.get_action(st, theta=theta_i, test=True)
            r, done = sim.step(a)

            st1 = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)

            tot_reward += r

            if done:
                break

            st = st1

        agent.writer.add_scalar("Test reward", tot_reward, int(epoch))

        # fig1 = plt.figure()
        # plt.imshow(frq)
        # agent.writer.add_figure('Exploration frq', fig1, int(i * 50 + e / 1000))

        # fig2 = plt.figure()
        # plt.imshow(grid_frq)
        # agent.writer.add_figure("Test path", fig2, int(i * 50 + e / 1000))

        ''' TEST ON OLD MAZES '''

        # Once trained in a new maze, test the performances in the previous mazes.
        if epoch > 0:
            tot_reward = 0
            first_maze = max(0, epoch-20)
            diff_mazes = epoch - first_maze
            for x, temp_maze in enumerate(mazes[first_maze:epoch]):

                sim = Simulator(temp_maze, swap_goal, x)
                T = int(paths_length[x] * HORIZON_MULTIPLIER)
                sim.reset()
                s0 = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)
                a0 = agent.get_action(s0)
                r0, done = sim.step(a0)
                s1 = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)
                theta_i = agent.adapt(s0, a0, r0, (not done), s1)

                sim.reset()
                st = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)
                tmp_reward = 0

                for t in range(T):

                    a = agent.get_action(st, theta=theta_i, test=True)
                    r, done = sim.step(a)

                    st1 = np.expand_dims(np.expand_dims(sim.grid.copy(), 0), 0)
                    tot_reward += r
                    tmp_reward += r

                    if done:
                        break
                    st = st1

            #     print('maze: ' + str(x) + ' reward: ' + str(tmp_reward), end=' ')
            # print()

            agent.writer.add_scalar("Previous mazes average reward", tot_reward / diff_mazes, int(epoch))

        # Q_values_mazes[i] = agent.get_Q_grid(maze)
        # np.save('Q_values_retraining_NO_eps_decay.npy', Q_values_mazes)

    print()


if __name__ == '__main__':
    grid_size = 15 #int(sys.argv[1])  # 10  # 20
    swap_goal = 0 #(int(sys.argv[2]) == 1)
    main(swap_goal, grid_size)
