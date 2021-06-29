import numpy as np

from constants import *
from env import Simulator
from agent import Net
from maze_gen import Maze_Gen
from ppo import PPO

def main(mbs, swap_goal, grid_size, model_type, ppo=False):

    MAZES_BATCH_SIZE = mbs
    #LOG_DIR = './logs/exp6_mbs='+str(mbs)+"_swap_goal="+str(swap_goal)
    if ppo:
        INSTANCE_NAME = "swap_goal="+str(swap_goal)+"_grid_size="+str(grid_size)+"_mbs="+str(mbs)+"_model_type="+str(model_type)+"_ppo="+str(ppo)
        PPO_MODEL_SAVE_DIR = "./ppo_checkpoints/"+INSTANCE_NAME
    LOG_DIR = "./logs_grid/swap_goal="+str(swap_goal)+"_grid_size="+str(grid_size)+"_mbs="+str(mbs)+"_model_type="+str(model_type)+"_ppo="+str(ppo)

    # # load datasets
    # mazes = np.load('datasets/mazes.npy')
    # paths_length = np.load('datasets/paths_length.npy')

    mazes = None
    paths_length = []
    maze_gen = Maze_Gen()

    if ppo:
        agent = PPO(LOG_DIR, grid_size)
    else:
        agent = Net(LOG_DIR, grid_size, model_type).to(device)
    # Q_values_mazes = np.zeros((mazes.shape[0], (GRID_SIZE+2)*3, (GRID_SIZE+2)*3))

    # tot_batches = int(len(mazes)/MAZES_BATCH_SIZE)

    # train over multiple MDPs batches
    for epoch in range(EPOCHS):

        for i in range(MAZES_BATCH_SIZE):
            maze, path_len = maze_gen.get_maze(grid_size)
            maze = np.expand_dims(maze, 0)
            paths_length.append(path_len)
            if mazes is None:
                mazes = maze
            else:
                mazes = np.concatenate((mazes, maze), 0)

        batch_mazes = mazes[epoch*MAZES_BATCH_SIZE:(epoch + 1) * MAZES_BATCH_SIZE]
        sim = [Simulator(batch_mazes[q], swap_goal, q) for q in range(MAZES_BATCH_SIZE)]
        T = np.max([(paths_length[q] * HORIZON_MULTIPLIER) for q in range(MAZES_BATCH_SIZE)])
        frqs = np.zeros((MAZES_BATCH_SIZE, grid_size+2, grid_size+2))

        if ppo:
            agent.clear_batchdata()
        else:
            agent.epsilon = agent.epsilon0

        # start training of the maze
        for e in tqdm(range(EPISODES)):

            tot_reward = 0
            final_r = 0

            for i in range(MAZES_BATCH_SIZE):

                sim[i].reset()
                st = [sim[i].get_state() for i in range(MAZES_BATCH_SIZE)]


                for t in range(T):
                    frqs[i, sim[i].actual_pos_x, sim[i].actual_pos_y] += 1

                    if ppo:
                        a, logprob = agent.get_action(st[i])
                    else:
                        a = agent.get_action(st[i])

                    r, done = sim[i].step(a)
                    st1 = sim[i].get_state()

                    if ppo:
                        agent.push_batchdata(st[i], a, logprob, r, done)
                    else:
                        agent.push_memory(st[i], a, r, (not done), st1)

                    tot_reward += r
                    final_r = r

                    if done:
                        break

                    st[i] = st1

            # Update the networks
            if ppo:
                if e % EPISODES_PER_UPDATE == 0:  # TODO or size of batchdata..
                    agent.update()
                    agent.clear_batchdata()  # reset the sampled policy trajectories
                # Save actor critic checkpoints every so often
                # if e % MODEL_SAVE_FREQ == 0 and e > 0:
                #     agent.save_model(epoch+1, e+1, PPO_MODEL_SAVE_DIR)
            else:
                agent.update_Q()
                agent.update_target(e)

            agent.write_reward(tot_reward/MAZES_BATCH_SIZE, final_r/MAZES_BATCH_SIZE)

            # perform a test of the policy where there is no exploration
            if e % EPISODES_TEST == EPISODES_TEST-1:

                tot_reward = 0

                for i in range(MAZES_BATCH_SIZE):

                    sim[i].reset()
                    st = [sim[i].get_state() for i in range(MAZES_BATCH_SIZE)]

                    for t in range(T):
                        a = agent.get_action(st[i], test=True)
                        r, done = sim[i].step(a)

                        st1 = sim[i].get_state()

                        tot_reward += r

                        if done:
                            break

                        st[i] = st1

                agent.writer.add_scalar("Test reward", tot_reward/MAZES_BATCH_SIZE, int(epoch * (EPISODES/EPISODES_TEST) + e / EPISODES_TEST))

                # fig1 = plt.figure()
                # plt.imshow(frq)
                # agent.writer.add_figure('Exploration frq', fig1, int(i * 50 + e / 1000))

                # fig2 = plt.figure()
                # plt.imshow(grid_frq)
                # agent.writer.add_figure("Test path", fig2, int(i * 50 + e / 1000))
        if ppo: # save after each epoch
            agent.save_model(epoch+1, e+1, PPO_MODEL_SAVE_DIR)
        # Once trained in a new maze, test the performances in the previous mazes.
        if epoch > 0:
            tot_reward = 0
            for x, temp_maze in enumerate(mazes[:epoch*MAZES_BATCH_SIZE]):

                sim = Simulator(temp_maze, swap_goal, x)
                T = int(paths_length[x] * HORIZON_MULTIPLIER)

                # sim.reset()
                st = sim.get_state()
                tmp_reward = 0

                for t in range(T):

                    a = agent.get_action(st, test=True)
                    r, done = sim.step(a)

                    st1 = sim.get_state()
                    tot_reward += r
                    tmp_reward += r

                    if done:
                        break
                    st = st1

                print('maze: ' + str(x) + ' reward: ' + str(tmp_reward), end=' ')
            print()

            agent.writer.add_scalar("Previous mazes average reward", tot_reward / (epoch*MAZES_BATCH_SIZE), int(epoch))

        # Q_values_mazes[i] = agent.get_Q_grid(maze)
        # np.save('Q_values_retraining_NO_eps_decay.npy', Q_values_mazes)
    if ppo:
        agent.save_model(epoch+1,e+1,PPO_MODEL_SAVE_DIR)  # Always save final model for ppo
    print()


if __name__ == '__main__':
    # mbs = int(sys.argv[1])
    # grid_size = int(sys.argv[2])  # 10  # 20
    # model_type = int(sys.argv[3])
    # swap_goal = (int(sys.argv[4]) == 1)
    # ppo = (int(sys.argv[5]) == 1)

    maze_batch_sizes = [1] # maze batch size
    grid_sizes = [10, 15, 20] # square matrices
    model_type = 0
    swap_goal = False
    ppo = True
for mbs in maze_batch_sizes:
    for gs in grid_sizes:
        main(mbs, swap_goal, gs, model_type, ppo)
