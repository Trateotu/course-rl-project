from constants import *

"""
Class which implements the memory buffer. It stores the episodes as a circular memory, which means that once the memory
is full it starts substituting elements from the oldest one. In this way the memory keeps being up to date with the 
actions chosen from the trained network.
The episodes are stored as a dictionary:
element = {'st': st, 'a': a, 'r': r, 'terminal': terminal, 'st1': st1}

It has a function that randomly samples a batch of episodes from its memory.
"""
class Memory(object):

    def __init__(self, size):
        self.size = size
        self.memory = []
        self.position = 0

    def push(self, st, a, r, terminal, st1):
        if len(self.memory) < self.size:
            self.memory.append(None)

        element = {'st': st, 'a': a, 'r': r, 'terminal': terminal, 'st1': st1}

        self.memory[int(self.position)] = element
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


"""
Class that implements the Q value functions. It has two identical network, one that is actually trained and one
that is updated once every self.update_target_episodes with the weights of the other network in order to make
the training more stable. Other fucntions are:
- get_action
- get_tensor
- push_memory
- update_Q
- update_target
- write_reward
- get_Q_grid
"""
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        memory_size = 100000
        lr = 0.001
        self.min_memory = 1000
        self.update_target_episodes = 100
        self.batch_size = 128
        self.gamma = 0.9  # 0.97
        self.epsilon0 = 0.9
        self.epsilon = self.epsilon0
        self.epsilon_decay = 0.997
        log_dir = LOG_DIR
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_idx = 0

        self.Q_values = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(14 * 14 * 4, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 4, bias=True)
        )


        # Target network updated only once every self.update_target_episodes
        self.Q_target = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(14 * 14 * 4, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 4, bias=True)
        )

        # Tentative where we tried to use and encoder but it wasn't working and we didn't spend to much time on it

        # self.encoder = nn.Sequential(
        #     nn.Linear(GRID_SIZE*GRID_SIZE, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 2)
        # )
        #
        # self.decoder = nn.Sequential(
        #     nn.Linear(2, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, GRID_SIZE*GRID_SIZE),
        # )
        # self.Q_values = nn.Sequential(
        #     nn.Linear(2, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 4)
        # )
        #
        # self.Q_target = nn.Sequential(
        #     nn.Linear(2, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 4)
        # )

        self.Q_target.load_state_dict(self.Q_values.state_dict())
        self.Q_target.eval()

        self.D = Memory(memory_size)
        # self.optimizer = optim.RMSprop(list(self.Q_values.parameters())+list(self.decoder.parameters())+list(self.encoder.parameters()), lr=lr)
        self.optimizer = optim.RMSprop(self.Q_values.parameters(), lr=lr)

    # This function was required to train the encoder
    # def reconstruction_loss(self, x):
    #     z = self.encoder(x)
    #     x_hat = self.decoder(z)
    #
    #     loss = torch.mean(torch.square(x_hat - x))
    #     return loss

    def get_action(self, x, test=False):
        """
        This function return the action based on the state (the grid with the position and the goal) and it gets it
        either from the network or from the exploration algorithm (epsilon-greedy right now)

        :param x: actual state of the grid
        :type x: nparray
        :param test: true if we are in a test fase where we don't need the exploration
        :type test: bool
        :return a: action that the agent has to take
        :rtype: int
        """
        # q = torch.softmax(self.Q_values(self.encoder(self.get_tensor(x))), -1)
        q = torch.softmax(self.Q_values(self.get_tensor(x)), -1)
        if random.random() > self.epsilon or test:
            a = torch.argmax(q).detach().cpu().item()
        else:
            a = random.randint(0, 3)
        return a

    def get_tensor(self, state):
        return torch.from_numpy(state).float().to(device)

    def push_memory(self, s, a, r, t, s1):
        self.D.push(s, a, r, t, s1)

    def update_Q(self):
        """
        This function updates the Q_value network by sampling a batch of episodes form the buffer
        {st, a, r, terminal, st1}
        and using them to compute the TD error for the specific action "a":

        r + gamma * argmax(Q_target(st1)) - Q_value(st)    if not terminal state
        r - Q_value(st)                                    if terminal state

        """
        if len(self.D) < self.min_memory:
            return

        self.Q_values.train()

        # sample action from the buffer and store in separate elements state, action taken, reward received and following state
        data = self.D.sample(self.batch_size)

        st = torch.cat([self.get_tensor(x['st']) for x in data], 0)

        a = [x['a'] for x in data]
        r = torch.cat([torch.tensor(x['r'], dtype=torch.float32).view(1) for x in data], 0).to(device)
        terminal = torch.cat([torch.tensor(x['terminal'] * 1.0, dtype=torch.float32).view(1) for x in data], 0).to(
            device)
        st1 = torch.cat([self.get_tensor(x['st1']) for x in data], 0)

        # Compute value of st from target network by r + gamma* argmax(Q_target(st1))
        # Qt1 = self.Q_target(self.encoder(st1))
        Qt1 = self.Q_target(st1)
        max_vals, _ = torch.max(Qt1, -1)
        y = (r + terminal * (self.gamma * max_vals)).detach()

        # Compute value of st from Q_value network by Q(st) and get the Q value just for the action given from the buffer
        # Q = self.Q_values(self.encoder(st))
        Q = self.Q_values(st)
        Q = torch.cat([Q[i, a[i]].view(1) for i in range(len(a))], 0)

        # Compute the loss that corresponds to the Temporal Difference error
        TDerror = (y - Q) ** 2
        loss_q = torch.mean(TDerror)
        # loss_rep = self.reconstruction_loss(st)
        # loss = loss_rep + loss_q
        loss = loss_q

        # backprop from the mean of the TD losses in the batch
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return

    def update_target(self, episode, grid=None):
        """
        Update the target network (self.Q_target) every self.update_target_episodes and decays epsilon for
        the exploration.
        Moreover it calls the get_Q_grid function to generate a table that shows the Q-values for each possible
        state-action combination.

        :param episode: actual episode
        :type episode: int
        :param grid: actual state
        :type grid: nparray
        """
        if episode % self.update_target_episodes == 0:
            _ = self.get_Q_grid(grid)
            self.Q_target.load_state_dict(self.Q_values.state_dict())
            self.Q_target.eval()
            self.epsilon *= self.epsilon_decay

    def write_reward(self, r, r2):
        """
        Function that write on tensorboard the rewards it gets

        :param r: cumulative reward of the episode
        :type r: float
        :param r2: final reword of the episode
        :type r2: float
        """
        self.writer.add_scalar('cumulative_reward', r, self.log_idx)
        self.writer.add_scalar('final_reward', r2, self.log_idx)
        self.log_idx += 1

    def get_Q_grid(self, grid=None):
        """
        This function evaluate each state-action policy and store it in a grid. It's just for visualization purposes.

        :param grid: actual state
        :type grid: nparray
        :return: grid representing the policy for each state-action pair
        :rtype: nparray
        """
        policy_grid = np.zeros((66, 66))
        pos = np.where(grid == POS_VAL)
        grid[pos[0], pos[1]] = 0

        for i in range(22):
            for j in range(22):
                if grid[i, j] == 0:
                    grid[i, j] = POS_VAL

                    q = self.Q_target(self.get_tensor(
                        np.expand_dims(np.expand_dims(grid, 0), 0)))  # need to update the state we give to Q
                    # q = self.Q_target(self.encoder(self.get_tensor(np.expand_dims(np.reshape(grid, -1), 0))))

                    policy_grid[3 * i + 2, 3 * j + 1] = q[0, 0].detach().cpu().item()
                    policy_grid[3 * i + 0, 3 * j + 1] = q[0, 1].detach().cpu().item()
                    policy_grid[3 * i + 1, 3 * j + 2] = q[0, 2].detach().cpu().item()
                    policy_grid[3 * i + 1, 3 * j + 0] = q[0, 3].detach().cpu().item()

                    grid[i][j] = 0
                    # grid[i][j] = torch.max(q).detach().cpu().item()
        fig = plt.figure()
        plt.imshow(policy_grid)
        self.writer.add_figure('q_vals', fig, self.log_idx / self.update_target_episodes * 1.)
        # frq = np.zeros((15, 15))
        # for m in range(len(self.D.memory)):
        #     # print(m , self.D.memory[m]['st'])
        #     frq[int(self.D.memory[m]['st'][0]), int(self.D.memory[m]['st'][1])] += 1
        # fig = plt.figure()
        # plt.imshow(frq)
        # self.writer.add_figure('frq', fig, self.log_idx / self.update_target_episodes * 1.)
        return policy_grid
