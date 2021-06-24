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

    def reset(self):
        self.position = 0
        self.memory = []

    def __len__(self):
        return len(self.memory)


class MamlParams(nn.Module):
    def __init__(self, grid_size):
        super(MamlParams, self).__init__()

        self.img_reduced_dim = grid_size + 2 - 2 * 4

        self.theta_shapes = [[4, 1, 3, 3], [4],
                             [4, 4, 3, 3], [4],
                             [4, 4, 3, 3], [4],
                             [4, 4, 3, 3], [4],
                             [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                             [4, 32], [4]]

        # self.batch_norm1 = nn.BatchNorm2d(self.filters, track_running_stats=False)
        # self.batch_norm2 = nn.BatchNorm2d(self.filters, track_running_stats=False)
        # self.batch_norm3 = nn.BatchNorm2d(self.filters, track_running_stats=False)
        # self.batch_norm4 = nn.BatchNorm2d(self.filters, track_running_stats=False)
        #
        # self.max_pool = nn.MaxPool2d(2)
        #
        # self.lr = nn.ParameterList([nn.Parameter(torch.tensor(lr))] * len(self.theta_shapes))

        self.theta_0 = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.theta_shapes])
        for i in range(len(self.theta_0)):
            if self.theta_0[i].dim() > 1:
                torch.nn.init.kaiming_uniform_(self.theta_0[i])

    def get_theta(self):
        return self.theta_0

    def get_size(self):
        return np.sum([np.prod(x) for x in self.theta_shapes])

    def forward(self, x, theta=None):

        if theta is None:
            theta = self.theta_0

        h = F.relu(F.conv2d(x, theta[0], bias=theta[1], stride=1, padding=0))
        h = F.relu(F.conv2d(h, theta[2], bias=theta[3], stride=1, padding=0))
        h = F.relu(F.conv2d(h, theta[4], bias=theta[5], stride=1, padding=0))
        h = F.relu(F.conv2d(h, theta[6], bias=theta[7], stride=1, padding=0))
        h = h.contiguous()
        h = h.view(-1, (self.img_reduced_dim * self.img_reduced_dim * 4))
        h = F.relu(F.linear(h, theta[8], bias=theta[9]))
        y = F.linear(h, theta[10], bias=theta[11])

        return y


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

    def __init__(self, LOG_DIR, GRID_SIZE):
        super(Net, self).__init__()

        # memory_size = 100000
        # lr = 0.001
        # self.min_memory = 1000
        # self.update_target_episodes = 100
        # self.batch_size = 128
        # self.gamma = 0.9  # 0.97
        # self.epsilon0 = 0.9
        # self.epsilon = self.epsilon0
        # self.epsilon_decay = 0.997
        # log_dir = LOG_DIR
        # self.writer = SummaryWriter(log_dir=log_dir)
        # self.log_idx = 0

        memory_size = 10000
        lr = 0.001
        self.batch_size = 32
        self.gamma = 0.9  # 0.97
        self.epsilon0 = 0.9
        self.epsilon = self.epsilon0
        self.epsilon_decay = 0.7
        log_dir = LOG_DIR
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_idx = 0

        self.inner_lr = 0.1
        self.training_steps = 10

        self.model_theta = MamlParams(GRID_SIZE)

        # self.Q_model = network_architecture(GRID_SIZE, model)
        #
        # # Target network updated only once every self.update_target_episodes
        # self.Q_target = network_architecture(GRID_SIZE, model)

        # self.Q_target.model.load_state_dict(self.Q_model.model.state_dict())
        # self.Q_target.eval()

        self.D = Memory(memory_size)
        # self.optimizer = optim.RMSprop(list(self.Q_values.parameters())+list(self.decoder.parameters())+list(self.encoder.parameters()), lr=lr)
        self.optimizer = optim.RMSprop(self.model_theta.parameters(), lr=lr)

    # This function was required to train the encoder
    # def reconstruction_loss(self, x):
    #     z = self.encoder(x)
    #     x_hat = self.decoder(z)
    #
    #     loss = torch.mean(torch.square(x_hat - x))
    #     return loss

    def get_action(self, x, theta=None, test=False):

        q = torch.softmax(self.model_theta(self.get_tensor(x), theta=theta), -1)
        if random.random() > self.epsilon or test:
            a = torch.argmax(q).detach().cpu().item()
        else:
            a = random.randint(0, 3)
        return a

    def get_tensor(self, state):
        return torch.from_numpy(state).float().to(device)

    def push_memory(self, s, a, r, t, s1):
        self.D.push(s, a, r, t, s1)

    def adapt(self, s, a, r, done, s1, train=False):

        self.model_theta.train()

        theta_i = self.model_theta.get_theta()

        s = self.get_tensor(s)
        r = torch.tensor(r, dtype=torch.float32).view(1).to(device)
        terminal = torch.tensor(done * 1.0, dtype=torch.float32).view(1).to(device)
        s1 = self.get_tensor(s1)

        Q1 = self.model_theta(s1)
        max_vals, _ = torch.max(Q1, -1)
        y = (r + terminal * (self.gamma * max_vals)).detach()

        Q = self.model_theta(s)
        Q = Q[0, a].view(1)

        TDerror = (y - Q) ** 2
        loss_q = torch.mean(TDerror)

        if train:
            theta_grad_s = torch.autograd.grad(outputs=loss_q, inputs=theta_i, create_graph=True)
            theta_i = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(theta_grad_s, theta_i)))
        else:
            theta_grad_s = torch.autograd.grad(outputs=loss_q, inputs=theta_i)
            theta_i = list(map(lambda p: p[1] - self.inner_lr * p[0].detach(), zip(theta_grad_s, theta_i)))

        return theta_i


    def update_Q(self, theta):

        self.model_theta.train()

        tot_loss = 0

        for _ in range(self.training_steps):

            # sample action from the buffer and store in separate elements state, action taken, reward received and following state
            data = self.D.sample(self.batch_size)

            st = torch.cat([self.get_tensor(x['st']) for x in data], 0)

            a = [x['a'] for x in data]
            r = torch.cat([torch.tensor(x['r'], dtype=torch.float32).view(1) for x in data], 0).to(device)
            terminal = torch.cat([torch.tensor(x['terminal'] * 1.0, dtype=torch.float32).view(1) for x in data], 0).to(device)
            st1 = torch.cat([self.get_tensor(x['st1']) for x in data], 0)

            # Compute value of st from target network by r + gamma* argmax(Q_target(st1))
            # Qt1 = self.Q_target(self.encoder(st1))
            Qt1 = self.model_theta(st1, theta)
            max_vals, _ = torch.max(Qt1, -1)
            y = (r + terminal * (self.gamma * max_vals)).detach()

            # Compute value of st from Q_value network by Q(st) and get the Q value just for the action given from the buffer
            # Q = self.Q_values(self.encoder(st))
            Q = self.model_theta(st, theta)
            Q = torch.cat([Q[i, a[i]].view(1) for i in range(len(a))], 0)

            # Compute the loss that corresponds to the Temporal Difference error
            TDerror = (y - Q) ** 2
            loss_q = torch.mean(TDerror)
            # loss_rep = self.reconstruction_loss(st)
            # loss = loss_rep + loss_q
            tot_loss += loss_q

        # backprop from the mean of the TD losses in the batch
        self.optimizer.zero_grad()
        tot_loss.backward()
        self.optimizer.step()

        return

    # def update_target(self, episode, grid=None):
    #     """
    #     Update the target network (self.Q_target) every self.update_target_episodes and decays epsilon for
    #     the exploration.
    #     Moreover it calls the get_Q_grid function to generate a table that shows the Q-values for each possible
    #     state-action combination.
    #
    #     :param episode: actual episode
    #     :type episode: int
    #     :param grid: actual state
    #     :type grid: nparray
    #     """
    #     if episode % self.update_target_episodes == 0:
    #         if grid != None:
    #             _ = self.get_Q_grid(grid)
    #         self.Q_target.model.load_state_dict(self.Q_model.model.state_dict())
    #         self.Q_target.eval()
    #         self.epsilon *= self.epsilon_decay

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

    # def get_Q_grid(self, grid=None):
    #     """
    #     This function evaluate each state-action policy and store it in a grid. It's just for visualization purposes.
    #
    #     :param grid: actual state
    #     :type grid: nparray
    #     :return: grid representing the policy for each state-action pair
    #     :rtype: nparray
    #     """
    #     policy_grid = np.zeros((66, 66))
    #     pos = np.where(grid == POS_VAL)
    #     grid[pos[0], pos[1]] = 0
    #
    #     for i in range(22):
    #         for j in range(22):
    #             if grid[i, j] == 0:
    #                 grid[i, j] = POS_VAL
    #
    #                 q = self.Q_target(self.get_tensor(
    #                     np.expand_dims(np.expand_dims(grid, 0), 0)))  # need to update the state we give to Q
    #                 # q = self.Q_target(self.encoder(self.get_tensor(np.expand_dims(np.reshape(grid, -1), 0))))
    #
    #                 policy_grid[3 * i + 2, 3 * j + 1] = q[0, 0].detach().cpu().item()
    #                 policy_grid[3 * i + 0, 3 * j + 1] = q[0, 1].detach().cpu().item()
    #                 policy_grid[3 * i + 1, 3 * j + 2] = q[0, 2].detach().cpu().item()
    #                 policy_grid[3 * i + 1, 3 * j + 0] = q[0, 3].detach().cpu().item()
    #
    #                 grid[i][j] = 0
    #                 # grid[i][j] = torch.max(q).detach().cpu().item()
    #     fig = plt.figure()
    #     plt.imshow(policy_grid)
    #     self.writer.add_figure('q_vals', fig, self.log_idx / self.update_target_episodes * 1.)
    #     # frq = np.zeros((15, 15))
    #     # for m in range(len(self.D.memory)):
    #     #     # print(m , self.D.memory[m]['st'])
    #     #     frq[int(self.D.memory[m]['st'][0]), int(self.D.memory[m]['st'][1])] += 1
    #     # fig = plt.figure()
    #     # plt.imshow(frq)
    #     # self.writer.add_figure('frq', fig, self.log_idx / self.update_target_episodes * 1.)
    #     return policy_grid
