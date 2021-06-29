"""
Implements PPO-Clip-esk:
https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
or something along those lines
"""
from constants import *
from network import ActorCritic
from torch.optim import Adam
import os

class BatchData:  # batchdata collected from policy
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []  # log probs of each action
        self.rewards = []
        #self.lens = []  # episodic lengths in batch, (dim=n_episodes)
        self.is_terminal = []  # whether or not terminal state was reached

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        #self.lens.clear()
        self.is_terminal.clear()

def calc_rtg(rewards, is_terminal, gamma):
    # Calculates reward-to-go
    assert len(rewards) == len(is_terminal)
    rtgs = []
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminal)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + gamma * discounted_reward
        rtgs.insert(0, discounted_reward)
    return rtgs

class PPO:
    def __init__(self, LOG_DIR, GRID_SIZE, load_pretrained=False):
        # extract environment info from maze....
        # self.mazesim = mazesim
        self.state_dim = 1  # I guess for 1 grid image?
        self.action_dim = 4  # {0: Down, 1: Up, 2: Right, 3: Left}
        self.batchdata = BatchData()
        self.writer = SummaryWriter(log_dir=LOG_DIR)
        self.log_idx = 0

        # Init params and actor-critic policy networks, old_policy used for sampling, policy for training
        self.lr_actor = 0.01
        self.lr_critic = 0.01
        self.eps_clip = 0.1
        self.gamma = 0.9
        self.c1 = 1  # VF loss coefficient
        self.c2 = 0.01  # Entropy bonus coefficient
        self.K_epochs = 5  # num epochs to train on batch data

        self.policy = ActorCritic(GRID_SIZE, self.state_dim, self.action_dim).to(device)
        if load_pretrained:  # if load actor-critic network params from file
            self.load_model()
        self.MSE_loss = nn.MSELoss()  # to calculate critic loss
        self.policy_optim = Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
        ])

        self.old_policy = ActorCritic(GRID_SIZE, self.state_dim, self.action_dim).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

    def get_action(self, state, test=False):
        # Sample actions from 'old policy'
        return self.old_policy.get_action(self.to_tensor(state))

    #     if(random.random() > self.epsilon or test):
    #         a = self.policy.act

    def update(self):
        """
            Updates the actor-critic networks for current batch data
        """
        rtgs = self.to_tensor(calc_rtg(self.batchdata.rewards,self.batchdata.is_terminal,self.gamma))  # reward-to-go
        # Normalize rewards
        rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-5)

        old_states = torch.cat([self.to_tensor(x) for x in self.batchdata.states], 0).detach()
        old_actions = self.to_tensor(self.batchdata.actions).detach()
        old_logprobs = self.to_tensor(self.batchdata.logprobs).detach()

        # Train policy for K epochs on collected trajectories, sample and update
        # Evaluate old actions and values using current policy
        for _ in range(self.K_epochs):
            logprobs, state_vals, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Importance ratio
            ratios = torch.exp(logprobs - old_logprobs.detach())  # new probs over old probs

            # Calc advantages
            A = rtgs - state_vals.detach()  # old rewards and old states evaluated by curr policy

            # Normalize advantages
            # advantages = (A-A.mean()) / (A.std() + 1e-5)

            # Actor loss using CLIP loss
            surr1 = ratios * A
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * A
            actor_loss = -torch.min(surr1, surr2)  # minus to maximize

            # Critic loss fitting to reward-to-go with entropy bonus
            critic_loss = self.c1 * self.MSE_loss(rtgs, state_vals) - self.c2 * dist_entropy

            loss = actor_loss + critic_loss

            self.policy_optim.zero_grad()
            loss.mean().backward()
            self.policy_optim.step()

        # Replace old policy with new policy
        self.old_policy.load_state_dict(self.policy.state_dict())

    def save_model(self, epoch, episode, out_dir="./"):  # TODO filename param
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        actor_filepath = os.path.join(out_dir, "ppo_actor_{}_{}.pth".format(epoch,episode))
        critic_filepath = os.path.join(out_dir,"ppo_critic_{}_{}.pth".format(epoch,episode))
        torch.save(self.policy.actor.state_dict(), actor_filepath)
        torch.save(self.policy.critic.state_dict(), critic_filepath)

    def load_model(self, actor_filepath='./ppo_actor.pth', critic_filepath='./ppo_critic.pth'):
        self.policy.actor.load_state_dict(torch.load(actor_filepath))
        self.policy.critic.load_state_dict(torch.load(critic_filepath))

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

    def push_batchdata(self, st, a, logprob, r, done):
        # adds a row of trajectory data to self.batchdata
        self.batchdata.states.append(st)
        self.batchdata.actions.append(a)
        self.batchdata.logprobs.append(logprob)
        self.batchdata.rewards.append(r)
        self.batchdata.is_terminal.append(done)

    def clear_batchdata(self):
        self.batchdata.clear()

    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).float().to(device)
        else:
            return torch.tensor(array, dtype=torch.float).to(device)

    # def set_mazesim(self, mazesim):
    #     # Assumes size of states and action space is the same as the previous one
    #     assert isinstance(mazesim, Simulator)
    #     self.mazesim = mazesim
