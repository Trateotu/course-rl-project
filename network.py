from constants import *
from torch.distributions import Categorical

"""
Options to generate policy-value networks for actor-critic
"""


class ActorCritic(nn.Module):
    """
    Ex self.policy = ActorCritic(state_dim, action_dim).to(device)
    """

    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Conv2d(obs_dim, 4, 3, stride=1),
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
            nn.Linear(32, action_dim, bias=True),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Conv2d(obs_dim, 4, 3, stride=1),
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
            nn.Linear(32, 1, bias=True)
        )

    def get_action(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.item(), action_logprob

    def evaluate(self, state, action):
        state_value = self.critic(state)

        # to calculate action score(logprobs) and distribution entropy
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy
