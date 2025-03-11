import torch
import torch.nn.functional as F


class ActorCritic(torch.nn.Module):

    def __init__(self, num_act, num_obs, num_privileged_obs):
        super().__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(num_obs + num_privileged_obs, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_obs, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_act),
        )
        self.logstd = torch.nn.parameter.Parameter(torch.full((1, num_act), fill_value=-2.0), requires_grad=True)

    def act(self, obs):
        action_mean = self.actor(obs)
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_std)

    def est_value(self, obs, privileged_obs):
        critic_input = torch.cat((obs, privileged_obs), dim=-1)
        return self.critic(critic_input).squeeze(-1)

    def forward(self, obs):
        return self.actor(obs)


class RMA(ActorCritic):

    def __init__(self, num_act, num_obs, num_stack, num_privileged_obs, num_embedding):
        super().__init__(num_act, num_obs, num_privileged_obs)
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_obs + num_embedding, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_act),
        )
        self.privileged_encoder = torch.nn.Sequential(
            torch.nn.Linear(num_privileged_obs, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_embedding),
        )
        self.adaptation_module = torch.nn.Sequential(
            torch.nn.Linear(num_obs * num_stack, 1024),
            torch.nn.ELU(),
            torch.nn.Linear(1024, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_embedding),
        )

    def act(self, obs, privileged_obs=None, stacked_obs=None):
        if privileged_obs is not None:
            embedding = self.privileged_encoder(privileged_obs)
        if stacked_obs is not None:
            embedding = self.adaptation_module(stacked_obs.flatten(start_dim=-2))
        act_input = torch.cat((obs, embedding), dim=-1)
        action_mean = self.actor(act_input)
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        return dist, embedding

    def ac_parameters(self):
        for p in self.critic.parameters():
            yield p
        for p in self.actor.parameters():
            yield p
        for p in self.privileged_encoder.parameters():
            yield p
        yield self.logstd

    def adapt_parameters(self):
        for p in self.adaptation_module.parameters():
            yield p

    def forward(self, obs, stacked_obs):
        embedding = self.adaptation_module(stacked_obs.flatten(start_dim=-2))
        act_input = torch.cat((obs, embedding), dim=-1)
        return self.actor(act_input)


class DenoisingRMA(RMA):

    def __init__(self, num_act, num_obs, num_stack, num_privileged_obs, num_embedding):
        super().__init__(num_act, num_obs, num_stack, num_privileged_obs, num_embedding)
        self.privileged_decoder = torch.nn.Sequential(
            torch.nn.Linear(num_embedding, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_privileged_obs),
        )

    def act(self, obs, stacked_obs, decoder=False):
        embedding = self.adaptation_module(stacked_obs.flatten(start_dim=-2))
        act_input = torch.cat((obs, embedding), dim=-1)
        action_mean = self.actor(act_input)
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        if decoder:
            privileged_obs_est = self.privileged_decoder(embedding)
            return dist, embedding, privileged_obs_est
        else:
            return dist, embedding


class OdomEstimator(torch.nn.Module):

    def __init__(self, num_obs, num_stack):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_obs * num_stack, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 2),
        )

    def forward(self, stacked_obs, stacked_yaw):
        input = torch.cat((stacked_obs, torch.cos(stacked_yaw).unsqueeze(-1), torch.sin(stacked_yaw).unsqueeze(-1)), dim=-1).flatten(
            start_dim=-2
        )
        return self.net(input)
