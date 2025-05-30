import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models


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


class OdomEstimator_wys(torch.nn.Module):

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

    def forward(self, stacked_obs, stacked_yaw, stacked_pos):
        input = torch.cat((stacked_obs, torch.cos(stacked_yaw).unsqueeze(-1), torch.sin(stacked_yaw).unsqueeze(-1), stacked_pos), dim=-1).flatten(
            start_dim=-2
        ) # 24,1024,50,48
        # # 打印模型输入里那些为0的数的索引
        # zero_indices = torch.nonzero(input == 1)
        # if zero_indices[0].numel() > 0:
        #     print("模型输入中有1的数，索引为：", zero_indices[:, 1].tolist())
        # else:
        #     print("模型输入中没有1的数")
        return self.net(input)
    
class OdomEstimator_Legolas(torch.nn.Module):

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

class OdomEstimator_baseline(torch.nn.Module):

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

    def forward(self, stacked_obs, stacked_yaw, start_mask):
        input = torch.cat(
            (stacked_obs, torch.cos(stacked_yaw).unsqueeze(-1), torch.sin(stacked_yaw).unsqueeze(-1), start_mask.unsqueeze(-1)), dim=-1
        ).flatten(start_dim=-2)
        return self.net(input)


class OdomEstimator_wys_LSTM(torch.nn.Module):
    def __init__(self, num_obs, num_stack, hidden_size=128, num_layers=2, dropout=0.5):
        super().__init__()
        input_size = num_obs + 2 + 2
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  # LSTM的dropout只在num_layers>1时生效
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, stacked_obs, stacked_yaw, stacked_pos):
        # 支持 [B, T, D] 或 [B, N, T, D], 即[batch_size, num_envs, time_steps, feature_dim] 或 [batch_size, time_steps, feature_dim]
        cos_yaw = torch.cos(stacked_yaw).unsqueeze(-1)
        sin_yaw = torch.sin(stacked_yaw).unsqueeze(-1)
        x = torch.cat((stacked_obs, cos_yaw, sin_yaw, stacked_pos), dim=-1)
        if x.dim() == 4:
            B, N, T, D = x.shape
            x = x.view(B * N, T, D)
            lstm_out, _ = self.lstm(x)
            lstm_out = self.dropout(lstm_out)
            out = self.fc(lstm_out[:, -1, :])
            out = out.view(B, N, 2)
        elif x.dim() == 3:
            B, T, D = x.shape
            lstm_out, _ = self.lstm(x)
            lstm_out = self.dropout(lstm_out)
            out = self.fc(lstm_out[:, -1, :])
            out = out.view(B, 2)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        return out

class OdomEstimator_wys_CNN(torch.nn.Module):
    def __init__(self, num_obs, num_stack):
        super().__init__()
        # 输入: [batch, num_stack, num_obs+2+2]
        input_channels = num_obs + 2 + 2  # obs + cos(yaw) + sin(yaw) + pos(x2)
        self.num_stack = num_stack
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, 128, kernel_size=3, padding=1),
            torch.nn.ELU(),
            torch.nn.Conv1d(128, 64, kernel_size=3, padding=1),
            torch.nn.ELU(),
        )
        self.dropout = torch.nn.Dropout(p=0.3)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * num_stack, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 2)
        )

    def forward(self, stacked_obs, stacked_yaw, stacked_pos):
        # stacked_obs: [B, num_stack, num_obs] or [B, N, num_stack, num_obs]
        # stacked_yaw: [B, num_stack] or [B, N, num_stack]
        # stacked_pos: [B, num_stack, 2] or [B, N, num_stack, 2]
        if stacked_obs.dim() == 4:
            B, N, T, D = stacked_obs.shape
            cos_yaw = torch.cos(stacked_yaw).unsqueeze(-1)  # [B, N, T, 1]
            sin_yaw = torch.sin(stacked_yaw).unsqueeze(-1)  # [B, N, T, 1]
            x = torch.cat((stacked_obs, cos_yaw, sin_yaw, stacked_pos), dim=-1)  # [B, N, T, D+2+2]
            x = x.view(B * N, T, -1)  # [B*N, T, C]
            x = x.transpose(1, 2)     # [B*N, C, T]
            x = self.conv(x)          # [B*N, 64, T]
            x = x.flatten(start_dim=1)  # [B*N, 64*T]
            x = self.dropout(x)
            out = self.fc(x)            # [B*N, 2]
            out = out.view(B, N, 2)
        elif stacked_obs.dim() == 3:
            B, T, D = stacked_obs.shape
            cos_yaw = torch.cos(stacked_yaw).unsqueeze(-1)  # [B, T, 1]
            sin_yaw = torch.sin(stacked_yaw).unsqueeze(-1)  # [B, T, 1]
            x = torch.cat((stacked_obs, cos_yaw, sin_yaw, stacked_pos), dim=-1)  # [B, T, D+2+2]
            x = x.transpose(1, 2)     # [B, C, T]
            x = self.conv(x)          # [B, 64, T]
            x = x.flatten(start_dim=1)  # [B, 64*T]
            x = self.dropout(x)
            out = self.fc(x)            # [B, 2]
        else:
            raise ValueError(f"Unexpected input shape: {stacked_obs.shape}")
        return out
