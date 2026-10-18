import torch
import torch.nn as nn

from flipzero.env.env import X_NO_ACTION_DIM, ACTION_DIM, Z_DIM, HISTORY_LENGTH


LSTM_HIDDEN = 128
MLP_HIDDEN = 512
MLP_LAYERS = 5
Z_FLAT_DIM = HISTORY_LENGTH * Z_DIM


class FlipZeroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(Z_DIM, LSTM_HIDDEN, batch_first=True)
        input_dim = LSTM_HIDDEN + X_NO_ACTION_DIM + ACTION_DIM
        layers = []
        prev_dim = input_dim
        for _ in range(MLP_LAYERS):
            layers.append(nn.Linear(prev_dim, MLP_HIDDEN))
            layers.append(nn.ReLU())
            prev_dim = MLP_HIDDEN
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_no_action, action, z, return_value=True):
        if z.dim() == 4:
            B, T, H, F = z.shape
            z = z.reshape(B * T, H, F)
        elif z.dim() == 3 and z.shape[1] == HISTORY_LENGTH and z.shape[2] == Z_DIM:
            pass
        else:
            z = z.reshape(z.shape[0], HISTORY_LENGTH, Z_DIM)

        lstm_out, _ = self.lstm(z)
        lstm_last = lstm_out[:, -1, :]

        if lstm_last.shape[0] != x_no_action.shape[0] and x_no_action.dim() == 3:
            lstm_last = lstm_last.reshape(x_no_action.shape[0], -1)

        x = torch.cat([lstm_last, x_no_action.reshape(lstm_last.shape[0], -1),
                        action.reshape(lstm_last.shape[0], -1)], dim=1)
        value = self.mlp(x).squeeze(-1)
        if return_value:
            return dict(value=value)
        return value


class Model:
    def __init__(self, device=None):
        self.device = device
        self.model = FlipZeroModel()
        if device is not None:
            self.model.to(device)

    def share_memory(self):
        self.model.share_memory()

    def eval(self):
        self.model.eval()

    def get_model(self):
        return self.model

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def predict(self, x_no_action, action, z):
        with torch.no_grad():
            x_no_action_t = torch.tensor(x_no_action, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
            z_t = torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)
            result = self.model(x_no_action_t, action_t, z_t, return_value=True)
            return result['value'].item()
