import numpy as np
import torch

from flipzero.env.env import Env, action2array, X_NO_ACTION_DIM, ACTION_DIM, Z_DIM, HISTORY_LENGTH
from flipzero.dmc.models import FlipZeroModel


class DeepAgent:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = FlipZeroModel()
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        # 训练 checkpoint 保存为 {'model_state_dict': ..., ...}，需要先解包
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def act(self, infoset):
        legal_actions = infoset.legal_actions
        if len(legal_actions) == 0:
            return None
        if len(legal_actions) == 1:
            return legal_actions[0]

        n = len(legal_actions)
        x_no_action = torch.from_numpy(infoset.x_no_action).to(self.device).unsqueeze(0).expand(n, -1).contiguous()
        actions = torch.from_numpy(infoset.action_encodings).to(self.device)
        z = torch.from_numpy(infoset.z).to(self.device)

        with torch.no_grad():
            # 所有合法动作共享同一条历史，LSTM 只算一次。
            lstm_last = self.model.encode_history(z).expand(n, -1)
            values = self.model.value_from_lstm(lstm_last, x_no_action, actions)
        best_idx = int(torch.argmax(values).item())
        return legal_actions[best_idx]


class RandomAgent:
    def act(self, infoset):
        legal_actions = infoset.legal_actions
        if len(legal_actions) == 0:
            return None
        return legal_actions[np.random.randint(len(legal_actions))]
