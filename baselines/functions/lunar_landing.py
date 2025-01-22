from typing import Tuple, Optional, ClassVar, Dict

import gym
import numpy as np
import torch

class BaseFunc():
    def __init__(self, dims: int, lb: np.ndarray, ub: np.ndarray):
        self._dims = dims
        self._lb = lb
        self._ub = ub

    @property
    def lb(self) -> np.ndarray:
        return self._lb

    @property
    def ub(self) -> np.ndarray:
        return self._ub

    @property
    def dims(self) -> int:
        return self._dims

class Lunarlanding(BaseFunc):
    def __init__(self, dtype, device):
        dims = 12
        super().__init__(dims, np.zeros(dims), 2 * np.ones(dims))
        self._env = gym.make('LunarLander-v2')
        self.bounds = torch.zeros((2, dims), dtype=dtype, device=device)
        self.dtype = dtype
        self.device = device

    @property
    def is_minimizing(self) -> bool:
        return False

    @staticmethod
    def heuristic_Controller(s, w):
        angle_targ = s[0] * w[0] + s[2] * w[1]
        if angle_targ > w[2]:
            angle_targ = w[2]
        if angle_targ < -w[2]:
            angle_targ = -w[2]
        hover_targ = w[3] * np.abs(s[0])

        angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
        hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

        if s[6] or s[7]:
            angle_todo = w[8]
            hover_todo = -(s[3]) * w[9]

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
            a = 2
        elif angle_todo < -w[11]:
            a = 3
        elif angle_todo > +w[11]:
            a = 1
        return a

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        x = x.cpu().detach().numpy().reshape(1,-1)
        rs = np.zeros(len(x))
        for i, actions in enumerate(x):
            total_rewards = []
            for t in range(0, 5):
                state = self._env.reset()
                total_reward = 0.0
                num_steps = 2000

                for step in range(num_steps):
                    # env.render()
                    received_action = self.heuristic_Controller(state, actions)
                    next_state, reward, done, info = self._env.step(received_action)
                    total_reward += reward
                    state = next_state
                    if done:
                        break
                total_rewards.append(total_reward)
            rs[i] = np.mean(total_rewards)
        return torch.tensor(rs, dtype=self.dtype, device=self.device).unsqueeze(-1)
