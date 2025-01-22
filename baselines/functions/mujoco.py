import os
from typing import Tuple, Optional, ClassVar, Dict
from typing import NamedTuple, Optional, Union, Any, Type, Dict, Generic, TypeVar, Tuple
import gym
import numpy as np
import torch

class RunningStat(object):
    def __init__(self, shape=None):
        self._n = 0
        self._M = np.zeros(shape, dtype=np.float64)
        self._S = np.zeros(shape,  dtype=np.float64)

    def copy(self):
        other = RunningStat()
        other._n = self._n
        other._M = np.copy(self._M)
        other._S = np.copy(self._S)
        return other

    def push(self, x):
        x = np.asarray(x)
        # Unvectorized update of the running statistics.
        assert x.shape == self._M.shape, ("x.shape = {}, self.shape = {}".format(x.shape, self._M.shape))
        n1 = self._n
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            delta = x - self._M
            self._M[...] += delta / self._n
            self._S[...] += delta * delta * n1 / self._n

    def update(self, other):
        n1 = self._n
        n2 = other._n
        n = n1 + n2
        delta = self._M - other._M
        delta2 = delta * delta
        M = (n1 * self._M + n2 * other._M) / n
        S = self._S + other._S + delta2 * n1 * n2 / n
        self._n = n
        self._M = M
        self._S = S

    def __repr__(self):
        return '(n={}, mean_mean={}, mean_std={})'.format(
            self.n, np.mean(self.mean), np.mean(self.std))

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

class MujucoPolicyFunc():
    ANT_ENV: ClassVar[Tuple[str, float, float, int]] = ('Ant-v2', -1.0, 1.0, 1)
    SWIMMER_ENV: ClassVar[Tuple[str, float, float, int]] = ('Swimmer-v2', -1.0, 1.0, 5)
    HALF_CHEETAH_ENV: ClassVar[Tuple[str, float, float, int]] = ('HalfCheetah-v2', -1.0, 1.0, 5)
    HOPPER_ENV: ClassVar[Tuple[str, float, float, int]] = ('Hopper-v2', -1.4, 1.4, 5)
    WALKER_2D_ENV: ClassVar[Tuple[str, float, float, int]] = ('Walker2d-v2', -1.8, 0.9, 5)
    HUMANOID_ENV: ClassVar[Tuple[str, float, float, int]] = ('Humanoid-v2', -1.0, 1.0, 5)

    ENV_CP = {
        ANT_ENV[0]: 10.0,
        SWIMMER_ENV[0]: 30.0,
        HALF_CHEETAH_ENV[0]: 10.0,
        HOPPER_ENV[0]: 100.0,
        WALKER_2D_ENV[0]: 50.0,
        HUMANOID_ENV[0]: 20.0
    }

    def __init__(self, policy_file: str, env: str, lb: float, ub: float, num_rollouts, dtype, device, seed, negate=True):
        # lin_policy = np.load(policy_file, allow_pickle=True)
        # lin_policy = lin_policy['arr_0']
        # self._policy = lin_policy[0]
        # self._mean = lin_policy[1]
        # self._std = lin_policy[2]
        # self._dims = len(self._policy.ravel())
        self._env_name = env
        self._env = gym.make(env)
        self._env.reset(seed=2025)
        state_dims = self._env.observation_space.shape[0]
        action_dims = self._env.action_space.shape[0]
        self._dims = state_dims * action_dims
        self.dim = self._dims
        self._lb = np.full(self._dims, lb)
        self._ub = np.full(self._dims, ub)
        # self._dims = len(self._policy.ravel())
        self._policy_shape = (action_dims, state_dims)
        self._num_rollouts = num_rollouts
        self._render = False
        self.dtype = dtype
        self.device = device
        self.bounds = torch.zeros((2, self.dim), dtype=dtype, device=device)
        self.negate = negate
        self.seed = seed
        self._rs = RunningStat(state_dims)

    @property
    def lb(self) -> np.ndarray:
        return self._lb

    @property
    def ub(self) -> np.ndarray:
        return self._ub

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def is_minimizing(self) -> bool:
        return False

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        x = x.cpu().detach().numpy().reshape(1, -1)
        fx = np.zeros(len(x))
        for i, actions in enumerate(x):
            m = actions.reshape(self._policy_shape)
            rewards = []
            observations = []
            actions = []
            for t in range(self._num_rollouts):
                # self._env.seed(self.seed + t)
                obs, _ = self._env.reset()
                done = False
                total_reward = 0.
                while True:
                    self._rs.push(obs)
                    norm_obs = (obs - self._rs.mean) / (self._rs.std + 1e-6)
                    action = np.dot(m, norm_obs)
                    obs, r, done, truncated, _ = self._env.step(action)
                    total_reward += r
                    if self._render:
                        self._env.render()
                    if done or truncated:
                        break
                rewards.append(total_reward)
            fx[i] = np.mean(rewards)
            #fx[i] = -np.mean(rewards) if self.negate else np.mean(rewards)
        return torch.tensor(fx, dtype=self.dtype, device=self.device).unsqueeze(-1)

T = TypeVar('T')

class ObjectFactory(Generic[T]):
    def __init__(self, clz: Type[T], args: Optional[Tuple] = None, kwargs: Optional[Dict] = None):
        self._clz = clz
        self._args = args
        self._kwargs = kwargs

    def make_object(self) -> T:
        if self._args is not None and self._kwargs is not None:
            return self._clz(*self._args, **self._kwargs)
        elif self._args is not None:
            return self._clz(*self._args)
        elif self._kwargs is not None:
            return self._clz(**self._kwargs)
        else:
            return self._clz()

    @property
    def clz(self) -> Type:
        return self._clz

    @property
    def args(self) -> Optional[Tuple]:
        return self._args

    @property
    def kwargs(self) -> Optional[Dict]:
        return self._kwargs


func_dir = os.path.dirname(os.path.abspath(__file__))
func_factories = {
    "ant": ObjectFactory(MujucoPolicyFunc,
                         (f"{func_dir}/trained_policies/Ant-v1/lin_policy_plus.npz", *MujucoPolicyFunc.ANT_ENV)),
    "half_cheetah": ObjectFactory(MujucoPolicyFunc,
                                  (f"{func_dir}/trained_policies/HalfCheetah-v1/lin_policy_plus.npz",
                                   *MujucoPolicyFunc.HALF_CHEETAH_ENV)),
    "hopper": ObjectFactory(MujucoPolicyFunc,
                            (f"{func_dir}/trained_policies/Hopper-v1/lin_policy_plus.npz",
                             *MujucoPolicyFunc.HOPPER_ENV)),
    "humanoid": ObjectFactory(MujucoPolicyFunc,
                              (f"{func_dir}/trained_policies/Humanoid-v1/lin_policy_plus.npz",
                               *MujucoPolicyFunc.HUMANOID_ENV)),
    "swimmer": ObjectFactory(MujucoPolicyFunc,
                             (f"{func_dir}/trained_policies/Swimmer-v1/lin_policy_plus.npz",
                              *MujucoPolicyFunc.SWIMMER_ENV)),
    "walker_2d": ObjectFactory(MujucoPolicyFunc,
                               (f"{func_dir}/trained_policies/Walker2d-v1/lin_policy_plus.npz",
                                *MujucoPolicyFunc.WALKER_2D_ENV)),
}