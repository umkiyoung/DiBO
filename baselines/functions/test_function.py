from typing import Tuple

from botorch.test_functions import Ackley, Branin, Rastrigin, Levy, Rosenbrock
from botorch.utils.transforms import unnormalize

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.quasirandom import SobolEngine
from baselines.functions.rover_planning import Rover
from baselines.functions.mujoco import MujucoPolicyFunc
from baselines.functions.lasso_benchmark import LassoDNABenchmark


class TestFunction(Dataset):
    def __init__(self, task: str, dim: int = 200, n_init: int = 200, seed: int = 0, dtype=torch.float64, device='cpu', negate=True,):
        self.task = task
        self.dim = dim
        self.n_init = n_init
        self.seed = seed
        self.dtype = dtype
        self.device = device
        self.lb, self.ub = None, None
        
        #NOTE: Synthetic Functions
        if task == 'Ackley':
            self.fun = Ackley(dim=dim, negate = negate).to(dtype=dtype, device=device)
            self.lb, self.ub = -5, 10 #Following TurBO
            
            # For LA-MCTS
            self.fun.dims_ = dim
            self.fun.lb_ = -5 * np.ones(dim)
            self.fun.ub_ = 10 * np.ones(dim)
            
            self.fun.Cp = 1
            self.fun.leaf_size = 10
            self.fun.kernel_type = "rbf"
            self.fun.ninits = n_init
            self.fun.gamma_type = "auto"
        elif task == 'Branin':
            self.fun = Branin(negate = negate).to(dtype=dtype, device=device)
        elif task == 'Rastrigin':
            self.fun = Rastrigin(dim=dim, negate = negate).to(dtype=dtype, device=device)
            self.lb, self.ub = -5, 5 #Following MCMC_BO
            
            # For LA-MCTS
            self.fun.dims_ = dim
            self.fun.lb_ = -5 * np.ones(dim)
            self.fun.ub_ = 5 * np.ones(dim)
            
            self.fun.Cp = 1
            self.fun.leaf_size = 10
            self.fun.kernel_type = "rbf"
            self.fun.ninits = n_init
            self.fun.gamma_type = "auto"
        elif task == 'Levy':
            self.fun = Levy(dim=dim, negate = negate).to(dtype=dtype, device=device)
            self.lb, self.ub = -10, 10 #Following LA-MCTS
            
            # For LA-MCTS
            self.fun.dims_ = dim
            self.fun.lb_ = -10 * np.ones(dim)
            self.fun.ub_ = 10 * np.ones(dim)
            
            self.fun.Cp = 1
            self.fun.leaf_size = 10
            self.fun.kernel_type = "rbf"
            self.fun.ninits = n_init
            self.fun.gamma_type = "auto"
        elif task == 'Rosenbrock':
            self.fun = Rosenbrock(dim=dim, negate = negate).to(dtype=dtype, device=device)
            self.lb, self.ub = -5, 10 #Following LA-MCTS
            
            # For LA-MCTS
            self.fun.dims_ = dim
            self.fun.lb_ = -5 * np.ones(dim)
            self.fun.ub_ = 10 * np.ones(dim)
            
            self.fun.Cp = 1
            self.fun.leaf_size = 10
            self.fun.kernel_type = "rbf"
            self.fun.ninits = n_init
            self.fun.gamma_type = "auto"
        # elif task in [
        #     'Swimmer', 'Hopper', 'Walker2d', 'HalfCheetah', 'Ant', 'Humanoid', 
        #     'HumanoidStandup', 'InvertedDoublePendulum', 'InvertedPendulum', 'Reacher'
        #     ]:
        #     self.fun = mujoco_gym_env.MujocoGymEnv(task + '-v2', 10, minimize=False)
        #     self.dim = self.fun.dim
        #     self.lb, self.ub = -1, 1
        elif task in ['Ant', 'Swimmer', 'HalfCheetah', 'Hopper', 'Walker2d', 'Humanoid']:
            env_settings = {
                'Ant': ('Ant-v4', -1.0, 1.0, 3),
                'Swimmer': ('Swimmer-v4', -1.0, 1.0, 3),
                'HalfCheetah': ('HalfCheetah-v4', -1.0, 1.0, 3),
                'Hopper': ('Hopper-v4', -1.0, 1.0, 3),
                'Walker2d': ('Walker2d-v4', -1.0, 1.0, 3),
                'Humanoid': ('Humanoid-v4', -1.0, 1.0, 3)
            }
            env_name, self.lb, self.ub, num_rollouts = env_settings[task]
            self.fun = MujucoPolicyFunc(
                policy_file = f'baselines/functions/trained_policies/{task}-v1/lin_policy_plus.npz',
                env = env_name,
                lb = self.lb,
                ub = self.ub,
                num_rollouts = num_rollouts,
                dtype = dtype,
                device = device,
                seed=seed,
                negate=negate             
            )
            # self.fun = MujocoGymEnv(env_name=env_settings[task][0],
            #                         num_rollouts=env_settings[task][3],
            #                         minimize=True,
            #                         dtype=dtype,
            #                         device=device)
            
            # For LA-MCTS
            self.fun.dims_ = dim
            self.fun.lb_ = env_settings[task][1] * np.ones(dim)
            self.fun.ub_ = env_settings[task][2] * np.ones(dim)
            
            self.fun.Cp = 10
            self.fun.leaf_size = 100
            self.fun.kernel_type = "poly"
            self.fun.ninits = n_init
            self.fun.gamma_type = "auto"
        elif task == 'RoverPlanning':
            self.fun = Rover(dim=dim, dtype=dtype, device=device)
            self.lb, self.ub = 0, 1
            
            # For LA-MCTS
            self.fun.dims_ = dim
            # self.fun.lb = -5 * np.ones(dim)
            # self.fun.ub = 10 * np.ones(dim)
            
            self.fun.Cp = 50
            self.fun.leaf_size = 10
            self.fun.kernel_type = "poly"
            self.fun.ninits = n_init
            self.fun.gamma_type = "scale"
        elif task == 'LunarLanding':
            self.fun = Lunarlanding(dtype=dtype, device=device)
            self.lb, self.ub = 0, 2
            
            # For LA-MCTS
            self.fun.dim_ = dim
            # self.fun.lb = -5 * np.ones(dim)
            # self.fun.ub = 10 * np.ones(dim)
            
            self.fun.Cp = 50
            self.fun.leaf_size = 10
            self.fun.kernel_type = "poly"
            self.fun.ninits = n_init
            self.fun.gamma_type = "scale"
        elif task == 'DNA':
            self.fun = LassoDNABenchmark(seed=seed, dtype=dtype, device=device)  
            self.lb, self.ub = -1, 1
            
            # For LA-MCTS
            self.fun.dim_ = dim
            self.fun.Cp = 50
            self.fun.leaf_size = 10
            self.fun.kernel_type = "poly"
            self.fun.ninits = n_init
            self.fun.gamma_type = "scale"
            
        else:
            raise ValueError(f"Unknown task: {task}")
        
        if self.lb is not None and self.ub is not None:
            self.fun.bounds[0, :].fill_(self.lb)
            self.fun.bounds[1, :].fill_(self.ub)
            self.fun.bounds.to(dtype=dtype, device=device)
        
        self.get_initial_points()
        
    def eval_objective(self, x):
        return self.fun(unnormalize(x, self.fun.bounds))
        
    def get_initial_points(self):
        sobol = SobolEngine(self.dim, scramble=True, seed=self.seed)
        self.X = sobol.draw(n=self.n_init).to(self.dtype).to(self.device)
        self.Y = torch.tensor([self.eval_objective(x) for x in self.X], dtype=self.dtype, device=self.device).unsqueeze(-1)
        return self.X, self.Y
    
    def reset(self):
        sobol = SobolEngine(self.dim, scramble=True)
        self.X = sobol.draw(n=self.n_init).to(self.dtype).to(self.device)
        self.Y = torch.tensor([self.eval_objective(x) for x in self.X], dtype=self.dtype, device=self.device).unsqueeze(-1)
        return self.X, self.Y
    
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class WeightTestFunction(TestFunction):
    def __getitem__(self, idx):
        return self.X_norm[idx], self.Y_norm[idx], self.weights[idx]