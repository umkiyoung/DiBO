import torch
import numpy as np
from cma import CMAEvolutionStrategy
from botorch.utils.sampling import SobolEngine
from botorch.utils.transforms import unnormalize
from botorch.test_functions.synthetic import Ackley
import random
from matplotlib import pyplot as plt
import argparse
from baselines.utils import set_seed
from baselines.functions.test_function import TestFunction
import time

parser = argparse.ArgumentParser(description='BO experiment')
parser.add_argument('-f', '--task', default='Hopper', type=str, help='function to optimize')
parser.add_argument('--dim', default=200, type=int, help='dimension of the function')
parser.add_argument('--max_evals', default=10000, type=int, help='evaluation number')
parser.add_argument('--n_init', default=200, type=int, help=' number of initial points')
parser.add_argument('--batch_size', default=100, type=int, help='batch size of each iteration')
parser.add_argument('--seed', default=0, type=int, help='random seed')
args = parser.parse_args()
set_seed(args.seed)
import os
if not os.path.exists("./baselines/results"):
    os.makedirs("./baselines/results")
if not os.path.exists("./baselines/results/cmaes/"):
    os.makedirs("./baselines/results/cmaes/")

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ =='__main__':
    test_function = TestFunction(task=args.task, dim=args.dim, n_init=args.n_init, seed=args.seed, dtype=dtype, device=device)
    dim = args.dim
    X0 = test_function.X.cpu().numpy()
    Y0 = test_function.Y.cpu().numpy()
    total_Y = torch.tensor(Y0, dtype=dtype, device=device)
    
    Y0 = Y0 * -1
    current_best = np.min(Y0)
    best_idx = np.argmin(Y0)
    x0 = X0[best_idx]
    set_seed(args.seed)
    options = {
        'bounds': [0, 1],
        'popsize': args.batch_size,
        'verb_filenameprefix': 'outcmaes/',
    }
    
    #optimization
    sigma0 = 0.1
    es = CMAEvolutionStrategy(x0, sigma0, options)
    n_generations = (args.max_evals-args.n_init)//args.batch_size
    #Optimization Loop
    for generation in range(n_generations):
        start_time = time.time()
        X = es.ask()
        X_  = np.array(X)
        X_= torch.tensor(X_, dtype=dtype, device=device)
        Y_= torch.tensor([test_function.eval_objective(x) for x in X_], dtype=dtype, device=device).unsqueeze(-1)
        
        
        total_Y = torch.cat((total_Y, Y_), dim=0)
        Y = -1 * Y_
        Y = Y.cpu().numpy().tolist()
        es.tell(X, Y)
        end_time = time.time()
        print(f'Generation {generation+1}/{n_generations}, best value: {-np.min(Y)}, time(min): {(end_time-start_time)/60}')
        current_best = min(current_best, np.min(Y))
        
        if es.stop():
            break
        
        # Save the results
        if len(total_Y) % 1000 == 0:
            np.save(
                f"./baselines/results/cmaes/cmaes_{args.task}_{dim}_{args.seed}_{args.n_init}_{args.max_evals}_{len(total_Y)}.npy",
                np.array(np.array(total_Y.cpu())),
            )
    
        
            
    