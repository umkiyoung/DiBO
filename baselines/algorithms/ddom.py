import argparse
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import wandb
from baselines.functions.test_function import WeightTestFunction
from baselines.models.conditional_diffusion import ConditionalDiffusionModel
from baselines.utils import (adaptive_temp_v2, get_value_based_weights,
                             set_seed, softmax)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Hopper")
    parser.add_argument("--dim", type=int, default=33)
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--n_init", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_evals", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--track", type=bool, default=False)
    parser.add_argument("--prior_hidden_dim", type=int, default=512) # Matched to our model
    parser.add_argument("--dropout_prob", type=float, default=0.15) # Following paper
    parser.add_argument("--gamma", type=float, default=2.0) # (1 + gamma) *cond - gamma * uncond, fixed to 2.0
    parser.add_argument("--temp", type=str, default="90") # Following paper. Tau = 0.1
    args = parser.parse_args()
    
    import os
    if not os.path.exists("./baselines/results"):
        os.makedirs("./baselines/results")
    if not os.path.exists("./baselines/results/ddom/"):
        os.makedirs("./baselines/results/ddom/")

    task = args.task

    dim = args.dim
    train_batch_size = args.train_batch_size
    batch_size = args.batch_size
    n_init = args.n_init
    seed = args.seed
    dtype = torch.double
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    
    test_function = WeightTestFunction(task = task, dim = dim, n_init = n_init, seed = seed, dtype=dtype, device=device)
    # dim = test_function.fun.dim
    # args.dim = dim
    total_Y = test_function.Y

    num_rounds = (args.max_evals - n_init) // batch_size
    num_epochs = args.num_epochs
    for round in range(num_rounds):
        start_time = time.time()
        test_function.X_mean = test_function.X.mean(dim=0)
        test_function.X_std = test_function.X.std(dim=0)
        test_function.X_norm = (test_function.X - test_function.X_mean) / (test_function.X_std + 1e-7)
        
        test_function.Y_mean = test_function.Y.mean()
        test_function.Y_std = test_function.Y.std()
        test_function.Y_norm = (test_function.Y - test_function.Y_mean) / (test_function.Y_std + 1e-7)
        
        #TODO: Implement Bin-based weighting
        weights = get_value_based_weights(test_function.Y_norm.cpu().numpy(), temp=args.temp)
        test_function.weights = torch.tensor(weights, dtype=dtype, device=device)
        # #NOTE: Testing non-weights
        # test_function.weights = torch.ones_like(test_function.Y_norm)
        data_loader = DataLoader(test_function, batch_size=train_batch_size)
     
        prior_model = ConditionalDiffusionModel(x_dim=dim, diffusion_steps=30, hidden_dim=args.prior_hidden_dim, y_dim=1, dropout_prob=args.dropout_prob).to(dtype=dtype, device=device)
        prior_model_optimizer = torch.optim.Adam(prior_model.parameters(), lr=1e-3)
        for epoch in range(num_epochs):
            total_loss = 0.0
            for x, y, w in data_loader:
                prior_model_optimizer.zero_grad()
                loss = prior_model.compute_loss(x, y, w)
                loss.backward()
                prior_model_optimizer.step()
                total_loss += loss.item()
        print(f"Round: {round+1}\tPrior model trained")
        # dim (bs, y_dim)
        y_target = torch.ones(batch_size, 1, dtype=dtype, device=device) * test_function.Y_norm.max()
        with torch.no_grad():
            X_sample = prior_model.sample(bs = batch_size, y = y_target, gamma = args.gamma, device = device)
        X_sample_unnorm = torch.clamp(X_sample * test_function.X_std + test_function.X_mean, 0.0, 1.0)
        Y_sample_unnorm = torch.tensor([test_function.eval_objective(x) for x in X_sample_unnorm], dtype=dtype, device=device).unsqueeze(-1)
        
        print(f"Query batch size: {batch_size}\tSample size: {X_sample_unnorm.size(0)}")
        print(f"Round: {round+1}\tMax in this round: {Y_sample_unnorm.max().item():.3f}")

        test_function.X = torch.cat([test_function.X, X_sample_unnorm], dim=0)
        test_function.Y = torch.cat([test_function.Y, Y_sample_unnorm], dim=0)
        total_Y = torch.cat([total_Y, Y_sample_unnorm], dim=0)
        print(f"Round: {round+1}\tMax so far: {test_function.Y.max().item():.3f}")
        print(f"Time taken: {time.time() - start_time:.3f} seconds")
        print()

        Y_numpy = total_Y.cpu().numpy()

        if len(Y_numpy) % 1000 == 0:
            np.save(
                f"./baselines/results/ddom/ddom_{args.task}_{dim}_{args.seed}_{n_init}_{args.max_evals}_{len(Y_numpy)}.npy",
                Y_numpy,
            )