import os
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from baselines.models.diffusion import QFlow, DiffusionModel
from baselines.functions.test_function import TestFunction
from baselines.models.value_functions import ProxyEnsemble, ProxyMCDropout, Proxy 
from baselines.utils import set_seed, get_value_based_weights, get_rank_based_weights
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Ackley")
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--n_init", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_evals", type=int, default=10000)
    parser.add_argument("--num_proxy_epochs", type=int, default=100)
    parser.add_argument("--num_prior_epochs", type=int, default=100)
    parser.add_argument("--num_posterior_epochs", type=int, default=100)
    parser.add_argument("--buffer_size", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--local_search", type=str, default="True") # True, False   
    parser.add_argument("--local_search_epochs", type=int, default=10)
    parser.add_argument("--diffusion_steps", type=int, default=30)
    parser.add_argument("--proxy_hidden_dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--num_ensembles", type=int, default=5)
    parser.add_argument("--reweighting", type=str, default="exp") # exp, uniform, value, rank
    parser.add_argument("--filtering", type=str, default='True') # True, False
    parser.add_argument("--num_proposals", type=int, default=10)
    parser.add_argument("--training_posterior", type=str, default='both') # both, on, off
    parser.add_argument("--uncertainty_estimation", type=str, default='ensemble') # ensemble, dropout, None
    parser.add_argument("--abalation", type=str, default="")
    parser.add_argument("--test_mode", type=str, default="False")
    args = parser.parse_args()
    if args.test_mode == "True":
        print("Test mode")
        args.max_evals = 1000
        args.num_proxy_epochs = 1
        args.num_prior_epochs = 1
        args.num_posterior_epochs = 1
        if args.local_search_epochs > 1:
            args.local_search_epochs = 1
    
    import os
    if not os.path.exists("./baselines/results"):
        os.makedirs("./baselines/results")
    if not os.path.exists("./baselines/results/pibo"):
        os.makedirs("./baselines/results/pibo")
    wandb.init(project="posterior-inference-bo",
               config=vars(args))
    
    task = args.task
    dim = args.dim
    train_batch_size = args.train_batch_size
    batch_size = args.batch_size
    n_init = args.n_init
    seed = args.seed
    dtype = torch.double
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(seed)

    test_function = TestFunction(task = task, dim = dim, n_init = n_init, seed = seed, dtype=dtype, device=device)

    num_rounds = (args.max_evals - n_init) // batch_size
    #num_epochs = args.num_epochs
    num_proxy_epochs = args.num_proxy_epochs
    num_prior_epochs = args.num_prior_epochs
    num_posterior_epochs = args.num_posterior_epochs
    
    X_total = test_function.X.cpu().numpy()
    Y_total = test_function.Y.cpu().numpy()
    for round in range(num_rounds):
        start_time = time.time()
        test_function.X_mean = test_function.X.mean(dim=0)
        test_function.X_std = test_function.X.std(dim=0)
        
        test_function.Y_mean = test_function.Y.mean()
        test_function.Y_std = test_function.Y.std()
        
        # Re-weighting for the training set (it seems cruical for the performance)
        # Prior implementation is for offline setting, so we should consider low-scoring regions to prevent deviation from the offline dataset
        # However, it is not necessary for online setting
        if args.reweighting == "uniform":
            weights = torch.ones_like(test_function.Y.squeeze())
        elif args.reweighting == "value":
            weights = get_value_based_weights(test_function.Y.cpu().numpy(), temp="90")
            weights = torch.tensor(weights, dtype=dtype, device=device).squeeze()
        elif args.reweighting == "rank":
            weights = get_rank_based_weights(test_function.Y.squeeze().cpu().numpy())
        else: #TODO Ours. exp.
            weights = torch.exp((test_function.Y.squeeze() - test_function.Y.mean()) / (test_function.Y.std() + 1e-7))
            
        # weights = torch.ones_like(test_function.Y.squeeze())
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        data_loader = DataLoader(test_function, batch_size=train_batch_size, sampler=sampler)
        
        # proxy_model = Proxy(x_dim=dim, hidden_dim=128, dropout_prob=0.1, num_hidden_layers=2).to(dtype=dtype, device=device)
        if args.uncertainty_estimation == "ensemble":
            proxy_model_ens = ProxyEnsemble(x_dim=dim, hidden_dim=args.proxy_hidden_dim, num_hidden_layers=3, n_ensembles=args.num_ensembles, ucb_reward=True).to(dtype=dtype, device=device)
            proxy_model_ens.gamma = args.gamma
            for proxy_model in proxy_model_ens.models:
                proxy_model_optimizer = torch.optim.Adam(proxy_model.parameters(), lr=1e-3)
                for epoch in tqdm(range(num_proxy_epochs), dynamic_ncols=True):
                    total_loss = 0.0
                    for x, y in data_loader:
                        x = (x - test_function.X_mean) / (test_function.X_std + 1e-7)
                        x += torch.randn_like(x) * 0.001
                        y = (y - test_function.Y_mean) / (test_function.Y_std + 1e-7)
                        proxy_model_optimizer.zero_grad()
                        loss = proxy_model.compute_loss(x, y)
                        loss.backward()
                        proxy_model_optimizer.step()
                        total_loss += loss.item()
        else:
            if args.uncertainty_estimation == "dropout":
                proxy_model_ens = ProxyMCDropout(x_dim=dim, hidden_dim=args.proxy_hidden_dim, num_hidden_layers=3, dropout_rate=0.1, dtype=dtype).to(dtype=dtype, device=device)
                proxy_model_ens.gamma = args.gamma
            else:
                proxy_model_ens = Proxy(x_dim=dim, hidden_dim=args.proxy_hidden_dim, num_hidden_layers=3).to(dtype=dtype, device=device)
                proxy_model_ens.gamma = 0.0 
                proxy_model_optimizer = torch.optim.Adam(proxy_model_ens.parameters(), lr=1e-3)
                for epoch in tqdm(range(num_proxy_epochs), dynamic_ncols=True):
                    total_loss = 0.0
                    for x, y in data_loader:
                        x = (x - test_function.X_mean) / (test_function.X_std + 1e-7)
                        x += torch.randn_like(x) * 0.001
                        y = (y - test_function.Y_mean) / (test_function.Y_std + 1e-7)
                        proxy_model_optimizer.zero_grad()
                        loss = proxy_model_ens.compute_loss(x, y)
                        loss.backward()
                        proxy_model_optimizer.step()
                        total_loss += loss.item()
        print(f"Round: {round+1}\tProxy model trained")
        
        
        # There is no big difference in the performance with different diffusion steps
        prior_model = DiffusionModel(x_dim=dim, diffusion_steps=args.diffusion_steps).to(dtype=dtype, device=device)
        prior_model.dtype=dtype
        prior_model_optimizer = torch.optim.Adam(prior_model.parameters(), lr=1e-3)
        for epoch in tqdm(range(num_prior_epochs), dynamic_ncols=True):
            total_loss = 0.0
            for x, y, in data_loader:
                x = (x - test_function.X_mean) / (test_function.X_std + 1e-7)
                x += torch.randn_like(x) * 0.001
                prior_model_optimizer.zero_grad()
                loss = prior_model.compute_loss(x)
                loss.backward()
                prior_model_optimizer.step()
                total_loss += loss.item()
            # print(f"Round: {round+1}\tEpoch: {epoch+1}\tLoss: {total_loss:.3f}")
        print(f"Round: {round+1}\tPrior model trained")

        alpha = args.alpha
        beta = args.beta
        posterior_model = QFlow(x_dim=dim, diffusion_steps=args.diffusion_steps, q_net=proxy_model_ens, bc_net=prior_model, alpha=alpha, beta=beta).to(dtype=dtype, device=device)
        posterior_model_optimizer = torch.optim.Adam(posterior_model.parameters(), lr=1e-4)
        
        # xs = torch.tensor(test_function.X, dtype=dtype, device=device)
        xs = test_function.X.clone().detach()
        xs = (xs - test_function.X_mean) / (test_function.X_std + 1e-7)
        ys = proxy_model_ens.log_reward(xs)
        y_weights = torch.softmax(ys, dim=0)
        
        if num_posterior_epochs > 0:
            for epoch in tqdm(range(num_posterior_epochs), dynamic_ncols=True):
                if args.training_posterior == "both":
                    s1 = random.randint(0, 1)
                elif args.training_posterior == "on":
                    s1 = 0
                else: # off
                    s1 = 1
                    
                if s1 == 0:
                    # on-policy
                    loss, logZ, x, logr = posterior_model.compute_loss(device, gfn_batch_size=train_batch_size)
                    y = proxy_model_ens.log_reward(x)
                else:
                    # off-policy (reward prioritization)
                    idx = torch.multinomial(y_weights.squeeze(), train_batch_size, replacement=True)
                    x = xs[idx]
                    x += torch.randn_like(x) * 0.01
                    loss, logZ = posterior_model.compute_loss_with_sample(x, device)
                    y = proxy_model_ens.log_reward(x)
                    
                xs = torch.cat([xs, x], dim=0)
                ys = torch.cat([ys, y], dim=0)
                y_weights = torch.softmax(ys, dim=0)
                
                posterior_model_optimizer.zero_grad()
                loss.backward()
                posterior_model_optimizer.step()                
                # print(f"Round: {round+1}\tEpoch: {epoch+1}\tLoss: {total_loss:.3f}")
            print(f"Round: {round+1}\tPosterior model trained")    
        
        
        # Filtering for selecting candidates
        X_sample_total = []
        logR_sample_total = []
        if args.filtering == "True":
            M = args.num_proposals
        else: # 
            M = 1
            
        for _ in tqdm(range(M)): #NOTE M**2 samples proposal.
            # Split into batches due to memory constraints
            X_sample, logpf_pi, logpf_p = posterior_model.sample(bs=batch_size * M, device=device)
            logpf_pi = posterior_model.compute_marginal_likelihood(X_sample)
            logr = posterior_model.posterior_log_reward(X_sample).squeeze()
            logR = logr + logpf_pi * alpha
            
            if args.local_search == "True" and args.local_search_epochs > 0:
                X_sample_optimizer = torch.optim.Adam([X_sample], lr=1e-2)
                for _ in range(args.local_search_epochs):
                    X_sample.requires_grad_(True)
                    logr_sample = posterior_model.posterior_log_reward(X_sample).squeeze()
                    logpf_pi_sample = posterior_model.compute_marginal_likelihood(X_sample)
                    logR_sample = logr_sample + logpf_pi_sample * alpha
                    loss = -torch.exp(logR_sample).sum()
                    # loss = -logR_sample.sum()
                    
                    X_sample_optimizer.zero_grad()
                    loss.backward()
                    X_sample_optimizer.step()
                X_sample = X_sample.detach()
                logR_sample = logR_sample.detach()
                
                X_sample_total.append(X_sample)
                logR_sample_total.append(logR_sample)
            else:
                X_sample_total.append(X_sample)
                logR_sample_total.append(logR)
            
                    
        X_sample = torch.cat(X_sample_total, dim=0)
        logR_sample = torch.cat(logR_sample_total, dim=0)
        
        X_sample = X_sample[torch.argsort(logR_sample, descending=True)][:batch_size]
        logR_sample = logR_sample[torch.argsort(logR_sample, descending=True)][:batch_size]

        X_sample = X_sample.detach()
        
        print(f"Round: {round+1}\tSampling done")

        X_sample_unnorm = X_sample * test_function.X_std + test_function.X_mean
        X_sample_unnorm = torch.clamp(X_sample_unnorm, 0.0, 1.0)
        Y_sample_unnorm = torch.tensor([test_function.eval_objective(x) for x in X_sample_unnorm], dtype=dtype, device=device).unsqueeze(-1)        
        print(f"Round: {round+1}\tSeed: {seed}\tMax in this round: {Y_sample_unnorm.max().item():.3f}")
        
        test_function.X = torch.cat([test_function.X, X_sample_unnorm], dim=0)
        test_function.Y = torch.cat([test_function.Y, Y_sample_unnorm], dim=0)
        print(f"Round: {round+1}\tMax so far: {test_function.Y.max().item():.3f}")
        print(f"Time taken: {time.time() - start_time:.3f} seconds")
        print()
        
        # Remove low-score samples in the training set (it seems cruical for the performance)
        idx = torch.argsort(test_function.Y.squeeze(), descending=True)[:args.buffer_size]
        test_function.X = test_function.X[idx]
        test_function.Y = test_function.Y[idx]
        print(len(test_function.X))
        X_total = np.concatenate([X_total, X_sample_unnorm.cpu().numpy()], axis=0)
        Y_total = np.concatenate([Y_total, Y_sample_unnorm.cpu().numpy()], axis=0)
        
        wandb.log({"round": round, 
                   "max_in_this_round": Y_sample_unnorm.max().item(), 
                   "max_so_far": test_function.Y.max().item(),
                   "time_taken": time.time() - start_time,
                   "num_samples": X_total.shape[0],
                   "beta": beta,
                   "histogram": wandb.Histogram(Y_sample_unnorm.cpu().numpy().flatten())
                })
        
        if len(Y_total) >= 1000:
            save_len = min(len(Y_total) // 1000 * 1000, args.max_evals)
            save_np = Y_total[:save_len]
        
            if args.abalation == "":
                if not os.path.exists(f"./baselines/results/pibo"):
                    os.makedirs(f"./baselines/results/pibo", exist_ok=True)
                np.save(
                    f"./baselines/results/pibo/pibo_{task}_{dim}_{seed}_{n_init}_{args.batch_size}_{args.buffer_size}_{args.local_search_epochs}_{args.max_evals}_{save_len}.npy",
                    np.array(save_np),
                )
            else:
                if not os.path.exists(f"./baselines/results/abalations"):
                    os.makedirs(f"./baselines/results/abalations", exist_ok=True)
                if not os.path.exists(f"./baselines/results/abalations/{args.abalation}"):
                    os.makedirs(f"./baselines/results/abalations/{args.abalation}", exist_ok=True)
                if args.uncertainty_estimation == "ensemble":
                    np.save(
                        f"./baselines/results/abalations/{args.abalation}/{task}_{dim}_{seed}_{n_init}_{args.batch_size}_{args.buffer_size}_{args.diffusion_steps}_{args.alpha}_{args.gamma}_{args.num_ensembles}_{args.local_search}_{args.local_search_epochs}_{args.reweighting}_{args.filtering}_{args.training_posterior}_{args.num_proposals}_{args.num_posterior_epochs}_{args.max_evals}_{save_len}.npy",
                        np.array(save_np),
                    )
                else:
                    np.save(
                        f"./baselines/results/abalations/{args.abalation}/{task}_{dim}_{seed}_{n_init}_{args.batch_size}_{args.buffer_size}_{args.diffusion_steps}_{args.alpha}_{args.gamma}_{args.num_ensembles}_{args.local_search}_{args.local_search_epochs}_{args.reweighting}_{args.filtering}_{args.training_posterior}_{args.num_proposals}_{args.num_posterior_epochs}_{args.uncertainty_estimation}_{args.max_evals}_{save_len}.npy",
                        np.array(save_np),
                    )
