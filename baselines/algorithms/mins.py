import os
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from baselines.models.diffusion import QFlow, DiffusionModel
from baselines.functions.test_function import TestFunction
from baselines.models.value_functions import ProxyEnsemble
from baselines.utils import set_seed
import wandb

# class Generator(nn.Module):
#     def __init__(self, x_dim, y_dim, hidden_dim, num_hidden_layers):
#         super(Generator, self).__init__()
#         self.x_dim = x_dim
#         self.y_dim = y_dim
#         self.hidden_dim = hidden_dim
#         self.num_hidden_layers = num_hidden_layers
        
#         self.y_emb = nn.Linear(y_dim, hidden_dim)
#         self.layers = nn.ModuleList()
#         self.layer_norms = nn.ModuleList()
        
#         self.layers.append(nn.Linear(hidden_dim*2, hidden_dim))
#         self.layer_norms.append(nn.LayerNorm(hidden_dim))
#         for _ in range(num_hidden_layers):
#             self.layers.append(nn.Linear(hidden_dim, hidden_dim))
#             self.layer_norms.append(nn.LayerNorm(hidden_dim))
#         self.decoder = nn.Linear(hidden_dim, x_dim)
        
#     def sample(self, y):
#         y_emb = self.y_emb(y)
#         z = torch.randn(y.shape[0], self.hidden_dim, device=y.device, dtype=y.dtype)
#         # print(z[:4, :4])
#         x = torch.cat([y_emb, z], dim=-1)
#         # print(x[:4, :4])
#         # print(kyle)
#         for layer, layer_norm in zip(self.layers, self.layer_norms):
#             x = layer(x)
#             x = layer_norm(x)
#             x = torch.tanh(x)
#             # x = torch.cat([y_emb, x], dim=-1)
#         x = self.decoder(x)
#         x = torch.tanh(x)
#         # print(x[:4, :4])
#         # print(kyle)
#         return x
    
# class Discriminator(nn.Module):
#     def __init__(self, x_dim, y_dim, hidden_dim, num_hidden_layers):
#         super(Discriminator, self).__init__()
#         self.x_dim = x_dim
#         self.hidden_dim = hidden_dim
#         self.num_hidden_layers = num_hidden_layers
        
#         self.y_emb = nn.Linear(y_dim, hidden_dim)
#         self.layers = nn.ModuleList()
#         self.layers.append(nn.Linear(x_dim + hidden_dim, hidden_dim))
#         self.layers.append(nn.LeakyReLU())
#         for _ in range(num_hidden_layers-1):
#             self.layers.append(nn.Linear(hidden_dim, hidden_dim))
#             self.layers.append(nn.LeakyReLU())
#         self.layers.append(nn.Linear(hidden_dim, y_dim))
        
#     def forward(self, x, y):
#         y_emb = self.y_emb(y)
#         x = torch.cat([y_emb, x], dim=-1)
#         for layer in self.layers:
#             x = layer(x)
#         return torch.sigmoid(x)

class Generator(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.y_emb = nn.Linear(1, 128)
        self.model = nn.Sequential(
            *block(128*2, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, x_dim),
            nn.Tanh()
        )

    def forward(self, z, y):
        y = self.y_emb(y)
        z = torch.cat((z, y), -1)
        img = self.model(z)
        # img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(x_dim + y_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, label):
        img_flat = img.view(img.size(0), -1)
        img_flat = torch.cat((img_flat, label), -1)
        validity = self.model(img_flat)

        return validity
                           

class WeightedGAN(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(WeightedGAN, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        # self.hidden_dim = hidden_dim
        # self.num_hidden_layers = num_hidden_layers
        
        self.generator = Generator(x_dim, y_dim)
        self.discriminator = Discriminator(x_dim, y_dim)
        
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        
        self.criterion = nn.BCELoss()
        
    def train(self, x, y):
        # x_fake = self.generator.sample(y)
        # d_real = self.discriminator(x)
        # d_fake = self.discriminator(x_fake)
        
        # loss = -d_real.mean() + d_fake.mean()
        # return loss
        
        self.generator_optimizer.zero_grad()
        z = torch.randn(y.shape[0], 128, device=y.device, dtype=y.dtype)
        x_fake = self.generator(z, y)
        y_pred_fake = self.discriminator(x_fake, y)
        g_loss = self.criterion(y_pred_fake, torch.ones_like(y_pred_fake))
        g_loss.backward()
        self.generator_optimizer.step()
        
        self.discriminator_optimizer.zero_grad()
        y_pred_real = self.discriminator(x, y)
        loss_real = self.criterion(y_pred_real, torch.ones_like(y_pred_real))
        
        y_pred_fake = self.discriminator(x_fake.detach(), y)
        loss_fake = self.criterion(y_pred_fake, torch.zeros_like(y_pred_fake))
        
        d_loss = loss_real + loss_fake
        d_loss.backward()
        self.discriminator_optimizer.step()
    
        return d_loss.item(), g_loss.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Ackley")
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--n_init", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_evals", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=100)
    # parser.add_argument("--num_proxy_epochs", type=int, default=100)
    # parser.add_argument("--num_prior_epochs", type=int, default=100)
    # parser.add_argument("--num_posterior_epochs", type=int, default=100)
    # parser.add_argument("--dropout", type=bool, default=False)
    # parser.add_argument("--mcmc", type=bool, default=False)
    parser.add_argument("--buffer_size", type=int, default=1000)
    # parser.add_argument("--alpha", type=float, default=0.001)
    # parser.add_argument("--beta", type=float, default=1.0)
    # parser.add_argument("--local_search", type=bool, default=False)
    # parser.add_argument("--local_search_epochs", type=int, default=10)
    # parser.add_argument("--diffusion_steps", type=int, default=30)
    # parser.add_argument("--proxy_hidden_dim", type=int, default=256)
    # parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()
    import os
    if not os.path.exists("./baselines/results"):
        os.makedirs("./baselines/results")
    if not os.path.exists("./baselines/results/mins"):
        os.makedirs("./baselines/results/mins")
    wandb.init(project="mins",
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
    num_epochs = args.num_epochs
    # num_proxy_epochs = args.num_proxy_epochs
    # num_prior_epochs = args.num_prior_epochs
    # num_posterior_epochs = args.num_posterior_epochs
    # reset_counter = 0
    # current_max = -torch.tensor(float("inf"), dtype=dtype, device=device)
    
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
        weights = torch.exp((test_function.Y.squeeze() - test_function.Y.mean()) / (test_function.Y.std() + 1e-7))
        # weights = torch.ones_like(test_function.Y.squeeze())
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        data_loader = DataLoader(test_function, batch_size=train_batch_size, sampler=sampler)
        
        # proxy_model = Proxy(x_dim=dim, hidden_dim=128, dropout_prob=0.1, num_hidden_layers=2).to(dtype=dtype, device=device)
        # proxy_model_ens = ProxyEnsemble(x_dim=dim, hidden_dim=args.proxy_hidden_dim, num_hidden_layers=3, n_ensembles=5, ucb_reward=True).to(dtype=dtype, device=device)
        # proxy_model_ens.gamma = args.gamma
        # for proxy_model in proxy_model_ens.models:
        #     proxy_model_optimizer = torch.optim.Adam(proxy_model.parameters(), lr=1e-3)
        #     for epoch in tqdm(range(num_proxy_epochs), dynamic_ncols=True):
        #         total_loss = 0.0
        #         for x, y in data_loader:
        #             x = (x - test_function.X_mean) / (test_function.X_std + 1e-7)
        #             x += torch.randn_like(x) * 0.001
        #             y = (y - test_function.Y_mean) / (test_function.Y_std + 1e-7)
        #             proxy_model_optimizer.zero_grad()
        #             loss = proxy_model.compute_loss(x, y)
        #             loss.backward()
        #             proxy_model_optimizer.step()
        #             total_loss += loss.item()
        #         # print(f"Round: {round+1}\tEpoch: {epoch+1}\tLoss: {total_loss:.3f}")
        # print(f"Round: {round+1}\tProxy model trained") 
        
        weighted_gan = WeightedGAN(x_dim=dim, y_dim=1).to(dtype=dtype, device=device)
        # for epoch in tqdm(range(num_proxy_epochs), dynamic_ncols=True):
        for epoch in range(num_epochs):
            total_d_loss, total_g_loss = 0.0, 0.0
            for x, y in data_loader:
                # x = (x - test_function.X_mean) / (test_function.X_std + 1e-7)
                # x += torch.randn_like(x) * 0.001
                y = (y - test_function.Y_mean) / (test_function.Y_std + 1e-7)
                d_loss, g_loss = weighted_gan.train(x, y)
                total_d_loss += d_loss
                total_g_loss += g_loss
            # if epoch % 100 == 0:
                # print(f"Round: {round+1}\tEpoch: {epoch+1}\tD Loss: {total_d_loss:.3f}\tG Loss: {total_g_loss:.3f}")
            # print(f"Round: {round+1}\tEpoch: {epoch+1}\tLoss: {total_loss:.3f}")
        # print(kyle)
        y_target = torch.tensor([(y.max().item() - test_function.Y_mean) / (test_function.Y_std + 1e-7)]).to(dtype=dtype, device=device)
        # y_target = torch.tensor([(0.0 - test_function.Y_mean) / (test_function.Y_std + 1e-7)]).to(dtype=dtype, device=device)
        y_target = y_target.repeat(args.batch_size, 1)

        z = torch.randn(args.batch_size, 128, device=device, dtype=dtype)
        x_target = weighted_gan.generator(z, y_target)
        X_sample = x_target.detach()
                
        # There is no big difference in the performance with different diffusion steps
        # prior_model = DiffusionModel(x_dim=dim, diffusion_steps=args.diffusion_steps).to(dtype=dtype, device=device)
        # prior_model.dtype=dtype
        # prior_model_optimizer = torch.optim.Adam(prior_model.parameters(), lr=1e-3)
        # for epoch in tqdm(range(num_prior_epochs), dynamic_ncols=True):
        #     total_loss = 0.0
        #     for x, y in data_loader:
        #         x = (x - test_function.X_mean) / (test_function.X_std + 1e-7)
        #         x += torch.randn_like(x) * 0.001
        #         prior_model_optimizer.zero_grad()
        #         loss = prior_model.compute_loss(x)
        #         loss.backward()
        #         prior_model_optimizer.step()
        #         total_loss += loss.item()
        #     # print(f"Round: {round+1}\tEpoch: {epoch+1}\tLoss: {total_loss:.3f}")
        # print(f"Round: {round+1}\tPrior model trained")

        # x_eval_prior = prior_model.sample(bs=10, device=device)
        # x_eval_prior = x_eval_prior * test_function.X_std + test_function.X_mean
        # y_eval_prior = torch.tensor([test_function.eval_objective(x) for x in x_eval_prior], dtype=dtype, device=device).unsqueeze(-1)
        # print(f"Round: {round+1}:\tPrior max in this round {y_eval_prior.max().item():.3f}")
        # print(f"Round: {round+1}:\tPrior mean in this round {y_eval_prior.mean().item():.3f}")

        # alpha = args.alpha
        # beta = args.beta
        # posterior_model = QFlow(x_dim=dim, diffusion_steps=args.diffusion_steps, q_net=proxy_model_ens, bc_net=prior_model, alpha=alpha, beta=beta).to(dtype=dtype, device=device)
        # posterior_model_optimizer = torch.optim.Adam(posterior_model.parameters(), lr=1e-4)
        
        # xs = torch.tensor(test_function.X, dtype=dtype, device=device)
        # xs = test_function.X.clone().detach()
        # xs = (xs - test_function.X_mean) / (test_function.X_std + 1e-7)
        # ys = proxy_model_ens.log_reward(xs)
        # y_weights = torch.softmax(ys, dim=0)
        
        # for epoch in tqdm(range(num_posterior_epochs), dynamic_ncols=True):
        #     s1 = random.randint(0, 1)
        #     if s1 == 0:
        #         # on-policy
        #         loss, logZ, x, logr = posterior_model.compute_loss(device, gfn_batch_size=train_batch_size)
        #         y = proxy_model_ens.log_reward(x)
        #     else:
        #         # off-policy (reward prioritization)
        #         idx = torch.multinomial(y_weights.squeeze(), train_batch_size, replacement=True)
        #         x = xs[idx]
        #         x += torch.randn_like(x) * 0.01
        #         loss, logZ = posterior_model.compute_loss_with_sample(x, device)
        #         y = proxy_model_ens.log_reward(x)
                
        #     xs = torch.cat([xs, x], dim=0)
        #     ys = torch.cat([ys, y], dim=0)
        #     y_weights = torch.softmax(ys, dim=0)
            
        #     posterior_model_optimizer.zero_grad()
        #     loss.backward()
        #     posterior_model_optimizer.step()                
        #     # print(f"Round: {round+1}\tEpoch: {epoch+1}\tLoss: {total_loss:.3f}")
        # print(f"Round: {round+1}\tPosterior model trained")    
        
        # # x_eval_posterior, _, _ = posterior_model.sample(bs=10, device=device)
        # # x_eval_posterior = x_eval_posterior * test_function.X_std + test_function.X_mean
        # # y_eval_posterior = torch.tensor([test_function.eval_objective(x) for x in x_eval_posterior], dtype=dtype, device=device).unsqueeze(-1)
        # # print(f"Round: {round+1}:\tPosterior max in this round {y_eval_posterior.max().item():.3f}")
        # # print(f"Round: {round+1}:\tPosterior mean in this round {y_eval_posterior.mean().item():.3f}")
        
        
        # # Filtering for selecting candidates
        # X_sample_total = []
        # logR_sample_total = []
        # for _ in tqdm(range(10)):
        #     # Split into batches due to memory constraints
        #     X_sample, logpf_pi, logpf_p = posterior_model.sample(bs=batch_size * 10, device=device)
        #     logpf_pi = posterior_model.compute_marginal_likelihood(X_sample)
        #     logr = posterior_model.posterior_log_reward(X_sample).squeeze()
        #     logR = logr + logpf_pi * alpha
        #     # print(logr[:4], logpf_pi[:4] * alpha)
        #     # print(kyle)
            
        #     X_sample_total.append(X_sample)
        #     logR_sample_total.append(logR)
            
        #     if args.local_search:
        #         X_sample_optimizer = torch.optim.Adam([X_sample], lr=1e-2)
        #         for _ in range(args.local_search_epochs):
        #             X_sample.requires_grad_(True)
        #             logr_sample = posterior_model.posterior_log_reward(X_sample).squeeze()
        #             logpf_pi_sample = posterior_model.compute_marginal_likelihood(X_sample)
        #             logR_sample = logr_sample + logpf_pi_sample * alpha
        #             loss = -logR_sample.sum()
                    
        #             X_sample_optimizer.zero_grad()
        #             loss.backward()
        #             X_sample_optimizer.step()
        #         X_sample = X_sample.detach()
        #         logR_sample = logR_sample.detach()
                
        #         X_sample_total.append(X_sample)
        #         logR_sample_total.append(logR_sample)
            
            
        #     # if args.dropout:
        #     #     X_sample_aug = X_sample.clone().detach()
        #     #     X_current_best_unnorm = test_function.X[np.random.randint(0, args.batch_size)]
        #     #     X_current_best = (X_current_best_unnorm - test_function.X_mean) / (test_function.X_std + 1e-7)
        #     #     idx = np.random.choice(dim, (dim // 4), replace=False)
        #     #     X_sample_aug[:, idx] = X_current_best[idx].repeat(batch_size * 10, 1)
        #     #     logpf_pi_aug = posterior_model.compute_likelihood(X_sample_aug, device)
        #     #     logr_aug = posterior_model.posterior_log_reward(X_sample_aug).squeeze()
        #     #     logR_aug = logr_aug + logpf_pi_aug * alpha
                
        #     #     X_sample_total.append(X_sample_aug)
        #     #     logR_sample_total.append(logR_aug)
        #     #     # X_sample_total.append(X_sample_aug)
            
        #     # if args.mcmc:
        #     #     for _ in range(10):
        #     #         X_sample_bf, logr_bf, logpf_pi_bf = posterior_model.back_and_forth(X_sample, ratio=0.1, device=device)
        #     #         logR_bf = logr_bf + logpf_pi_bf * alpha
                
        #     #         X_sample_total.append(X_sample_bf)
        #     #         logR_sample_total.append(logR_bf)
                    
        #     #         idx = logR_bf > logR
        #     #         X_sample[idx] = X_sample_bf[idx]
                    
        # X_sample = torch.cat(X_sample_total, dim=0)
        # logR_sample = torch.cat(logR_sample_total, dim=0)
        
        # X_sample = X_sample[torch.argsort(logR_sample, descending=True)][:batch_size]
        # logR_sample = logR_sample[torch.argsort(logR_sample, descending=True)][:batch_size]
        
        # # if args.mcmc:
        # #     for _ in range(10):
        # #         X_sample_bf, logr_bf, logpf_pi_bf = posterior_model.back_and_forth(X_sample, ratio=0.1, device=device)
        # #         logR_bf = logr_bf + logpf_pi_bf * alpha
        # #         idx = logR_bf > logR_sample
        # #         X_sample[idx] = X_sample_bf[idx]
        
        # # if args.local_search:
        # #     X_sample_optimizer = torch.optim.Adam([X_sample], lr=1e-3)
        # #     for _ in range(100):
        # #         X_sample.requires_grad_(True)
        # #         logR_sample = posterior_model.posterior_log_reward(X_sample).squeeze()
        # #         logpf_pi_sample = posterior_model.compute_likelihood(X_sample, device)
        # #         logR_sample = logR_sample + logpf_pi_sample * alpha
        # #         loss = -logR_sample.sum()
                
        # #         X_sample_optimizer.zero_grad()
        # #         loss.backward()
        # #         X_sample_optimizer.step()
        # X_sample = X_sample.detach()
        
        # print(f"Round: {round+1}\tSampling done")

        # X_sample_unnorm = X_sample * test_function.X_std + test_function.X_mean
        # X_sample_unnorm = torch.clamp(X_sample_unnorm, 0.0, 1.0)
        X_sample_unnorm = torch.clamp(X_sample, 0.0, 1.0)
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
                #    "beta": beta,
                   "histogram": wandb.Histogram(Y_sample_unnorm.cpu().numpy().flatten())
                })
        
        
        # counter
        # if Y_sample_unnorm.max() > current_max:
            # current_max = Y_sample_unnorm.max()
            # reset_counter = 0
        # else:
            # reset_counter += 1
        # if reset_counter >= 5 and len(Y_total) < args.max_evals - n_init:
            # test_function.X, test_function.Y = test_function.reset()
            # reset_counter = 0
            # current_max = -torch.tensor(float("inf"), dtype=dtype, device=device)
            # print("Reset")
# 
            # X_total = np.concatenate([X_total, test_function.X.cpu().numpy()], axis=0)
            # Y_total = np.concatenate([Y_total, test_function.Y.cpu().numpy()], axis=0)

        
        
        
        #if (round + 1) % 10 == 0:
        #    if not os.path.exists(f"./baselines/results/{task}_dim{dim}"):
        #        os.makedirs(f"results/{task}_dim{dim}", exist_ok=True)
        #    np.savez_compressed(f"results/{task}_dim{dim}/ninit{n_init}_query{batch_size}_round{round+1}_seed{seed}_epoch{num_posterior_epochs}_mcmc{args.mcmc}_dropout{args.dropout}.npz", X=X_total, Y=Y_total)
        
        
        if len(Y_total) % 1000 == 0:
            if not os.path.exists(f"./baselines/results/mins"):
                os.makedirs(f"./baselines/results/mins", exist_ok=True)
            np.save(
                f"./baselines/results/mins/mins_{task}_{dim}_{seed}_{n_init}_{args.max_evals}_{len(Y_total)}.npy",
                np.array(Y_total),
            )