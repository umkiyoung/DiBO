import copy

import numpy as np
import torch
import torch.nn as nn
    
class ConditionalQFlowMLP(nn.Module):
    def __init__(self, x_dim, hidden_dim=512, y_dim=1, is_qflow=False, q_net=None, beta=None, dtype = torch.float32, dropout_prob=0.0):
        super(ConditionalQFlowMLP, self).__init__()
        self.is_qflow = is_qflow
        self.q = q_net
        self.beta = beta
        self.y_dim = y_dim
        self.dropout_prob = dropout_prob
        
        self.y_embedding = nn.Sequential(
            nn.Linear(y_dim, 128, dtype=dtype),
            nn.GELU(),
            nn.Linear(128, 128, dtype=dtype),
        )

        self.x_model = nn.Sequential(
            nn.Linear(x_dim + 128 + 128, hidden_dim, dtype=dtype), nn.GELU(), nn.Linear(hidden_dim, hidden_dim, dtype=dtype), nn.GELU()
        )

        self.out_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
            nn.LayerNorm(hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, x_dim, dtype=dtype),
        )

        self.means_scaling_model = nn.Sequential(
            nn.Linear(128, hidden_dim // 2, dtype=dtype),
            nn.LayerNorm(hidden_dim // 2, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2, dtype=dtype),
            nn.LayerNorm(hidden_dim // 2, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, x_dim, dtype=dtype),
        )

        self.harmonics = nn.Parameter(torch.arange(1, 64 + 1, dtype=dtype) * 2 * np.pi).requires_grad_(False)

    def forward(self, x, t, y=None, use_dropout=False, force_unconditional=False):
        # print(x[:4, :4])
        t_fourier1 = (t.unsqueeze(1) * self.harmonics).sin()
        t_fourier2 = (t.unsqueeze(1) * self.harmonics).cos()
        t_emb = torch.cat([t_fourier1, t_fourier2], 1)
        y_emb = self.y_embedding(y)
        if force_unconditional:
            y_emb = y_emb * 0.0
        if use_dropout:
            mask = torch.rand((x.shape[0],128), device=x.device) < self.dropout_prob
            y_emb[mask] = 0.0
        x_emb = self.x_model(torch.cat([x, t_emb, y_emb], 1))
        return self.out_model(x_emb)


class ConditionalDiffusionModel(nn.Module):
    def __init__(self, x_dim, diffusion_steps, schedule="linear", predict="epsilon", policy_net="mlp", hidden_dim=512, y_dim=1, dtype=torch.float32, dropout_prob = 0.0):
        super(ConditionalDiffusionModel, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.diffusion_steps = diffusion_steps
        self.schedule = schedule
        self.dtype = dtype
        self.policy = ConditionalQFlowMLP(x_dim=x_dim, hidden_dim=hidden_dim, y_dim = y_dim, dtype=dtype, dropout_prob=dropout_prob)
        self.diffusion_steps = diffusion_steps
        self.predict = predict
        self.dropout_prob = dropout_prob
        if self.schedule == "linear":
            beta1 = 0.02
            beta2 = 1e-4
            beta_t = (beta1 - beta2) * torch.arange(diffusion_steps + 1, 0, step=-1, dtype=dtype) / (
                diffusion_steps
            ) + beta2
        alpha_t = 1 - torch.flip(beta_t, dims=[0])
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)
        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
        self.register_buffer("beta_t", beta_t)
        self.register_buffer("alpha_t", torch.flip(alpha_t, dims=[0]))
        self.register_buffer("log_alpha_t", torch.flip(log_alpha_t, dims=[0]))
        self.register_buffer("alphabar_t", torch.flip(alphabar_t, dims=[0]))
        self.register_buffer("sqrtab", torch.flip(sqrtab, dims=[0]))
        self.register_buffer("oneover_sqrta", torch.flip(oneover_sqrta, dims=[0]))
        self.register_buffer("sqrtmab", torch.flip(sqrtmab, dims=[0]))
        self.register_buffer("mab_over_sqrtmab_inv", torch.flip(mab_over_sqrtmab_inv, dims=[0]))

    def forward(self, x, t, y, use_dropout=False, force_unconditional=False):
        epsilon = self.policy(x, t, y, use_dropout = use_dropout, force_unconditional=force_unconditional)
        return epsilon

    def sample(self, bs, y, gamma, device):
        x = torch.randn(bs, self.x_dim, dtype=self.dtype, device=device)
        t = torch.zeros((bs,), dtype=self.dtype, device=device)
        dt = 1 / self.diffusion_steps
        for i in range(self.diffusion_steps):
            epsilon_1 = self(x, t, y, use_dropout=False, force_unconditional=True)
            epsilon_2 = self(x, t, y, use_dropout=False, force_unconditional=False)
            epsilon = (1+gamma) * epsilon_2 -gamma * epsilon_1
            if self.predict == "epsilon":
                x = self.oneover_sqrta[i] * (x - self.mab_over_sqrtmab_inv[i] * epsilon) + torch.sqrt(
                    self.beta_t[i]
                ) * torch.randn_like(x, dtype=self.dtype, device=device)
            elif self.predict == "x0":
                x = (1 / torch.sqrt(self.alpha_t[i])) * (
                    (1 - (1 - self.alpha_t[i]) / (1 - self.alphabar_t[i])) * x
                    + ((1 - self.alpha_t[i]) / (1 - self.alphabar_t[i])) * self.sqrtab[i] * epsilon
                ) + torch.sqrt(self.beta_t[i]) * torch.randn_like(x, dtype=self.dtype, device=device)
            t += dt
        return x

    def compute_loss(self, x, y, w_0):
        t_idx = torch.randint(0, self.diffusion_steps, (x.shape[0], 1)).to(x.device)
        t = t_idx.float().squeeze(1) / self.diffusion_steps
        epsilon = torch.randn_like(x, dtype=self.dtype).to(x.device)
        x_t = self.sqrtab[t_idx] * x + self.sqrtmab[t_idx] * epsilon
        epsilon_pred = self(x_t, t, y, use_dropout=True, force_unconditional=False)
        if self.predict == "epsilon":
            w = torch.minimum(
                torch.tensor(5, dtype=self.dtype) / ((self.sqrtab[t_idx] / self.sqrtmab[t_idx]) ** 2), torch.tensor(1, dtype=self.dtype)
            )  # Min-SNR-gamma weights
            loss = (w_0 * w * (epsilon - epsilon_pred) ** 2).mean()
        elif self.predict == "x0":
            w = torch.minimum((self.sqrtab[t_idx] / self.sqrtmab[t_idx]) ** 2, torch.tensor(5, dtype=self.dtype))
            loss = (w_0 * w * (x - epsilon_pred) ** 2).mean()
        return loss

class ConditionalDiffusionModelEns(nn.Module):
    def __init__(self, x_dim, diffusion_steps, schedule="linear", predict="epsilon", policy_net="mlp", num_ensembles = 5, hidden_dim=512, y_dim=1, dtype=torch.float32, dropout_prob = 0.0):
        super(ConditionalDiffusionModelEns, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.diffusion_steps = diffusion_steps
        self.schedule = schedule
        self.dtype = dtype
        self.num_ensembles = num_ensembles
        self.policy = nn.ModuleList([ConditionalQFlowMLP(x_dim=x_dim, hidden_dim=hidden_dim, y_dim = y_dim, dtype=dtype, dropout_prob=dropout_prob) for _ in range(num_ensembles)])
        self.diffusion_steps = diffusion_steps
        self.predict = predict
        self.dropout_prob = dropout_prob
        if self.schedule == "linear":
            beta1 = 0.02
            beta2 = 1e-4
            beta_t = (beta1 - beta2) * torch.arange(diffusion_steps + 1, 0, step=-1, dtype=dtype) / (
                diffusion_steps
            ) + beta2
        alpha_t = 1 - torch.flip(beta_t, dims=[0])
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)
        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
        self.register_buffer("beta_t", beta_t)
        self.register_buffer("alpha_t", torch.flip(alpha_t, dims=[0]))
        self.register_buffer("log_alpha_t", torch.flip(log_alpha_t, dims=[0]))
        self.register_buffer("alphabar_t", torch.flip(alphabar_t, dims=[0]))
        self.register_buffer("sqrtab", torch.flip(sqrtab, dims=[0]))
        self.register_buffer("oneover_sqrta", torch.flip(oneover_sqrta, dims=[0]))
        self.register_buffer("sqrtmab", torch.flip(sqrtmab, dims=[0]))
        self.register_buffer("mab_over_sqrtmab_inv", torch.flip(mab_over_sqrtmab_inv, dims=[0]))

    def forward(self, x, t, y, idx, use_dropout=False, force_unconditional=False): 
        epsilon = self.policy[idx](x, t, y, use_dropout = use_dropout, force_unconditional=force_unconditional)
        return epsilon

    def sample(self, bs, y, idx, gamma, device):
        x = torch.randn(bs, self.x_dim, dtype=self.dtype, device=device)
        t = torch.zeros((bs,), dtype=self.dtype, device=device)
        dt = 1 / self.diffusion_steps
        for i in range(self.diffusion_steps):
            epsilon_1 = self(x, t, y, idx, use_dropout=False, force_unconditional=True)
            epsilon_2 = self(x, t, y, idx, use_dropout=False, force_unconditional=False)
            epsilon = (1+gamma) * epsilon_2 -gamma * epsilon_1
            if self.predict == "epsilon":
                x = self.oneover_sqrta[i] * (x - self.mab_over_sqrtmab_inv[i] * epsilon) + torch.sqrt(
                    self.beta_t[i]
                ) * torch.randn_like(x, dtype=self.dtype, device=device)
            elif self.predict == "x0":
                x = (1 / torch.sqrt(self.alpha_t[i])) * (
                    (1 - (1 - self.alpha_t[i]) / (1 - self.alphabar_t[i])) * x
                    + ((1 - self.alpha_t[i]) / (1 - self.alphabar_t[i])) * self.sqrtab[i] * epsilon
                ) + torch.sqrt(self.beta_t[i]) * torch.randn_like(x, dtype=self.dtype, device=device)
            t += dt
        return x       
    
    def sample_mean(self, bs, y, gamma, device):    
        x = torch.randn(bs, self.x_dim, dtype=self.dtype, device=device)
        t = torch.zeros((bs,), dtype=self.dtype, device=device)
        dt = 1 / self.diffusion_steps
        for i in range(self.diffusion_steps):
            epsilon_1 = torch.zeros_like(x, dtype=self.dtype, device=device)
            epsilon_2 = torch.zeros_like(x, dtype=self.dtype, device=device)
            for j in range(self.num_ensembles):
                epsilon_1 += self(x, t, y, j, use_dropout=False, force_unconditional=True)
                epsilon_2 += self(x, t, y, j, use_dropout=False, force_unconditional=False)
            epsilon_1 /= self.num_ensembles
            epsilon_2 /= self.num_ensembles
            epsilon = (1+gamma) * epsilon_2 -gamma * epsilon_1
            if self.predict == "epsilon":
                x = self.oneover_sqrta[i] * (x - self.mab_over_sqrtmab_inv[i] * epsilon) + torch.sqrt(
                    self.beta_t[i]
                ) * torch.randn_like(x, dtype=self.dtype, device=device)
            elif self.predict == "x0":
                x = (1 / torch.sqrt(self.alpha_t[i])) * (
                    (1 - (1 - self.alpha_t[i]) / (1 - self.alphabar_t[i])) * x
                    + ((1 - self.alpha_t[i]) / (1 - self.alphabar_t[i])) * self.sqrtab[i] * epsilon
                ) + torch.sqrt(self.beta_t[i]) * torch.randn_like(x, dtype=self.dtype, device=device)
            t += dt
        return x
        
    def compute_loss(self, x, y, w_0, idx):
        t_idx = torch.randint(0, self.diffusion_steps, (x.shape[0], 1)).to(x.device)
        t = t_idx.float().squeeze(1) / self.diffusion_steps
        epsilon = torch.randn_like(x, dtype=self.dtype).to(x.device)
        x_t = self.sqrtab[t_idx] * x + self.sqrtmab[t_idx] * epsilon
        epsilon_pred = self(x_t, t, y, idx, use_dropout=True, force_unconditional=False)
        if self.predict == "epsilon":
            w = torch.minimum(
                torch.tensor(5, dtype=self.dtype) / ((self.sqrtab[t_idx] / self.sqrtmab[t_idx]) ** 2), torch.tensor(1, dtype=self.dtype)
            )  # Min-SNR-gamma weights
            loss = (w_0 * w * (epsilon - epsilon_pred) ** 2).mean()
        elif self.predict == "x0":
            w = torch.minimum((self.sqrtab[t_idx] / self.sqrtmab[t_idx]) ** 2, torch.tensor(5, dtype=self.dtype))
            loss = (w_0 * w * (x - epsilon_pred) ** 2).mean()
        return loss

    def get_epistemic_uncertainty(self, y, N, gamma, device):
        ensemble_avgs = []
        for idx in range(self.num_ensembles):
            # sample: shape (N, D)
            sample = self.sample(N, y, idx, gamma, device)
            sample = torch.norm(sample, dim=1)  # shape (N,)
            ensemble_avgs.append(sample.mean(dim=0)) # shape (1,)
        
        # Now ensemble dimension => shape (num_ensembles,)
        ensemble_avgs = torch.stack(ensemble_avgs, dim=0)
        
        # Mean across ensembles => shape (D,)
        mean = ensemble_avgs.mean(dim=0)
        
        # Variance across ensembles => shape (D,)
        variance = (ensemble_avgs - mean).pow(2).mean(dim=0)
        
        return variance