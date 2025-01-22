import copy

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

from torch.nn import functional as F
    
class ResidualBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation: str = "relu", layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=True)
        if layer_norm:
            self.ln = nn.LayerNorm(dim_in)
        else:
            self.ln = torch.nn.Identity()
        self.activation = getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.activation(self.ln(x)))
    
    
class ResidualMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            width: int,
            depth: int,
            output_dim: int,
            activation: str = "gelu",
            layer_norm: bool = False,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            *[ResidualBlock(width, width, activation, layer_norm) for _ in range(depth)],
            nn.LayerNorm(width) if layer_norm else torch.nn.Identity(),
        )

        self.activation = getattr(F, activation)
        self.final_linear = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_linear(self.activation(self.network(x)))    

    
class QFlowMLP(nn.Module):
    def __init__(self, x_dim, hidden_dim=512, is_qflow=False, q_net=None, beta=None, dtype = torch.float32):
        super(QFlowMLP, self).__init__()
        self.is_qflow = is_qflow
        self.q = q_net
        self.beta = beta

        # self.x_model = nn.Sequential(
        #     nn.Linear(x_dim + 128, hidden_dim, dtype=dtype), nn.GELU(), nn.Linear(hidden_dim, hidden_dim, dtype=dtype), nn.GELU()
        # )

        # self.out_model = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
        #     nn.LayerNorm(hidden_dim, dtype=dtype),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, x_dim, dtype=dtype),
        # )

        self.proj = nn.Linear(x_dim, hidden_dim)
        self.residual_mlp = ResidualMLP(
            input_dim=hidden_dim + 128,
            width=hidden_dim,
            depth=3,
            output_dim=x_dim,
            activation="gelu",
            layer_norm=True,
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

    def forward(self, x, t):
        t_fourier1 = (t.unsqueeze(1) * self.harmonics).sin()
        t_fourier2 = (t.unsqueeze(1) * self.harmonics).cos()
        t_emb = torch.cat([t_fourier1, t_fourier2], 1)
        # if not self.is_qflow:
        #     x_emb = self.x_model(torch.cat([x, t_emb], 1))
        # if self.is_qflow:
        #     # with torch.no_grad():
        #     x_emb = self.x_model(torch.cat([x, t_emb], 1))
        #     with torch.enable_grad():
        #         x.requires_grad_(True)
        #         means_scaling = self.means_scaling_model(t_emb) * self.q.score(x, beta=self.beta)
        #     return self.out_model(x_emb) + means_scaling
        # return self.out_model(x_emb)
        # x = self.proj(x) + t_emb
        x = torch.cat([self.proj(x), t_emb], 1)
        return self.residual_mlp(x)


class DiffusionModel(nn.Module):
    def __init__(self, x_dim, diffusion_steps, schedule="linear", predict="epsilon", policy_net="mlp", hidden_dim=512, dtype=torch.float32):
        super(DiffusionModel, self).__init__()
        self.x_dim = x_dim
        self.diffusion_steps = diffusion_steps
        self.schedule = schedule
        self.dtype = dtype
        self.policy = QFlowMLP(x_dim=x_dim, hidden_dim=hidden_dim, dtype=dtype)
        self.diffusion_steps = diffusion_steps
        self.predict = predict
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

    def forward(self, x, t):
        epsilon = self.policy(x, t)
        return epsilon

    def score(self, x, t):
        t_idx = (t * self.diffusion_steps).long().unsqueeze(1)
        epsilon = self(x, t)
        if self.predict == "epsilon":
            score = -epsilon / self.sqrtmab[t_idx]
        elif self.predict == "x0":
            score = (self.sqrtab[t_idx] * epsilon - x) / (1 - self.alphabar_t[t_idx])
        return score

    def sample(self, bs, device):
        x = torch.randn(bs, self.x_dim, dtype=self.dtype, device=device)
        t = torch.zeros((bs,), dtype=self.dtype, device=device)
        dt = 1 / self.diffusion_steps
        for i in range(self.diffusion_steps):
            epsilon = self(x, t)
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

    def compute_loss(self, x):
        t_idx = torch.randint(0, self.diffusion_steps, (x.shape[0], 1)).to(x.device)
        t = t_idx.float().squeeze(1) / self.diffusion_steps
        epsilon = torch.randn_like(x, dtype=self.dtype).to(x.device)
        x_t = self.sqrtab[t_idx] * x + self.sqrtmab[t_idx] * epsilon
        epsilon_pred = self(x_t, t)
        if self.predict == "epsilon":
            w = torch.minimum(
                torch.tensor(5, dtype=self.dtype) / ((self.sqrtab[t_idx] / self.sqrtmab[t_idx]) ** 2), torch.tensor(1, dtype=self.dtype)
            )  # Min-SNR-gamma weights
            loss = (w * (epsilon - epsilon_pred) ** 2).mean()
        elif self.predict == "x0":
            w = torch.minimum((self.sqrtab[t_idx] / self.sqrtmab[t_idx]) ** 2, torch.tensor(5, dtype=self.dtype))
            loss = (w * (x - epsilon_pred) ** 2).mean()
        return loss

class QFlow(nn.Module):
    def __init__(
        self,
        x_dim,
        diffusion_steps,
        schedule="linear",
        # predict="epsilon",
        q_net=None,
        bc_net=None,
        alpha=1.0,
        beta=1.0,
        dtype=torch.float64,
    ):
        super(QFlow, self).__init__()
        self.x_dim = x_dim
        self.diffusion_steps = diffusion_steps
        self.schedule = schedule
        # self.predict = predict
        self.logZ = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.q_net = q_net
        self.bc_net = bc_net
        self.qflow = copy.deepcopy(bc_net.policy)
        # self.qflow.is_qflow = True  # This makes things more than 1.5x slower
        self.qflow.q = q_net
        self.qflow.beta = beta

        self.alpha = alpha
        self.beta = beta
        self.dtype = dtype

    def forward(self, x, t):
        q_epsilon = self.qflow(x, t)
        with torch.no_grad():
            bc_epsilon = self.bc_net(x, t).detach()
        return q_epsilon, bc_epsilon

    def sample(self, bs, device, extra=False):
        normal_dist = torch.distributions.Normal(
            torch.zeros((bs, self.x_dim), device=device, dtype=self.dtype), 
            torch.ones((bs, self.x_dim), device=device, dtype=self.dtype)
        )
        x = normal_dist.sample()
        t = torch.zeros((bs,), device=device, dtype=self.dtype)
        dt = 1 / self.diffusion_steps

        logpf_pi = normal_dist.log_prob(x).sum(1)
        logpf_p = normal_dist.log_prob(x).sum(1)
        # print(logpf_pi[:4])
        extra_steps = 1
        if extra:
            extra_steps = 20
        for i in range(self.diffusion_steps):
            for j in range(extra_steps):
                q_epsilon, bc_epsilon = self(x, t)

                epsilon = q_epsilon + bc_epsilon
                new_x = self.bc_net.oneover_sqrta[i] * (
                    x - self.bc_net.mab_over_sqrtmab_inv[i] * epsilon.detach()
                ) + torch.sqrt(self.bc_net.beta_t[i]) * torch.randn_like(x, dtype=self.dtype)

                pf_pi_dist = torch.distributions.Normal(
                    self.bc_net.oneover_sqrta[i] * (x - self.bc_net.mab_over_sqrtmab_inv[i] * bc_epsilon),
                    torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(x, dtype=self.dtype),
                )
                logpf_pi += pf_pi_dist.log_prob(new_x).sum(1)

                pf_p_dist = torch.distributions.Normal(
                    self.bc_net.oneover_sqrta[i] * (x - self.bc_net.mab_over_sqrtmab_inv[i] * epsilon),
                    torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(new_x, dtype=self.dtype),
                )
                logpf_p += pf_p_dist.log_prob(new_x).sum(1)

                x = new_x
                if i < self.diffusion_steps - 1:
                    break
            t = t + dt
        return x, logpf_pi, logpf_p
    
    def back_and_forth(self, x, ratio, device):
        # Back
        t = torch.ones((x.shape[0],), device=device, dtype=self.dtype) * (1.0 - ratio)
        t_idx = (t * self.diffusion_steps).to(dtype=torch.long).unsqueeze(1)
        epsilon = torch.randn_like(x, dtype=self.dtype).to(x.device)
        x = self.bc_net.sqrtab[t_idx] * x + self.bc_net.sqrtmab[t_idx] * epsilon
        
        # Forth
        dt = 1 / self.diffusion_steps
        for i in range(int(self.diffusion_steps * (1.0 - ratio)), self.diffusion_steps):
            q_epsilon, bc_epsilon = self(x, t)
            epsilon = q_epsilon + bc_epsilon
            
            new_x = self.bc_net.oneover_sqrta[i] * (
                    x - self.bc_net.mab_over_sqrtmab_inv[i] * epsilon.detach()
                ) + torch.sqrt(self.bc_net.beta_t[i]) * torch.randn_like(x, dtype=self.dtype)

            x = new_x
            t = t + dt
        
        x_gen = x.clone().detach()
            
        # Compute the log probability
        t = torch.zeros((x.shape[0],), device=device, dtype=self.dtype)
        logr = self.posterior_log_reward(x)
        logpf_pi = torch.zeros((x.shape[0],), device=device, dtype=self.dtype)
        for i in range(self.diffusion_steps - 1, -1, -1):
            pb_dist = torch.distributions.Normal(
                torch.sqrt(self.bc_net.alpha_t[i]) * x,
                torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(x, dtype=self.dtype),
            )
            new_x = pb_dist.sample()
            
            q_epsilon, bc_epsilon = self(new_x, t + i * dt)
            epsilon = q_epsilon + bc_epsilon
            
            pf_pi_dist = torch.distributions.Normal(
                self.bc_net.oneover_sqrta[i] * (new_x - self.bc_net.mab_over_sqrtmab_inv[i] * bc_epsilon),
                torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(new_x, dtype=self.dtype),
            )
            logpf_pi += pf_pi_dist.log_prob(x).sum(1)
            
            x = new_x
        prior_dist = torch.distributions.Normal(torch.zeros_like(x, dtype=self.dtype), torch.ones_like(x, dtype=self.dtype))
        logpf_pi += prior_dist.log_prob(x).sum(1)
        return x_gen, logr, logpf_pi * self.alpha
    
    def compute_likelihood(self, x, device):
        dt = 1 / self.diffusion_steps
        t = torch.zeros((x.shape[0],), device=device, dtype=self.dtype)
        logpf_pi = torch.zeros((x.shape[0],), device=device, dtype=self.dtype)
        for i in range(self.diffusion_steps - 1, -1, -1):
            pb_dist = torch.distributions.Normal(
                torch.sqrt(self.bc_net.alpha_t[i]) * x,
                torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(x, dtype=self.dtype),
            )
            new_x = pb_dist.sample()
            
            q_epsilon, bc_epsilon = self(new_x, t + i * dt)
            epsilon = q_epsilon + bc_epsilon
            
            pf_pi_dist = torch.distributions.Normal(
                self.bc_net.oneover_sqrta[i] * (new_x - self.bc_net.mab_over_sqrtmab_inv[i] * bc_epsilon),
                torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(new_x, dtype=self.dtype),
            )
            logpf_pi += pf_pi_dist.log_prob(x).sum(1)
            
            x = new_x
        prior_dist = torch.distributions.Normal(torch.zeros_like(x, dtype=self.dtype), torch.ones_like(x, dtype=self.dtype))
        logpf_pi += prior_dist.log_prob(x).sum(1)
        return logpf_pi
    
    def get_sigma(self):
        beta1 = 0.02
        beta2 = 1e-4
        beta_t = (beta1 - beta2) * torch.arange(self.diffusion_steps + 1, 0, step=-1, dtype=self.dtype) / (
            self.diffusion_steps
        ) + beta2
        alpha_t = 1 - torch.flip(beta_t, dims=[0]) #alphas
        log_alpha_t = torch.log(alpha_t) #log_alphas
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp() #alphas_cumprod
        sigmas = torch.sqrt((1 - alphabar_t) / alphabar_t) #sqrt((1-alphas_cumprod)/alphas_cumprod)
        log_sigmas = torch.log(sigmas)
        return sigmas[-1], sigmas[0], log_sigmas

    # @torch.no_grad()
    def compute_marginal_likelihood(self, x):
        v = torch.randint_like(x, 2) * 2 -1
        sigma_min, sigma_max, log_sigmas = self.get_sigma()
        log_sigmas = log_sigmas.to(x.device)
        model = self.bc_net
        def sigma_to_t(sigma, log_sigmas, x):
            # get log sigma
            log_sigma = torch.log(sigma)

            # get distribution
            dists = log_sigma - log_sigmas[:, None]

            # get sigmas range
            low_idx = torch.cumsum((dists >= 0), dim=0).argmax(dim=0).clamp(max=log_sigmas.shape[0] - 2)
            high_idx = low_idx + 1

            low = log_sigmas[low_idx]
            high = log_sigmas[high_idx]

            # interpolate sigmas
            w = (low - log_sigma) / (low - high)
            w = torch.clamp(w, 0, 1)

            # transform interpolation to time range
            t = (1 - w) * low_idx + w * high_idx
            t = t / self.diffusion_steps
            t = torch.ones((x.shape[0],), device=x.device, dtype=self.dtype) * t
            return t
        
        class ODEfunc(torch.nn.Module):
            def __init__(self):
                super(ODEfunc, self).__init__()

                self.nfev = 0

            def forward(self, sigma, x):
                with torch.enable_grad():
                    x = x[0].requires_grad_()

                    x = x.to(dtype=torch.float64)
                    x = x / ((sigma**2 + 1) ** 0.5)

                    t = sigma_to_t(sigma, log_sigmas, x)

                    # predict the noise residual
                    noise_pred = model(x,t)

                    noise_pred = noise_pred.to(dtype=torch.float64)

                    d = noise_pred
                    
                    x_clone = x.clone().detach().requires_grad_()
                    d_clone = model(x_clone,t)
                    grad = torch.autograd.grad((d_clone * v).sum(), x_clone)[0].detach()
                    d_ll = (v * grad).flatten(1).sum(1)
                self.nfev += 1

                return d, d_ll
            
        x_min = x, x.new_zeros([x.shape[0]])
        t = x.new_tensor([sigma_min, sigma_max])
        ode_func = ODEfunc().cuda()

        method = "rk4"
        atol = 1e-5
        rtol = 1e-5
        step_size = abs(sigma_min - sigma_max) / 4
        sol = odeint(ode_func, x_min, t, atol=atol, rtol=rtol, method=method)
        
        latent, delta_ll = sol[0][-1], sol[1][-1]
        ll_prior = torch.distributions.Normal(0, sigma_max).log_prob(latent).flatten(1).sum(1)
        return ll_prior + delta_ll
        
    def posterior_log_reward(self, x):
        q_r = self.q_net.log_reward(x, beta=self.beta).squeeze()
        return q_r

    def compute_loss_with_sample(self, x, device):
        bs = x.shape[0]
        # minlogvar, maxlogvar = -4, 4
        t = torch.zeros((bs,), device=device, dtype=self.dtype)
        dt = 1 / self.diffusion_steps

        logpf_pi = torch.zeros((bs,), device=device, dtype=self.dtype)
        logpf_p = torch.zeros((bs,), device=device, dtype=self.dtype)
        logr = self.posterior_log_reward(x)

        for i in range(self.diffusion_steps - 1, -1, -1):
            pb_dist = torch.distributions.Normal(
                torch.sqrt(self.bc_net.alpha_t[i]) * x,
                torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(x, dtype=self.dtype),
            )
            new_x = pb_dist.sample()

            q_epsilon, bc_epsilon = self(new_x, t + i * dt)
            epsilon = q_epsilon + bc_epsilon

            pf_pi_dist = torch.distributions.Normal(
                self.bc_net.oneover_sqrta[i] * (new_x - self.bc_net.mab_over_sqrtmab_inv[i] * bc_epsilon),
                torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(new_x, dtype=self.dtype),
            )
            logpf_pi += pf_pi_dist.log_prob(x).sum(1)

            pf_p_dist = torch.distributions.Normal(
                self.bc_net.oneover_sqrta[i] * (new_x - self.bc_net.mab_over_sqrtmab_inv[i] * epsilon),
                torch.sqrt(self.bc_net.beta_t[i]) * torch.ones_like(new_x, dtype=self.dtype),
            )
            logpf_p += pf_p_dist.log_prob(x).sum(1)

            x = new_x
        prior_dist = torch.distributions.Normal(torch.zeros_like(x, dtype=self.dtype), torch.ones_like(x, dtype=self.dtype))
        logpf_pi += prior_dist.log_prob(x).sum(1)
        logpf_p += prior_dist.log_prob(x).sum(1)
        loss = 0.5 * ((self.logZ + logpf_p * self.alpha - logr.detach() - logpf_pi * self.alpha) ** 2).mean()
        return loss, self.logZ

    def compute_loss(self, device, gfn_batch_size=512):
        x, logpf_pi, logpf_p = self.sample(bs=gfn_batch_size, device=device)
        logr = self.posterior_log_reward(x)
        loss = 0.5 * ((self.logZ + logpf_p * self.alpha - logr.detach() - logpf_pi * self.alpha) ** 2).mean()
        return loss, self.logZ, x, logr