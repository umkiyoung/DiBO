import argparse
import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass

import gpytorch
import numpy as np
import torch
import wandb
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.utils.transforms import unnormalize
from gpytorch.constraints import Interval
from gpytorch.kernels import LinearKernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import norm
from torch.quasirandom import SobolEngine

from baselines.functions.test_function import TestFunction
from baselines.utils import set_seed

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

def get_initial_points(dim, n_pts, seed=None):
    if seed is None:
        seed = torch.randint(1000000, (1,)).item()
        sobol = SobolEngine(dimension=dim, scramble=True)
    else:
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def cdf(value, mean=0, std=1):
    return 0.5 * (1 + torch.erf((value - mean) / (std * math.sqrt(2))))

def get_point_in_tr(x, lb, ub):
    assert x.ndim == 1
    x = x.detach().cpu().numpy()
    if np.all(x>lb) and np.all(x<ub):
        return True
    else:
        return False

def mcmc_one_transit(original_x,noise_size,model, lb=0, ub=1):
    x_set = torch.zeros(
        (original_x.shape[0], 2,original_x.shape[1])
        ).to(dtype=dtype, device=device)
    x_set[:,0,:] = original_x
    x_set[:,1,:] = original_x + noise_size * \
        torch.randn(original_x.shape).to(dtype=dtype, device=device)
    # reject points out of trust region
    for i, x in enumerate(x_set):
        if get_point_in_tr(x_set[i,1,:], lb, ub) == False:
            x_set[i,1,:] = x_set[i,0,:]
    with torch.no_grad():
        x_set_dis = model(x_set)
    a = torch.tensor([1.0,-1.0]).to(dtype=dtype, device=device)
    new_cov = torch.matmul(x_set_dis.lazy_covariance_matrix.matmul(a),a.t())
    new_cov[new_cov<0] = 0
    mean = x_set_dis.mean[:,0]-x_set_dis.mean[:,1]

    temp = (torch.zeros(mean.shape).cuda()-mean)/(torch.sqrt(new_cov+1e-6))
    m = torch.distributions.normal.Normal(
        torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda()
        )
    p = m.cdf(temp)  # normally distributed with loc=0 and scale=1
    p_ratio = (p)/(1-p+1e-7)
    alpha = torch.clamp(p_ratio,0,1)
    sample = torch.rand(alpha.shape).to(dtype=dtype, device=device)
    idx = torch.zeros(alpha.shape).to(dtype=dtype, device=device)
    idx[sample<alpha] = 1
    new_x = x_set[idx==1,1]
    old_x = x_set[idx==0,0]
    new_pop = torch.cat((old_x,new_x),dim=0)

    return new_pop

def langevin_update(x_cur, langevin_epsilon,model,lb=0,ub=1, h = 5e-5, n_splits = 4):
    beta = 2
    n, d = x_cur.shape[0], x_cur.shape[1]
    if n > n_splits:
        batch_size = n // n_splits
    else:
        batch_size = n
    gradients_all = torch.zeros_like(x_cur)
    for idx in range(0,n,batch_size):
        aug_x = x_cur[idx:idx+batch_size,:].unsqueeze(1).unsqueeze(1).repeat(1,d,2,1)
        for dim in range(d):
            aug_x[:,dim,1,dim] += h
        
        posterior = model.posterior(aug_x)
        f_mean, f_covar = posterior.mvn.mean, posterior.mvn.covariance_matrix
        Sigma = f_covar.detach()
        Sigma[:, :, 0, 0] += 1e-4
        Sigma[:, :, 1, 1] += 1e-4
        Sigma_nd = Sigma[:,:,0,0]+Sigma[:,:,1,1]-Sigma[:,:,1,0]-Sigma[:,:,0,1]
        mu_nd = f_mean[:,:,0] - f_mean[:,:,1] + beta * (Sigma[:,:,0,0]-Sigma[:,:,1,1])
        x_grad = mu_nd / torch.sqrt(4 * Sigma_nd)
        x_grad = cdf(-1 * x_grad)
        try:
            x_grad = ((x_grad / (1 - x_grad)) - 1) / h  # (n, d)b``
        except ZeroDivisionError:
            print(f"ZeroDivisionError: {x_grad}")
        gradients_all[idx:idx+batch_size,:] = x_grad
    noise = torch.randn_like(x_cur, device=device) * torch.sqrt(torch.Tensor([2 * langevin_epsilon]).to(device=device))
    x_cur = (
                x_cur + langevin_epsilon * gradients_all + noise
            )  # (n, d)
    if torch.isnan(gradients_all).any():
        raise AssertionError('Gradient nan')
    x_next = torch.clamp(x_cur,torch.tensor(lb).cuda(),torch.tensor(ub).cuda())
    return x_next

def generate_batch_multiple_tr(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    acqf="ts",  # "ei" or "ts"
    use_mcmc=False,
    mcmc_round=200,
    # use_langevin=False
):
    assert acqf in ("ts", "ei")
    tr_num = len(state)
    for tr_idx in range(tr_num):
        assert X[tr_idx].min() >= 0.0 and X[tr_idx].max() <= 1.0 \
            and torch.all(torch.isfinite(Y[tr_idx]))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))
    dim = X[0].shape[1]
    # Scale the TR to be proportional to the lengthscales
    X_cand = torch.zeros(
        tr_num, n_candidates, dim
        ).to(device=device, dtype=dtype)
    Y_cand = torch.zeros(
        tr_num, n_candidates, batch_size
        ).to(device=device, dtype=dtype)
    tr_lb = torch.zeros(tr_num, dim).to(device=device, dtype=dtype)
    tr_ub = torch.zeros(tr_num, dim).to(device=device, dtype=dtype)
    for tr_idx in range(tr_num):
        x_center = X[tr_idx][Y[tr_idx].argmax(), :].clone()
        try:
            weights = model[tr_idx].covar_module.base_kernel.lengthscale.squeeze().detach()
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
            tr_lb[tr_idx] = torch.clamp(
                x_center - weights * state[tr_idx].length / 2.0, 0.0, 1.0
                )
            tr_ub[tr_idx] = torch.clamp(
                x_center + weights * state[tr_idx].length / 2.0, 0.0, 1.0
                )
        except: # Linear kernel
            weights = 1
            tr_lb[tr_idx] = torch.clamp(
                x_center - state[tr_idx].length / 2.0, 0.0, 1.0
                )
            tr_ub[tr_idx] = torch.clamp(
                x_center + state[tr_idx].length / 2.0, 0.0, 1.0
                )

        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb[tr_idx] + (tr_ub[tr_idx] - tr_lb[tr_idx]) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        # prob_perturb = 1
        mask = (
            torch.rand(n_candidates, dim, dtype=dtype, device=device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask        
        X_cand[tr_idx] = x_center.expand(n_candidates, dim).clone()
        X_cand[tr_idx][mask] = pert[mask]

        # Sample on the candidate points
        with torch.no_grad():
            posterior = model[tr_idx].posterior(X_cand[tr_idx])
            samples = posterior.rsample(sample_shape=torch.Size([batch_size]))
        samples = samples.reshape([batch_size, n_candidates])
        Y_cand[tr_idx] = samples.permute(1,0)
        # recover from normalized value
        Y_cand[tr_idx] = Y[tr_idx].mean() + Y_cand[tr_idx] * Y[tr_idx].std()
        
    # Compare across trust region
    y_cand = Y_cand.detach().cpu().numpy()
    X_next = torch.zeros(batch_size, dim).to(device=device, dtype=dtype)
    tr_idx_next = np.zeros(batch_size)
    for k in range(batch_size):
        i, j = np.unravel_index(np.argmax(y_cand[:, :, k]), (tr_num, n_candidates))
        X_next[k] = X_cand[i, j]
        tr_idx_next[k] = i
        assert np.isfinite(y_cand[i, j, k])  # Just to make sure we never select nan or inf
        # Make sure we never pick this point again
        y_cand[i, j, :] = -np.inf

    if use_mcmc == 'MH':
        print('use MH')

        for tr_idx in range(tr_num):
            noise_size = state[tr_idx].length * weights/2
            idx_in_tr = np.argwhere(tr_idx_next==tr_idx).reshape(-1)
            if idx_in_tr.shape[0] == 0:
                continue
            with torch.no_grad():  # We don't need gradients when using TS
                mcmc_round = max(200, dim)
                for i in range(mcmc_round):
                    if i % 5 == 0:
                        print(f'{i}/{mcmc_round}')
                    new_pop = mcmc_one_transit(
                        X_next[idx_in_tr],
                        0.004*noise_size,
                        model[tr_idx], 
                        tr_lb[tr_idx].detach().cpu().numpy(), 
                        tr_ub[tr_idx].detach().cpu().numpy()
                        )
                    X_next[idx_in_tr] = new_pop
    elif use_mcmc == 'Langevin':
        print('use Langevin')
        for tr_idx in range(tr_num):
            noise_size = state[tr_idx].length
            idx_in_tr = np.argwhere(tr_idx_next==tr_idx).reshape(-1)
            if idx_in_tr.shape[0] == 0:
                continue
            with torch.no_grad():  # We don't need gradients when using TS
                round = max(200, mcmc_round)
                for i in range(round):
                    new_pop= langevin_update(
                        X_next[idx_in_tr],
                        2e-3*noise_size,
                        model[tr_idx], 
                        tr_lb[tr_idx].detach().cpu().numpy(), 
                        tr_ub[tr_idx].detach().cpu().numpy()
                        )
                    X_next[idx_in_tr] = new_pop

    return X_next, tr_idx_next

def run(args):
    test_function = TestFunction(args.task, args.dim, args.n_init, args.seed, dtype, device)
    func = test_function.fun
    func.name = args.task
    dim = args.dim
    use_mcmc = False if args.use_mcmc == '0' else args.use_mcmc
    
    wandb.init(project='MCMC_bo',
            config=vars(args),
    )

    print('dimension', dim)

    bounds = func.bounds
    bounds = bounds.to(dtype=dtype, device=device)
    #max_cholesky_size = float("inf")  # Always use Cholesky
    max_cholesky_size = 2000
    g_noise_var = args.noise_var
    def eval_objective(x, noise_var=g_noise_var):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return func(unnormalize(x, bounds)).unsqueeze(-1) + noise_var * torch.randn(1, 1).to(dtype=dtype, device=device)

    all_Y = []
    cur_time = int(time.time())
    for repeat in range(args.repeat_num):
        all_Y.append([])

        # Create turbo state
        state = []
        X_turbo = []
        Y_turbo = []
        for tr_idx in range(args.tr_num):
            X_turbo.append(get_initial_points(dim, args.n_init, args.seed))
            Y_turbo.append(torch.tensor(
                [eval_objective(x) for x in X_turbo[tr_idx]], 
                dtype=dtype, device=device
            ).unsqueeze(-1))
            Y_true = deepcopy(Y_turbo[tr_idx])

            Y_list = list(Y_true.detach().cpu().numpy().flatten())
            all_Y[repeat].extend([round(y, 4) for y in Y_list])
            print(f"Init: Total evaluation: {len(all_Y[repeat])},\
                 current Best:{np.max(all_Y[repeat]):.2e}")
            state.append(TurboState(
                    dim, batch_size=args.batch_size, 
                    best_value=max(Y_turbo[tr_idx]).item()
                    ))

        N_CANDIDATES = min(5000, max(2000, 200 * dim)) 

        while len(all_Y[repeat])<args.max_evals:
            start_time = time.time()
            # fit GP model
            model = []
            mll = []
            train_Y = []
            for tr_idx in range(args.tr_num):               
                train_Y.append((Y_turbo[tr_idx] - Y_turbo[tr_idx].mean()) / Y_turbo[tr_idx].std())
                likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                # if args.task in ['Walker2d','Ant', 'Humanoid', 'HumanoidStandup', 
                # 'InvertedDoublePendulum', 'InvertedPendulum', 'Reacher']:
                #     covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                #                         LinearKernel()
                #                     )
                # else:
                covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                    MaternKernel(nu=2.5, ard_num_dims=dim, 
                    lengthscale_constraint=Interval(0.005, 4.0))
                )
                
                    
                model.append(SingleTaskGP(X_turbo[tr_idx], train_Y[tr_idx], 
                covar_module=covar_module, likelihood=likelihood))
                mll.append(ExactMarginalLogLikelihood(model[tr_idx].likelihood, model[tr_idx]))

                # Do the fitting and acquisition function optimization inside the Cholesky context
                with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                    # Fit the model
                    fit_gpytorch_model(mll[tr_idx])
            
            # Get next selection
            X_next, tr_idx_next = generate_batch_multiple_tr(
                    state=state,
                    model=model,
                    X=X_turbo,
                    Y=Y_turbo, # add Y_turbo to recover normalized GP sampling
                    batch_size=args.batch_size,
                    n_candidates=N_CANDIDATES,
                    acqf="ts",
                    use_mcmc=use_mcmc,
                    mcmc_round=dim,
                    # use_langevin=use_langevin
                )
            
            Y_next = torch.tensor(
                [eval_objective(x) for x in X_next], dtype=dtype, device=device
            ).unsqueeze(-1)
            Y_true = deepcopy(Y_next)
            # Update state
            for tr_idx in range(args.tr_num):
                idx_in_tr = np.argwhere(tr_idx_next == tr_idx).reshape(-1)
                if idx_in_tr.shape[0] > 0:
                    state[tr_idx] = update_state(
                        state=state[tr_idx], Y_next=Y_next[idx_in_tr])
                    # Append data
                    X_turbo[tr_idx] = torch.cat((
                        X_turbo[tr_idx], X_next[idx_in_tr]
                        ), dim=0)
                    Y_turbo[tr_idx] = torch.cat((
                        Y_turbo[tr_idx], Y_next[idx_in_tr]
                        ), dim=0)
            Y_list = list(Y_true.detach().cpu().numpy().flatten())
            all_Y[repeat].extend([round(y, 4) for y in Y_list])
            print(f"{args.task} Evaluations: {[x.shape[0] for x in X_turbo]})")
            print(f"Best value: {[round(x.best_value, 4) for x in state]}")
            print(f"TR length: {[x.length for x in state]}")
            print(f"Total evaluation: {len(all_Y[repeat])}, \
                current Best:{np.max(all_Y[repeat]):.2e}")
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.3f} seconds")
            wandb.log({'best_value': np.max(all_Y[repeat])
                       ,'total_eval': len(all_Y[repeat])})
            
            del model[:]
            del mll[:]
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # Check restart state
            for tr_idx in range(args.tr_num):
                if state[tr_idx].restart_triggered:
                    X_turbo[tr_idx] = get_initial_points(dim, args.n_init)
                    Y_turbo[tr_idx] = torch.tensor(
                        [eval_objective(x) for x in X_turbo[tr_idx]], 
                        dtype=dtype, device=device
                    ).unsqueeze(-1)

                    Y_true = torch.tensor(
                        [eval_objective(x, noise_var=0.0) for x in X_turbo[tr_idx]], 
                        dtype=dtype, device=device
                    ).unsqueeze(-1)
                    Y_list = list(deepcopy(Y_true).detach().cpu().numpy().flatten())
                    all_Y[repeat].extend([round(y, 4) for y in Y_list])
                    state[tr_idx] = TurboState(dim, batch_size=args.batch_size)
                    print('Trust region ', tr_idx, 'restart')
                    
        # Save data
            total_eval = len(all_Y[repeat])
            if total_eval % 1000 == 0 or total_eval >= args.max_evals:
                import os
                if not os.path.exists('./baselines/results'):
                    os.makedirs('./baselines/results')
                if not os.path.exists('./baselines/results/turbo'):
                    os.makedirs('./baselines/results/turbo')
                if not os.path.exists('./baselines/results/mcmcbo'):
                    os.makedirs('./baselines/results/mcmcbo')
                if use_mcmc == False:
                    np.save(
                        f"./baselines/results/turbo/turbo_{args.task}_{dim}_{args.seed}_{args.n_init}_{args.max_evals}_{total_eval}.npy",
                        np.array(all_Y[repeat])[:args.max_evals],
                    )
                elif use_mcmc == 'MH':
                    np.save(
                        f"./baselines/results/mcmcbo/mcmcbo_{args.task}_{dim}_{args.seed}_{args.n_init}_{args.max_evals}_{total_eval}.npy",
                        np.array(all_Y[repeat])[:args.max_evals],
                    )
                else:
                    raise

if __name__ == '__main__':
    #NOTE: Turbo/MCMCBO/Langevin-BO Implemented here.
    #NOTE: Use below environment with gym >= 0.22.0
    # """
    # numpy==1.21.6
    # torch==1.11.0
    # botorch==0.6.5
    # gpytorch==1.7.0
    # """
    parser = argparse.ArgumentParser(description='BO experiment')
    parser.add_argument('-f', '--task', default='Ackley', type=str, help='function to optimize')
    parser.add_argument('--dim', default=200, type=int, help='dimension of the function')
    parser.add_argument('--tr_num', default=1, type=int, help='trust region number')
    parser.add_argument('--max_evals', default=6000, type=int, help='evaluation number')
    parser.add_argument('--n_init', default=200, type=int, help=' number of initial points')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size of each iteration')
    parser.add_argument('--noise_var', default=0, type=int, help='evaluation noise variance')
    parser.add_argument('--repeat_num', default=1, type=int, help='number of repetition')
    parser.add_argument('--use_mcmc', default='Langevin', type=str, help='if use mcmc, 0 not use, ["MH","Langevin" ]')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    args = parser.parse_args()
    set_seed(args.seed)
    run(args)