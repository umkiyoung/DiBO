import os
import math
import warnings
from dataclasses import dataclass
import argparse

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize, normalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from baselines.functions.test_function import TestFunction
from baselines.utils import set_seed

import numpy as np
import time
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BO experiment')
    parser.add_argument('-f', '--task', default='Ackley', type=str, help='function to optimize')
    parser.add_argument('--dim', default=200, type=int, help='dimension of the function')
    parser.add_argument('--max_evals', default=6000, type=int, help='evaluation number')
    parser.add_argument('--n_init', default=200, type=int, help=' number of initial points')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size of each iteration')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    args = parser.parse_args()
    import os
    if not os.path.exists("./baselines/results"):
        os.makedirs("./baselines/results")
    if not os.path.exists("./baselines/results/turbo/"):
        os.makedirs("./baselines/results/turbo/")
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    SMOKE_TEST = os.environ.get("SMOKE_TEST")

    test_function = TestFunction(task=args.task, dim=args.dim, n_init=args.n_init, seed=args.seed, dtype=dtype, device=device)
    fun = test_function.fun
    dim = args.dim
    lb, ub = fun.bounds
    batch_size = args.batch_size
    n_init = args.n_init
    max_cholesky_size = 2000
    
    set_seed(args.seed)
    
    def generate_batch(
        state,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        batch_size,
        n_candidates=None,  # Number of candidates for Thompson sampling
        num_restarts=10,
        raw_samples=512,
        acqf="ts",  # "ei" or "ts"
    ):
        assert acqf in ("ts", "ei")
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

        if acqf == "ts":
            dim = X.shape[-1]
            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

            # Create candidate points from the perturbations and the mask
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            # Sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():  # We don't need gradients when using TS
                X_next = thompson_sampling(X_cand, num_samples=batch_size)

        elif acqf == "ei":
            ei = qExpectedImprovement(model, train_Y.max())
            X_next, acq_value = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )

        return X_next
    
    @dataclass
    class TurboState:
        dim: int
        batch_size: int
        length: float = 0.8
        length_min: float = 0.5**7
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
    X_turbo = test_function.X.clone()
    Y_turbo = test_function.Y.clone()
    state = TurboState(dim=dim, batch_size=batch_size, best_value = max(Y_turbo).item())
    
    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 4
    N_CANDIDATES = min(5000, max(2000, 200 * dim)) if not SMOKE_TEST else 4

    model = None
    
    while True:
        start_time = time.time()
        if len(X_turbo) >= args.max_evals:
            break
        train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(
                nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
            )
        )
        model = SingleTaskGP(
            X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            # Fit the model
            fit_gpytorch_model(mll)

            # Create a batch
            X_next = generate_batch(
                state=state,
                model=model,
                X=X_turbo,
                Y=train_Y,
                batch_size=batch_size,
                n_candidates=N_CANDIDATES,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                acqf="ts",
            )
        Y_next = torch.tensor([test_function.eval_objective(x) for x in X_next], dtype=dtype, device=device).unsqueeze(-1)  
        state = update_state(state, Y_next)
        X_turbo = torch.cat([X_turbo, X_next], dim=0)
        Y_turbo = torch.cat([Y_turbo, Y_next], dim=0)
        
        print(
            f"{args.task}_{args.dim}_{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}"
        )
        stop_time = time.time()
        print(f"Time taken: {stop_time - start_time:.3f} seconds")


        if len(X_turbo) % 1000 == 0:
            np.save(
                f"./baselines/results/turbo/turbo_{args.task}_{dim}_{args.seed}_{args.n_init}_{args.max_evals}_{len(X_turbo)}.npy",
                np.array(np.array(Y_turbo.cpu())),
            )