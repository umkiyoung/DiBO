import os
import sys
import json
import argparse
import pickle
import operator
import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch.quasirandom import SobolEngine
import wandb

import copy as cp
from scipy.stats import norm
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

import gpytorch
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

import matplotlib.pyplot as plt
from matplotlib import cm

from baselines.functions.test_function import TestFunction
from baselines.utils import set_seed, to_unit_cube, from_unit_cube, latin_hypercube

class GP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_constraint, outputscale_constraint, ard_dims):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.ard_dims = ard_dims
        self.mean_module = ConstantMean()
        base_kernel = MaternKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, nu=2.5)
        self.covar_module = ScaleKernel(base_kernel, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def train_gp(train_x, train_y, use_ard, num_steps, hypers={}):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized."""
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    # Create hyper parameter bounds
    noise_constraint = Interval(5e-4, 0.2)
    if use_ard:
        lengthscale_constraint = Interval(0.005, 2.0)
    else:
        lengthscale_constraint = Interval(0.005, math.sqrt(train_x.shape[1]))  # [0.005, sqrt(dim)]
    outputscale_constraint = Interval(0.05, 20.0)

    # Create models
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device, dtype=train_y.dtype)
    ard_dims = train_x.shape[1] if use_ard else None
    model = GP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=ard_dims,
    ).to(device=train_x.device, dtype=train_x.dtype)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Initialize model hypers
    if hypers:
        model.load_state_dict(hypers)
    else:
        hypers = {}
        hypers["covar_module.outputscale"] = 1.0
        hypers["covar_module.base_kernel.lengthscale"] = 0.5
        hypers["likelihood.noise"] = 0.005
        model.initialize(**hypers)

    # Use the adam optimizer
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # Switch to eval mode
    model.eval()
    likelihood.eval()

    return model

class Turbo1:
    """The TuRBO-1 algorithm.

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Example usage:
        turbo1 = Turbo1(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals)
        turbo1.optimize()  # Run optimization
        X, fX = turbo1.X, turbo1.fX  # Evaluated points
    """

    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
        boundary = [],
        X_init   = np.array([])
    ):

        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
        assert max_evals > n_init and max_evals > batch_size
        assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"
        
        if device == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"

        # Save function information
        self.boundary = boundary
        self.X_init   = X_init
        self.f = f
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

        # Settings
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps  = n_training_steps

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))

        # Tolerances and counters
        self.n_cand = min(100 * self.dim, 5000)
        self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
        self.succtol = 3
        self.n_evals = 0

        # Trust region sizes
        self.length_min = 0.5 ** 7
        self.length_max = 1.6
        self.length_init = 0.8

        # Save the full history
        self.X       = np.zeros((0, self.dim))
        self.fX      = np.zeros((0, 1))
        self.X_hist  = np.zeros((0, self.dim))
        self.fX_hist = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()
        print("===>boundary:", self.boundary )
        # Initialize parameters
        self._restart()

    def _restart(self):
        self._X = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init

    def _adjust_length(self, fX_next):
        if np.min(fX_next) < np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        if self.succcount == self.succtol:  # Expand trust region
            self.length = min([2.0 * self.length, self.length_max])
            self.succcount = 0
        elif self.failcount == self.failtol:  # Shrink trust region
            # self.length /= 2.0
            self.length = max([0.5 * self.length, self.length_min])
            self.failcount = 0

    def _create_candidates(self, X, fX, length, n_training_steps, hypers):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0

        # Standardize function values.
        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers
            )

            # Save state dict
            hypers = gp.state_dict()

        # Create the trust region boundaries
        x_center = X[fX.argmin().item(), :][None, :]
        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        lb = np.clip( x_center - weights * length / 2.0, 0.0, 1.0 )
        ub = np.clip( x_center + weights * length / 2.0, 0.0, 1.0 )

        # Draw a Sobolev sequence in [lb, ub] in [0, 1]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert
        

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones((self.n_cand, self.dim))
        X_cand[mask] = pert[mask]

        # Figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        gp = gp.to(dtype=dtype, device=device)

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            y_cand       = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()
        # Remove the torch variables
        del X_torch, y_torch, X_cand_torch, gp

        # De-standardize the sampled values
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, hypers

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates."""
        X_next = np.ones((self.batch_size, self.dim))
        for i in range(self.batch_size):
            # Pick the best point and make sure we never pick it again
            indbest = np.argmin(y_cand[:, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_cand[indbest, :] = np.inf
        return X_next
        
    def get_samples_in_region( self, cands ):
        if len(self.boundary) == 0:
            # no boundary, return all candidates
            return 1.0, cands
        elif len(cands) == 0:
            return 0.0, cands
        else:
            # with boundaries, return filtered cands
            total = len(cands)
            for node in self.boundary:
                boundary = node[0].classifier.svm
                if len(cands) == 0:
                    return 0, np.array([])
                assert len(cands) > 0
                cands = cands[ boundary.predict( cands ) == node[1] ] 
                # node[1] store the direction to go
            ratio = len(cands) / total
            assert len(cands) <= total
            return ratio, cands
    
    def get_init_samples(self):
        
        num_samples   = 5000
        while True:
            X_init        = latin_hypercube( num_samples, self.dim )
            X_init        = from_unit_cube(X_init, self.lb, self.ub)
            ratio, X_init = self.get_samples_in_region(X_init)
            print("sampling for init:", X_init.shape, " target=", self.n_init )
            
             # print("init ratio:", ratio, num_samples)
            if len(X_init) > self.n_init:
                X_init_idx = np.random.choice( len(X_init), self.n_init )
                return X_init[X_init_idx]
            else:
                num_samples *= 2
                
    def solution_dist(self, X_max, X_min):
        target = [0.67647699, 0.02470704, 0.17509452, 0.52625823, 0.01533873, 0.23564648,
                   0.02683509, 0.4015465,  0.06774012, 0.46741845, 0.14822474, 0.28144135,
                   0.37140203, 0.16719317, 0.20886799, 0.78002471, 0.08521446, 0.92605524,
                   0.23940475, 0.2922662,  0.72604942, 0.4934763,  0.54875525, 0.83353381,
                   0.91081349, 0.92451653, 0.67479518, 0.10795649, 0.23629373, 0.93527296,
                   0.79859278, 0.47183663, 0.60424984, 0.82342833, 0.82568537, 0.03397018,
                   0.17525656, 0.44860477, 0.38917436, 0.7433467,  0.38558197, 0.54083661,
                   0.04085656, 0.59639248, 0.9753219,  0.83503397, 0.78734637, 0.74482509,
                   0.74704426, 0.93000639, 0.98498581, 0.8575799,  0.97067501, 0.85890235,
                   0.77135328, 0.58061348, 0.96214013, 0.53402563, 0.59676158, 0.80739623]
        located_in = 0
        for idx in range(0, len(X_max) ):
            if target[idx] < X_max[idx] and target[idx] > X_min[idx]:
                located_in += 1
        
        return located_in
        
                
    def optimize(self):
        """Run the full optimization process."""
        while self.n_evals < self.max_evals:
            if len(self._fX) > 0 and self.verbose:
                n_evals, fbest = self.n_evals, self._fX.min()
                print(f"{n_evals}) Restarting with fbest = {fbest:.4}")
                sys.stdout.flush()

            # Initialize parameters
            self._restart()

            # Generate and evalute initial design points
            X_init = self.X_init #self.get_init_samples()
            # X_init = deepcopy( self.boundary[-1][0].classifier.X )
            #assert ratio == 1
            X_max = np.max(X_init, axis= 0)
            X_min = np.min(X_init, axis= 0)
            # print("--->max:", X_max)
            # print("--->min:", X_min)
            # print("--->dist:", X_max - X_min)
            # print("--->summ:", self.solution_dist(X_max, X_min)," in ", len(X_max) )
            fX_init = np.array([[self.f(torch.from_numpy(x).to(dtype=dtype, device=device)).cpu().detach().item()] for x in X_init])
            self.X_hist = np.vstack((self.X_hist, deepcopy(X_init)))
            self.fX_hist = np.vstack((self.fX_hist, deepcopy(fX_init)))
            
            # Update budget and set as initial data for this TR
            self.n_evals += len(X_init)
            self._X = deepcopy(X_init)
            self._fX = deepcopy(fX_init)

            # Append data to the global history
            self.X = np.vstack((self.X, deepcopy(X_init)))
            self.fX = np.vstack((self.fX, deepcopy(fX_init)))

            if self.verbose:
                fbest = self._fX.min()
                print(f"Starting from fbest = {fbest:.4}")
                sys.stdout.flush()

            # Thompson sample to get next suggestions
            while self.n_evals < self.max_evals and self.length >= self.length_min:
                # Warp inputs
                X = to_unit_cube(deepcopy(self._X), self.lb, self.ub) # project X to [lb, ub] as X was in [0, 1]

                # Standardize values
                fX = deepcopy(self._fX).ravel()

                # Create th next batch
                X_cand, y_cand, _ = self._create_candidates(
                    X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
                )
                X_next = self._select_candidates(X_cand, y_cand)

                # Undo the warping
                X_next = from_unit_cube(X_next, self.lb, self.ub)

                # Evaluate batch
                # fX_next = np.array([[self.f(x)] for x in X_next])
                fX_next = np.array([[self.f(torch.from_numpy(x).to(dtype=dtype, device=device)).cpu().detach().item()] for x in X_next])

                # Update trust region
                self._adjust_length(fX_next)

                # Update budget and append data
                self.n_evals += self.batch_size
                self._X = np.vstack((self._X, X_next))
                self._fX = np.vstack((self._fX, fX_next))

                if self.verbose and fX_next.min() < self.fX.min():
                    n_evals, fbest = self.n_evals, fX_next.min()
                    print(f"{n_evals}) New best: {fbest:.4}")
                    # wandb.log({"n_evals": n_evals, "fbest": -fbest})
                    sys.stdout.flush()

                # Append data to the global history
                self.X = np.vstack((self.X, deepcopy(X_next)))
                self.fX = np.vstack((self.fX, deepcopy(fX_next)))
                self.X_hist = np.vstack((self.X_hist, deepcopy(X_next)))
                self.fX_hist = np.vstack((self.fX_hist, deepcopy(fX_next)))
                
                # print(self.n_evals, self.max_evals, self.length, self.length_min)
                
            
            return self.X_hist, self.fX_hist.ravel()

class Classifier():
    def __init__(self, samples, dims, kernel_type, gamma_type = "auto"):
        self.training_counter = 0
        assert dims >= 1
        assert type(samples)  ==  type([])
        self.dims    =   dims
        
        #create a gaussian process regressor
        noise        =   0.1
        m52          =   ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gpr     =   GaussianProcessRegressor(kernel=m52, alpha=noise**2) #default to CPU
        self.kmean   =   KMeans(n_clusters=2, n_init="auto")
        #learned boundary
        self.svm     =   SVC(kernel = kernel_type, gamma=gamma_type)
        #data structures to store
        self.samples = []
        self.X       = np.array([])
        self.fX      = np.array([])
        
        #good region is labeled as zero
        #bad  region is labeled as one
        self.good_label_mean  = -1
        self.bad_label_mean   = -1
        samples = [sample.cpu().detach().numpy() for sample in samples]
        self.update_samples(samples)
    
    def is_splittable_svm(self):
        plabel = self.learn_clusters()
        self.learn_boundary(plabel)
        svm_label = self.svm.predict( self.X )
        if len( np.unique(svm_label) ) == 1:
            return False
        else:
            return True
        
    def get_max(self):
        return np.max(self.fX)
        
    def plot_samples_and_boundary(self, func, name):
        assert func.dims == 2
        
        plabels   = self.svm.predict( self.X )
        good_counts = len( self.X[np.where( plabels == 0 )] )
        bad_counts  = len( self.X[np.where( plabels == 1 )] )
        good_mean = np.mean( self.fX[ np.where( plabels == 0 ) ] )
        bad_mean  = np.mean( self.fX[ np.where( plabels == 1 ) ] )
        
        if np.isnan(good_mean) == False and np.isnan(bad_mean) == False:
            assert good_mean > bad_mean

        lb = func.lb
        ub = func.ub
        x = np.linspace(lb[0], ub[0], 100)
        y = np.linspace(lb[1], ub[1], 100)
        xv, yv = np.meshgrid(x, y)
        true_y = []
        for row in range(0, xv.shape[0]):
            for col in range(0, xv.shape[1]):
                x = xv[row][col]
                y = yv[row][col]
                true_y.append( func( np.array( [x, y] ) ) )
        true_y = np.array( true_y )
        pred_labels = self.svm.predict( np.c_[xv.ravel(), yv.ravel()] )
        pred_labels = pred_labels.reshape( xv.shape )
        
        fig, ax = plt.subplots()
        ax.contour(xv, yv, true_y.reshape(xv.shape), cmap=cm.coolwarm)
        ax.contourf(xv, yv, pred_labels, alpha=0.4)
        
        ax.scatter(self.X[ np.where(plabels == 0) , 0 ], self.X[ np.where(plabels == 0) , 1 ], marker='x', label="good-"+str(np.round(good_mean, 2))+"-"+str(good_counts) )
        ax.scatter(self.X[ np.where(plabels == 1) , 0 ], self.X[ np.where(plabels == 1) , 1 ], marker='x', label="bad-"+str(np.round(bad_mean, 2))+"-"+str(bad_counts)    )
        ax.legend(loc="best")
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        plt.savefig(name)
        plt.close()
    
    def get_mean(self):
        return np.mean(self.fX)
        
    def update_samples(self, latest_samples):
        assert type(latest_samples) == type([])
        X  = []
        fX  = []
        for sample in latest_samples:
            X.append(  sample[0].cpu().detach().numpy() if isinstance(sample[0], torch.Tensor) else sample[0] )
            fX.append( sample[1].cpu().detach().numpy().item() if isinstance(sample[1], torch.Tensor) else sample[1] )
        
        self.X          = np.asarray(X, dtype=np.float32).reshape(-1, self.dims)
        self.fX         = np.asarray(fX,  dtype=np.float32).reshape(-1)

        self.samples    = latest_samples       
        
    def train_gpr(self, samples):
        X  = []
        fX  = []
        for sample in samples:
            X.append(  sample[0] )
            fX.append( sample[1] )
        X  = np.asarray(X).reshape(-1, self.dims)
        fX = np.asarray(fX).reshape(-1)
        
        # print("training GPR with ", len(X), " data X")        
        self.gpr.fit(X, fX)
    
    ###########################
    # BO sampling with EI
    ###########################
    
        
    def expected_improvement(self, X, xi=0.0001, use_ei = True):
        ''' Computes the EI at points X based on existing samples X_sample and Y_sample using a Gaussian process surrogate model.
        Args: X: Points at which EI shall be computed (m x d). X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
        Returns: Expected improvements at points X. '''
        X_sample = self.X
        Y_sample = self.fX.reshape((-1, 1))
        
        gpr = self.gpr
        
        mu, sigma = gpr.predict(X, return_std=True)
        
        if not use_ei:
            return mu
        else:
            #calculate EI
            mu_sample = gpr.predict(X_sample)
            sigma = sigma.reshape(-1, 1)
            mu_sample_opt = np.max(mu_sample)
            with np.errstate(divide='warn'):
                imp = mu - mu_sample_opt - xi
                imp = imp.reshape((-1, 1))
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
            return ei
            
    def plot_boundary(self, X):
        if X.shape[1] > 2:
            return
        fig, ax = plt.subplots()
        ax.scatter( X[ :, 0 ], X[ :, 1 ] , marker='.')
        ax.scatter(self.X[ : , 0 ], self.X[ : , 1 ], marker='x')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        plt.savefig("boundary.pdf")
        plt.close()
    
    def get_sample_ratio_in_region( self, cands, path ):
        total = len(cands)
        for node in path:
            boundary = node[0].classifier.svm
            if len(cands) == 0:
                return 0, np.array([])
            assert len(cands) > 0
            cands = cands[ boundary.predict( cands ) == node[1] ] 
            # node[1] store the direction to go
        ratio = len(cands) / total
        assert len(cands) <= total
        return ratio, cands

    def propose_rand_samples_probe(self, nums_samples, path, lb, ub):

        seed   = np.random.randint(int(1e6))
        sobol  = SobolEngine(dimension = self.dims, scramble=True, seed=seed)

        center = np.mean(self.X, axis = 0)
        #check if the center located in the region
        ratio, tmp = self.get_sample_ratio_in_region( np.reshape(center, (1, len(center) ) ), path )
        if ratio == 0:
            print("==>center not in the region, using random samples")
            return self.propose_rand_samples(nums_samples, lb, ub)
        # it is possible that the selected region has no points,
        # so we need check here

        axes    = len( center )
        
        final_L = []
        for axis in range(0, axes):
            L       = np.zeros( center.shape )
            L[axis] = 0.01
            ratio   = 1
            
            while ratio >= 0.9:
                L[axis] = L[axis]*2
                if L[axis] >= (ub[axis] - lb[axis]):
                    break
                lb_     = np.clip( center - L/2, lb, ub )
                ub_     = np.clip( center + L/2, lb, ub )
                cands_  = sobol.draw(10000).to(dtype=torch.float64).cpu().detach().numpy()
                cands_  = (ub_ - lb_)*cands_ + lb_
                ratio, tmp = self.get_sample_ratio_in_region(cands_, path )
            final_L.append( L[axis] )

        final_L   = np.array( final_L )
        lb_       = np.clip( center - final_L/2, lb, ub )
        ub_       = np.clip( center + final_L/2, lb, ub )
        # print("center:", center)
        # print("final lb:", lb_)
        # print("final ub:", ub_)
    
        count         = 0
        cands         = np.array([])
        while len(cands) < 10000:
            count    += 10000
            cands     = sobol.draw(count).to(dtype=torch.float64).cpu().detach().numpy()
        
            cands     = (ub_ - lb_)*cands + lb_
            ratio, cands = self.get_sample_ratio_in_region(cands, path)
            samples_count = len( cands )
        
        #extract candidates 
        
        return cands
            
    def propose_rand_samples_sobol(self, nums_samples, path, lb, ub):
        
        #rejected sampling
        selected_cands = np.zeros((1, self.dims))
        seed   = np.random.randint(int(1e6))
        sobol  = SobolEngine(dimension = self.dims, scramble=True, seed=seed)
        
        # scale the samples to the entire search space
        # ----------------------------------- #
        # while len(selected_cands) <= nums_samples:
        #     cands  = sobol.draw(100000).to(dtype=torch.float64).cpu().detach().numpy()
        #     cands  = (ub - lb)*cands + lb
        #     for node in path:
        #         boundary = node[0].classifier.svm
        #         if len(cands) == 0:
        #             return []
        #         cands = cands[ boundary.predict(cands) == node[1] ] # node[1] store the direction to go
        #     selected_cands = np.append( selected_cands, cands, axis= 0)
        #     print("total sampled:", len(selected_cands) )
        # return cands
        # ----------------------------------- #
        #shrink the cands region
        
        ratio_check, centers = self.get_sample_ratio_in_region(self.X, path)
        # no current samples located in the region
        # should not happen
        # print("ratio check:", ratio_check, len(self.X) )
        # assert ratio_check > 0
        if ratio_check == 0 or len(centers) == 0:
            return self.propose_rand_samples( nums_samples, lb, ub )
        
        lb_    = None
        ub_    = None
        
        final_cands = []
        for center in centers:
            center = self.X[ np.random.randint( len(self.X) ) ]
            cands  = sobol.draw(2000).to(dtype=torch.float64).cpu().detach().numpy()
            ratio  = 1
            L      = 0.0001
            Blimit = np.max(ub - lb)
            
            while ratio == 1 and L < Blimit:                    
                lb_    = np.clip( center - L/2, lb, ub )
                ub_    = np.clip( center + L/2, lb, ub )
                cands_ = cp.deepcopy( cands )
                cands_ = (ub_ - lb_)*cands_ + lb_
                ratio, cands_ = self.get_sample_ratio_in_region(cands_, path)
                if ratio < 1:
                    final_cands.extend( cands_.tolist() )
                L = L*2
        final_cands      = np.array( final_cands )
        if len(final_cands) > nums_samples:
            final_cands_idx  = np.random.choice( len(final_cands), nums_samples )
            return final_cands[final_cands_idx]
        else:
            if len(final_cands) == 0:
                return self.propose_rand_samples( nums_samples, lb, ub )
            else:
                return final_cands
        
    def propose_samples_bo( self, nums_samples = 10, path = None, lb = None, ub = None, samples = None):
        ''' Proposes the next sampling point by optimizing the acquisition function. 
        Args: acquisition: Acquisition function. X_sample: Sample locations (n x d). 
        Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples. 
        Returns: Location of the acquisition function maximum. '''
        assert path is not None and len(path) >= 0
        assert lb is not None and ub is not None
        assert samples is not None and len(samples) > 0
        
        self.train_gpr( samples ) # learn in unit cube
        
        dim  = self.dims
        nums_rand_samples = 10000
        if len(path) == 0:
            return self.propose_rand_samples(nums_samples, lb, ub)
        
        X    = self.propose_rand_samples_sobol(nums_rand_samples, path, lb, ub)
        # print("samples in the region:", len(X) )
        # self.plot_boundary(X)
        if len(X) == 0:
            return self.propose_rand_samples(nums_samples, lb, ub)
        
        X_ei = self.expected_improvement(X, xi=0.001, use_ei = True)
        row, col = X.shape
    
        X_ei = X_ei.reshape(len(X))
        n = nums_samples
        if X_ei.shape[0] < n:
            n = X_ei.shape[0]
        indices = np.argsort(X_ei)[-n:]
        proposed_X = X[indices]
        return proposed_X
        
    ###########################
    # sampling with turbo
    ###########################
    # version 1: select a partition, perform one-time turbo search

    def propose_samples_turbo(self, num_samples, path, func):
        #throw a uniform sampling in the selected partition
        X_init = self.propose_rand_samples_sobol(30, path, func.lb_, func.ub_)
        #get samples around the selected partition
        print("sampled ", len(X_init), " for the initialization")
        turbo1 = Turbo1(
            f  = func,              # Handle to objective function
            lb = func.lb_,           # Numpy array specifying lower bounds
            ub = func.ub_,           # Numpy array specifying upper bounds
            n_init = 30,            # Number of initial bounds from an Latin hypercube design
            max_evals  = num_samples, # Maximum number of evaluations
            batch_size = 1,         # How large batch size TuRBO uses
            verbose=True,           # Print information from each batch
            use_ard=True,           # Set to true if you want to use ARD for the GP kernel
            max_cholesky_size=2000, # When we switch from Cholesky to Lanczos
            n_training_steps=50,    # Number of steps of ADAM to learn the hypers
            min_cuda=1024,          #  Run on the CPU for small datasets
            device="cuda", # "cpu" or "cuda"
            dtype="float64",        # float64 or float32
            X_init = X_init,
        )
    
        proposed_X, fX = turbo1.optimize( )
        fX = fX*-1
    
        return proposed_X, fX
    
    ###########################
    # random sampling
    ###########################
    
    def propose_rand_samples(self, nums_samples, lb, ub):
        x = np.random.uniform(lb, ub, size = (nums_samples, self.dims) )
        return x
        
        
    def propose_samples_rand( self, nums_samples = 10):
        return self.propose_rand_samples(nums_samples, self.lb, self.ub)
                
    ###########################
    # learning boundary
    ###########################
    
        
    def get_cluster_mean(self, plabel):
        assert plabel.shape[0] == self.fX.shape[0] 
        
        zero_label_fX = []
        one_label_fX  = []
        
        for idx in range(0, len(plabel)):
            if plabel[idx] == 0:
                zero_label_fX.append( self.fX[idx]  )
            elif plabel[idx] == 1:
                one_label_fX.append( self.fX[idx] )
            else:
                print("kmean should only predict two clusters, Classifiers.py:line73")
                os._exit(1)
                
        good_label_mean = np.mean( np.array(zero_label_fX) )
        bad_label_mean  = np.mean( np.array(one_label_fX) )
        return good_label_mean, bad_label_mean
        
    def learn_boundary(self, plabel):
        assert len(plabel) == len(self.X)
        self.svm.fit(self.X, plabel)
        
    def learn_clusters(self):
        assert len(self.samples) >= 2, "samples must > 0"
        assert self.X.shape[0], "points must > 0"
        assert self.fX.shape[0], "fX must > 0"
        assert self.X.shape[0] == self.fX.shape[0]
        
        tmp = np.concatenate( (self.X, self.fX.reshape([-1, 1]) ), axis = 1 )
        assert tmp.shape[0] == self.fX.shape[0]
        
        self.kmean  = self.kmean.fit(tmp)
        plabel      = self.kmean.predict( tmp )
        
        # the 0-1 labels in kmean can be different from the actual
        # flip the label is not consistent
        # 0: good cluster, 1: bad cluster
        
        self.good_label_mean , self.bad_label_mean = self.get_cluster_mean(plabel)
        
        if self.bad_label_mean > self.good_label_mean:
            for idx in range(0, len(plabel)):
                if plabel[idx] == 0:
                    plabel[idx] = 1
                else:
                    plabel[idx] = 0
                    
        self.good_label_mean , self.bad_label_mean = self.get_cluster_mean(plabel)
        
        return plabel
        
    def split_data(self):
        good_samples = []
        bad_samples  = []
        train_good_samples = []
        train_bad_samples  = []
        if len( self.samples ) == 0:
            return good_samples, bad_samples
        
        plabel = self.learn_clusters( )
        self.learn_boundary( plabel )
        
        for idx in range(0, len(plabel)):
            if plabel[idx] == 0:
                #ensure the consistency
                assert self.samples[idx][-1] - self.fX[idx] <= 1
                good_samples.append( self.samples[idx] )
                train_good_samples.append( self.X[idx] )
            else:
                bad_samples.append( self.samples[idx] )
                train_bad_samples.append( self.X[idx] )
        
        train_good_samples = np.array(train_good_samples)
        train_bad_samples  = np.array(train_bad_samples)
                        
        assert len(good_samples) + len(bad_samples) == len(self.samples)
                
        return  good_samples, bad_samples

class Node:
    obj_counter   = 0
    # If a leave holds >= SPLIT_THRESH, we split into two new nodes.
    
    def __init__(self, parent = None, dims = 0, reset_id = False, kernel_type = "rbf", gamma_type = "auto"):
        # Note: every node is initialized as a leaf,
        # only internal nodes equip with classifiers to make decisions
        # if not is_root:
        #     assert type( parent ) == type( self )
        self.dims          = dims
        self.x_bar         = float('inf')
        self.n             = 0
        self.uct           = 0
        self.classifier    = Classifier( [], self.dims, kernel_type, gamma_type )
            
        #insert curt into the kids of parent
        self.parent        = parent        
        self.kids          = [] # 0:good, 1:bad
        
        self.bag               = []
        self.is_svm_splittable = False 
        
        if reset_id:
            Node.obj_counter = 0

        self.id            = Node.obj_counter
                
        #data for good and bad kids, respectively
        Node.obj_counter += 1
    
    def update_kids(self, good_kid, bad_kid):
        assert len(self.kids) == 0
        self.kids.append( good_kid )
        self.kids.append( bad_kid )
        assert self.kids[0].classifier.get_mean() > self.kids[1].classifier.get_mean()
        
    def is_good_kid(self):
        if self.parent is not None:
            if self.parent.kids[0] == self:
                return True
            else:
                return False
        else:
            return False
    
    def is_leaf(self):
        if len(self.kids) == 0:
            return True
        else:
            return False 
            
    def visit(self):
        self.n += 1
        
    def print_bag(self):
        sorted_bag = sorted(self.bag.items(), key=operator.itemgetter(1))
        print("BAG"+"#"*10)
        for item in sorted_bag:
            print(item[0],"==>", item[1])            
        print("BAG"+"#"*10)
        print('\n')
        
    def update_bag(self, samples):
        assert len(samples) > 0
        
        self.bag.clear()
        self.bag.extend( samples )
        self.classifier.update_samples( self.bag )
        if len(self.bag) <= 2:
            self.is_svm_splittable = False
        else:
            self.is_svm_splittable = self.classifier.is_splittable_svm()
        self.x_bar             = self.classifier.get_mean()
        self.n                 = len( self.bag )
        
    def clear_data(self):
        self.bag.clear()
    
    def get_name(self):
        # state is a list of jsons
        return "node" + str(self.id)
    
    def pad_str_to_8chars(self, ins, total):
        if len(ins) <= total:
            ins += ' '*(total - len(ins) )
            return ins
        else:
            return ins[0:total]
            
    def get_rand_sample_from_bag(self):
        if len( self.bag ) > 0:
            upeer_boundary = len(list(self.bag))
            rand_idx = np.random.randint(0, upeer_boundary)
            return self.bag[rand_idx][0]
        else:
            return None
            
    def get_parent_str(self):
        return self.parent.get_name()
            
    def propose_samples_bo(self, num_samples, path, lb, ub, samples):
        proposed_X = self.classifier.propose_samples_bo(num_samples, path, lb, ub, samples)
        return proposed_X
        
    def propose_samples_turbo(self, num_samples, path, func):
        proposed_X, fX = self.classifier.propose_samples_turbo(num_samples, path, func)
        return proposed_X, fX

    def propose_samples_rand(self, num_samples):
        assert num_samples > 0
        samples = self.classifier.propose_samples_rand(num_samples)
        return samples
    
    def __str__(self):
        name   = self.get_name()
        name   = self.pad_str_to_8chars(name, 7)
        name  += ( self.pad_str_to_8chars( 'is good:' + str(self.is_good_kid() ), 15 ) )
        name  += ( self.pad_str_to_8chars( 'is leaf:' + str(self.is_leaf() ), 15 ) )
        
        val    = 0
        name  += ( self.pad_str_to_8chars( ' val:{0:.4f}   '.format(round(self.get_xbar(), 3) ), 20 ) )
        name  += ( self.pad_str_to_8chars( ' uct:{0:.4f}   '.format(round(self.get_uct(), 3) ), 20 ) )

        name  += self.pad_str_to_8chars( 'sp/n:'+ str(len(self.bag))+"/"+str(self.n), 15 )
        upper_bound = np.around( np.max(self.classifier.X, axis = 0), decimals=2 )
        lower_bound = np.around( np.min(self.classifier.X, axis = 0), decimals=2 )
        boundary    = ''
        for idx in range(0, self.dims):
            boundary += str(lower_bound[idx])+'>'+str(upper_bound[idx])+' '
            
        #name  += ( self.pad_str_to_8chars( 'bound:' + boundary, 60 ) )

        parent = '----'
        if self.parent is not None:
            parent = self.parent.get_name()
        parent = self.pad_str_to_8chars(parent, 10)
        
        name += (' parent:' + parent)
        
        kids = ''
        kid  = ''
        for k in self.kids:
            kid   = self.pad_str_to_8chars( k.get_name(), 10 )
            kids += kid
        name  += (' kids:' + kids)
        
        return name
    

    def get_uct(self, Cp = 10 ):
        if self.parent == None:
            return float('inf')
        if self.n == 0:
            return float('inf')
        return self.x_bar + 2*Cp*math.sqrt( 2* np.power(self.parent.n, 0.5) / self.n )
    
    def get_xbar(self):
        return self.x_bar

    def get_n(self):
        return self.n
        
    def train_and_split(self):
        assert len(self.bag) >= 2
        self.classifier.update_samples( self.bag )
        good_kid_data, bad_kid_data = self.classifier.split_data()
        assert len( good_kid_data ) + len( bad_kid_data ) ==  len( self.bag )
        return good_kid_data, bad_kid_data

    def plot_samples_and_boundary(self, func):
        name = self.get_name() + ".pdf"
        self.classifier.plot_samples_and_boundary(func, name)

    def sample_arch(self):
        if len(self.bag) == 0:
            return None
        net_str = np.random.choice( list(self.bag.keys() ) )
        del self.bag[net_str]
        return json.loads(net_str )
    

class MCTS:
    #############################################

    def __init__(self, lb, ub, dims, ninits, xinits, func, Cp = 1, leaf_size = 20, kernel_type = "rbf", gamma_type = "auto"):
        self.dims                    =  dims
        self.samples                 =  []
        self.nodes                   =  []
        self.Cp                      =  Cp
        self.lb                      =  lb
        self.ub                      =  ub
        self.ninits                  =  ninits
        self.xinits                  =  xinits
        self.func                    =  func
        self.curt_best_value         =  float("-inf")
        self.curt_best_sample        =  None
        self.best_value_trace        =  []
        self.sample_counter          =  0
        self.visualization           =  False
        
        self.LEAF_SAMPLE_SIZE        =  leaf_size
        self.kernel_type             =  kernel_type
        self.gamma_type              =  gamma_type
        
        # self.solver_type             = 'bo' #solver can be 'bo' or 'turbo'
        self.solver_type             = 'turbo' #solver can be 'bo' or 'turbo'
        
        print("gamma_type:", gamma_type)
        
        #we start the most basic form of the tree, 3 nodes and height = 1
        root = Node( parent = None, dims = self.dims, reset_id = True, kernel_type = self.kernel_type, gamma_type = self.gamma_type )
        self.nodes.append( root )
        
        self.ROOT = root
        self.CURT = self.ROOT
        self.init_train()
        
    def populate_training_data(self):
        #only keep root
        self.ROOT.obj_counter = 0
        for node in self.nodes:
            node.clear_data()
        self.nodes.clear()
        new_root  = Node(parent = None,   dims = self.dims, reset_id = True, kernel_type = self.kernel_type, gamma_type = self.gamma_type )
        self.nodes.append( new_root )
        
        self.ROOT = new_root
        self.CURT = self.ROOT
        self.ROOT.update_bag( self.samples )
    
    def get_leaf_status(self):
        status = []
        for node in self.nodes:
            if node.is_leaf() == True and len(node.bag) > self.LEAF_SAMPLE_SIZE and node.is_svm_splittable == True:
                status.append( True  )
            else:
                status.append( False )
        return np.array( status )
        
    def get_split_idx(self):
        split_by_samples = np.argwhere( self.get_leaf_status() == True ).reshape(-1)
        return split_by_samples
    
    def is_splitable(self):
        status = self.get_leaf_status()
        if True in status:
            return True
        else:
            return False
        
    def dynamic_treeify(self):
        # we bifurcate a node once it contains over 20 samples
        # the node will bifurcate into a good and a bad kid
        self.populate_training_data()
        assert len(self.ROOT.bag) == len(self.samples)
        assert len(self.nodes)    == 1
                
        while self.is_splitable():
            to_split = self.get_split_idx()
            #print("==>to split:", to_split, " total:", len(self.nodes) )
            for nidx in to_split:
                parent = self.nodes[nidx] # parent check if the boundary is splittable by svm
                assert len(parent.bag) >= self.LEAF_SAMPLE_SIZE
                assert parent.is_svm_splittable == True
                # print("spliting node:", parent.get_name(), len(parent.bag))
                good_kid_data, bad_kid_data = parent.train_and_split()
                #creat two kids, assign the data, and push into lists
                # children's lb and ub will be decided by its parent
                assert len(good_kid_data) + len(bad_kid_data) == len(parent.bag)
                assert len(good_kid_data) > 0
                assert len(bad_kid_data)  > 0
                good_kid = Node(parent = parent, dims = self.dims, reset_id = False, kernel_type = self.kernel_type, gamma_type = self.gamma_type )
                bad_kid  = Node(parent = parent, dims = self.dims, reset_id = False, kernel_type = self.kernel_type, gamma_type = self.gamma_type )
                good_kid.update_bag( good_kid_data )
                bad_kid.update_bag(  bad_kid_data  )
            
                parent.update_kids( good_kid = good_kid, bad_kid = bad_kid )
            
                self.nodes.append(good_kid)
                self.nodes.append(bad_kid)
                
            #print("continue split:", self.is_splitable())
        
        # self.print_tree()
        
    def collect_samples(self, sample, value = None):
        #TODO: to perform some checks here
        if value == None:
            value = self.func(sample).item()*-1
    
        if value > self.curt_best_value:
            self.curt_best_value  = value
            self.curt_best_sample = sample 
            self.best_value_trace.append( (value, self.sample_counter) )
        self.sample_counter += 1
        self.samples.append( (sample, value) )
        return value
        
    def init_train(self):
        
        # # here we use latin hyper space to generate init samples in the search space
        init_points = latin_hypercube(self.ninits, self.dims)
        init_points = from_unit_cube(init_points, self.lb, self.ub)
        # init_points = self.xinits
        
        for point in init_points:
            self.collect_samples(torch.from_numpy(point).to(dtype=dtype, device=device))
            # self.collect_samples(point)
        
        print("="*10 + 'collect '+ str(len(self.samples) ) +' points for initializing MCTS'+"="*10)
        print("lb:", self.lb)
        print("ub:", self.ub)
        print("Cp:", self.Cp)
        print("inits:", self.ninits)
        print("dims:", self.dims)
        print("="*58)
        
    def print_tree(self):
        print('-'*100)
        for node in self.nodes:
            print(node)
        print('-'*100)

    def reset_to_root(self):
        self.CURT = self.ROOT
    
    def load_agent(self):
        node_path = 'mcts_agent'
        if os.path.isfile(node_path) == True:
            with open(node_path, 'rb') as json_data:
                self = pickle.load(json_data)
                print("=====>loads:", len(self.samples)," samples" )

    def dump_agent(self):
        node_path = 'mcts_agent'
        print("dumping the agent.....")
        with open(node_path,"wb") as outfile:
            pickle.dump(self, outfile)
            
    def dump_samples(self):
        sample_path = 'samples_'+str(self.sample_counter)
        with open(sample_path, "wb") as outfile:
            pickle.dump(self.samples, outfile)
    
    def dump_trace(self):
        trace_path = 'best_values_trace'
        final_results_str = json.dumps(self.best_value_trace)
        with open(trace_path, "a") as f:
            f.write(final_results_str + '\n')

    def greedy_select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        path      = [ ]
        if self.visualization == True:
            curt_node.plot_samples_and_boundary(self.func)
        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_xbar() )
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append( (curt_node, choice) )
            curt_node = curt_node.kids[choice]
            if curt_node.is_leaf() == False and self.visualization == True:
                curt_node.plot_samples_and_boundary(self.func)
            print("=>", curt_node.get_name(), end=' ' )
        print("")
        return curt_node, path

    def select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        path      = [ ]
        
        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_uct(self.Cp) )
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append( (curt_node, choice) )
            curt_node = curt_node.kids[choice]
            print("=>", curt_node.get_name(), end=' ' )
        print("")
        return curt_node, path
    
    def backpropogate(self, leaf, acc):
        curt_node = leaf
        while curt_node is not None:
            assert curt_node.n > 0
            curt_node.x_bar = (curt_node.x_bar*curt_node.n + acc) / (curt_node.n + 1)
            curt_node.n    += 1
            curt_node       = curt_node.parent

    def search(self, iterations):
        # for idx in range(self.sample_counter, iterations+1, batch_size):
        while self.sample_counter < iterations:
            # print("")
            # print("="*10)
            # print("iteration:", idx)
            # print("="*10)
            self.dynamic_treeify()
            leaf, path = self.select()
            for i in range(0, 1):
                if self.solver_type == 'bo':
                    samples = leaf.propose_samples_bo( 1, path, self.lb, self.ub, self.samples )
                elif self.solver_type == 'turbo':
                    samples, values = leaf.propose_samples_turbo(100, path, self.func )
                    # print(len(samples))
                else:
                    raise Exception("solver not implemented")
                for idx in range(0, len(samples)):
                    if self.solver_type == 'bo':
                        value = self.collect_samples( samples[idx])
                    elif self.solver_type == 'turbo':
                        value = self.collect_samples( samples[idx], values[idx] )
                    else:
                        raise Exception("solver not implemented")
                    
                    self.backpropogate( leaf, value )
            # print(self.sample_counter)
            print("total samples:", len(self.samples) )
            print("current best f(x):", self.curt_best_value )
            # print("current best x:", np.around(self.curt_best_sample, decimals=1) )
            # print("current best x:", self.curt_best_sample )
            wandb.log({"best_value": self.curt_best_value, "total_samples": len(self.samples) })

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process inputs')
    parser.add_argument('--task', type=str, help='specify the test function')
    parser.add_argument('--dim', type=int, help='specify the problem dimensions')
    parser.add_argument('--max_evals', type=int, help='specify the iterations to collect in the search')
    parser.add_argument('--n_init', default=200, type=int, help=' number of initial points')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size of each iteration')
    parser.add_argument('--seed', type=int, help='specify the seed for the random number generator')    

    args = parser.parse_args()
    set_seed(args.seed)
    wandb.init(project="LA-MCTS", config=vars(args))
    import os
    if not os.path.exists("baselines/results"):
        os.makedirs("baselines/results")
    if not os.path.exists("baselines/results/lamcts"):
        os.makedirs("baselines/results/lamcts")
    test_function = TestFunction(args.task, args.dim, args.n_init, args.seed, dtype, device, negate=False)
    f = test_function.fun
    f.name = args.task
    dim = args.dim
    
    agent = MCTS(
             lb = f.lb_,              # the lower bound of each problem dimensions
             ub = f.ub_,              # the upper bound of each problem dimensions
             dims = f.dims_,          # the problem dimensions
             ninits = f.ninits,      # the number of random samples used in initializations
             xinits = test_function.X, # the initial samples 
             func = f,               # function object to be optimized
             Cp = f.Cp,              # Cp for MCTS
             leaf_size = f.leaf_size, # tree leaf size
             kernel_type = f.kernel_type, #SVM configruation
             gamma_type = f.gamma_type    #SVM configruation
             )
    agent.search(iterations = args.max_evals)