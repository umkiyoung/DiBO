from typing import List, Optional, Union
import numpy as np
import torch
from baselines.functions.LassoBench import RealBenchmark, SyntheticBenchmark
class LassoRealWorldBenchmark:
    '''
    The base class for Lasso realworld benchmark. Use the derived classes
    '''
    def __init__(self, pick_data: str, seed: Optional[int] = None, dtype=torch.float64, device = 'cpu', **kwargs):
        """
        Constructs all the necessary attributes for real-world bench.

        Parameters
        ----------
            pick_data : str
                name of dataset such as
                Diabetes, Breast_cancer, DNA, Leukemia, RCV1
            seed: int, optional
                seed number
        """
        self._b = RealBenchmark(
            pick_data=pick_data, mf_opt="discrete_fidelity", seed=seed
        )
        self.input_dim = self._b.n_features
        self.effective_dim = None # to be filled in derived classes
        #self.bounds = [(-1.0, 1.0)]*self.input_dim
        self.bounds = torch.zeros((2, self.input_dim), dtype=dtype, device=device)
        self.name = f'lasso-{pick_data}'
        self.noise_std = 0
        self.dtype = dtype
        self.device = device

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        x = np.array(x.cpu().detach(), dtype=np.double)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        result_list = []
        for y in x:
            result = self._b.evaluate(y)
            result_list.append(result)
        result = np.array(result_list).squeeze()
        return torch.tensor(result + np.random.normal(
            np.zeros_like(result), np.ones_like(result) * self.noise_std, result.shape
        ), dtype=self.dtype, device=self.device) * -1
    
    def func(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        return self.__call__(x)
    
class LassoSyntheticBenchmark:
    '''
    The base class for Lasso realworld benchmark. Use the derived classes
    '''

    def __init__(self, pick_bench: str, seed: Optional[int] = None, **kwargs):
        """
        Constructs all the necessary attributes for real-world bench.

        Parameters
        ----------
            pick_bench : str
                name of a predefined bench such as: synt_simple, synt_medium, synt_high, synt_hard
            seed: int, optional
                seed number
        """
        
        self._b = SyntheticBenchmark(
            pick_bench=pick_bench, seed=seed
        )
        self.input_dim = self._b.n_features

        self.effective_dims = np.arange(self.input_dim)[self._b.w_true != 0]
        print(f"function effective dimensions: {self.effective_dims.tolist()}")

        self.bounds = [(-1.0, 1.0)]*self.input_dim
        problem = pick_bench.replace('synt_','')
        self.name = f'lasso-{problem}'

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        x = np.array(x, dtype=np.double)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        result_list = []
        for y in x:
            result = self._b.evaluate(y)
            result_list.append(result)
        return np.array(result_list).squeeze()

    def func(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        return self.__call__(x)
    


class LassoDiabetesBenchmark(LassoRealWorldBenchmark):
    """
   8-D diabetes benchmark from https://github.com/ksehic/LassoBench

   Args:
       seed: seed number
       **kwargs:
   """
    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__(pick_data="diabetes", seed=seed)
        self.effective_dim = 5

class LassoRCV1Benchmark(LassoRealWorldBenchmark):
    """
    19 959-D RCV1 benchmark from https://github.com/ksehic/LassoBench

   Args:
       seed: seed number
       **kwargs:
   """
    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__(pick_data="rcv1", seed=seed)
        self.effective_dim = 75

class LassoDNABenchmark(LassoRealWorldBenchmark):
    """
    180-D DNA RCV1 benchmark from https://github.com/ksehic/LassoBench

   Args:
       seed: seed number
       **kwargs:
   """
    def __init__(self, seed: Optional[int] = None, dtype=torch.float64, device='cpu', **kwargs):
        super().__init__(pick_data="dna", seed=seed, dtype=dtype, device=device)
        self.effective_dim = 43



class LassoSimpleBenchmark(LassoSyntheticBenchmark):
    """
    60-D synthetic Lasso simple benchmark from https://github.com/ksehic/LassoBench .
    Effective dimensionality: 5% of input dimensionality.

    Args:
        seed: optional int | None
        **kwargs:
    """

    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__(pick_bench="synt_simple", seed=seed)

class LassoMediumBenchmark(LassoSyntheticBenchmark):
    """
    100-D synthetic Lasso medium benchmark from https://github.com/ksehic/LassoBench .
    Effective dimensionality: 5% of input dimensionality.

    Args:
        seed: optional int | None
        **kwargs:
    """

    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__(pick_bench="synt_medium", seed=seed)


class LassoHighBenchmark(LassoSyntheticBenchmark):
    """
    300-D synthetic Lasso high benchmark from https://github.com/ksehic/LassoBench .
    Effective dimensionality: 5% of input dimensionality.

    Args:
        seed: optional int | None
        **kwargs:
    """

    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__(pick_bench="synt_high", seed=seed)


class LassoHardBenchmark(LassoSyntheticBenchmark):
    """
    1000-D synthetic Lasso hard benchmark from https://github.com/ksehic/LassoBench .
    Effective dimensionality: 5% of input dimensionality.

    Args:
        seed: optional int | None
        **kwargs:
    """

    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__(pick_bench="synt_hard", seed=seed)