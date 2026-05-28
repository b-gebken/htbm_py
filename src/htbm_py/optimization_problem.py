from dataclasses import dataclass
import numpy as np

@dataclass
class OptimizationProblem:
    oracle: list
    x0: np.ndarray