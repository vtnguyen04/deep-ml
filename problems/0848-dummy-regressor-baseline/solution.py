import numpy as np

from abc import ABC, abstractmethod
import numpy as np

class RegressionStrategy(ABC):
    @abstractmethod
    def compute(self, y_train) -> float:
        pass

class MeanStrategy(RegressionStrategy):
    def compute(self, y_train) -> float:
        return float(np.mean(y_train))

class MedianStrategy(RegressionStrategy):
    def compute(self, y_train) -> float:
        return float(np.median(y_train))

class QuantileStrategy(RegressionStrategy):
    def __init__(self, q: float):
        if q is None or not (0.0 <= q <= 1.0):
            raise ValueError(f"quantile must be in [0, 1], got: {q}")
        self.q = q

    def compute(self, y_train) -> float:
        return float(np.quantile(y_train, self.q))

class ConstantStrategy(RegressionStrategy):
    def __init__(self, constant):
        if constant is None:
            raise ValueError("constant cannot be None")
        self.constant = float(constant)

    def compute(self, y_train) -> float:
        return self.constant

class DummyRegressor:
    def __init__(self, strategy: RegressionStrategy):
        self.strategy = strategy

    def predict(self, y_train, n_test: int):
        p = self.strategy.compute(y_train)
        return [p] * n_test

def dummy_regressor(y_train, n_test, strategy='mean', constant=None, quantile=None):
    """
    Baseline regressor that predicts a constant value derived from y_train.

    Args:
        y_train: 1D array-like of training target values.
        n_test: number of test predictions to return (int >= 0).
        strategy: one of 'mean', 'median', 'quantile', 'constant'.
        constant: required when strategy='constant'.
        quantile: required when strategy='quantile', must be in [0, 1].

    Returns:
        List[float] of length n_test, all equal to the chosen summary value.
    """
    strategy_register = {
        'mean': lambda: MeanStrategy(),
        'median': lambda: MedianStrategy(),
        'quantile': lambda: QuantileStrategy(quantile),
        'constant': lambda: ConstantStrategy(constant),
    }

    selected_strategy = strategy_register.get(strategy)()
    regressor = DummyRegressor(strategy=selected_strategy)
    return regressor.predict(y_train, n_test)
