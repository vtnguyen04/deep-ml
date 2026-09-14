import numpy as np
from collections import Counter
from abc import ABC, abstractmethod 

class BaseStrategy(ABC):
    @abstractmethod
    def predict(self, y_train, n_test):
        pass

class MostFrequentStrategy(BaseStrategy):
    def predict(self, y_train, n_test):
        counters = Counter(y_train)

        best_class = min(counters.keys(), key=lambda c: (-counters[c], c))

        return [best_class] * n_test

class ConstantStrategy(BaseStrategy):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant
    def predict(self, y_train, n_test):
        return [self.constant] * n_test

class UniformStrategy(BaseStrategy):
    def predict(self, y_train, n_test):
        classes = sorted(set(y_train))
        num_class = len(classes)
        return [classes[i % num_class] for i in range(n_test)]
    
class StratifiedStrategy(BaseStrategy):
    def predict(self, y_train, n_test):
        counts = Counter(y_train)
        classes = sorted(counts.keys())
        n_train = len(y_train)

        allocations = {}
        fracs = []
        total = 0
        for cls, count in counts.items():
            exact = n_test * (count / n_train)
            base = int(exact)
            allocations[cls] = base
            fracs.append((exact - base, cls))
            total += base

        remainder = n_test - total  
        fracs.sort(key=lambda x: (-x[0], x[1]))
        for i in range(remainder):
            allocations[fracs[i][1]] += 1

        result = []
        for c in classes:
            result.extend([c] * allocations[c])
        return result

class DummyClassifier:
    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy

    def predict(self, y_train, n_test):
        return self.strategy.predict(y_train, n_test)  

def dummy_classifier(y_train, n_test, strategy, constant=None):
    """
    Produce baseline predictions of length n_test using the given strategy.
    Returns a Python list of predicted labels.
    """

    strategy_registry = {
        'most_frequent': lambda: MostFrequentStrategy(),
        'constant': lambda: ConstantStrategy(constant),
        'uniform': lambda: UniformStrategy(),
        'stratified': lambda: StratifiedStrategy()
    }

    selected_trategy = strategy_registry.get(strategy)()

    classifier = DummyClassifier(selected_trategy)

    return classifier.predict(y_train, n_test)    
