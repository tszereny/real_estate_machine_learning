from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import abc


class BaseTransformer(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):

    def fit(self, X, y=None):
        return self

    @abc.abstractmethod
    def transform(self, X):
        return self


class SlicedPipeline(Pipeline):

    def __init__(self, steps: List[tuple], start_step: str = None, stop_step: str = None, memory=None):
        self.start_step = start_step
        self.stop_step = stop_step
        super().__init__(steps=steps, memory=memory)

    @property
    def steps(self):
        return self._steps

    @steps.setter
    def steps(self, steps):
        step_names = [name for name, transformer in steps]
        start_step_idx = None if self.start_step is None else step_names.index(self.start_step)
        stop_step_idx = None if self.stop_step is None else step_names.index(self.stop_step) + 1
        self._steps = steps[start_step_idx:stop_step_idx]
