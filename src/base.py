from sklearn.base import BaseEstimator, TransformerMixin
import abc


class BaseTransformer(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):

    def fit(self, X, y=None):
        return self

    @abc.abstractmethod
    def transform(self, X, y=None):
        return self