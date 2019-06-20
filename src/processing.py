from typing import List, Callable, Union
import re
import pandas as pd, numpy as np
import string
from zlib import crc32
from src.base import BaseTransformer
from src.preparation import Elevation
import src.preparation as prep
from src.utils import load_elevation_data


class ColumnRenamer(BaseTransformer):

    def __init__(self, old_to_new=None, hun_to_eng=None):
        self.old_to_new = old_to_new
        self.hun_to_eng = hun_to_eng

    def to_new(self, df):
        df.columns = df.columns.str.lower()
        return df.rename(columns=self.old_to_new)

    def to_eng(self, df):
        return df.rename(columns=self.hun_to_eng)

    def transform(self, X):
        X = X.copy()
        X = self.to_new(X)
        X = self.to_eng(X)
        return X


class Translator(BaseTransformer):

    def __init__(self, column_name, hun_eng_map):
        self.column_name = column_name
        self.hun_eng_map = hun_eng_map

    def transform(self, X):
        X[self.column_name] = X[self.column_name].map(self.hun_eng_map)
        return X


class StringStandardizer(BaseTransformer):

    def __init__(self, func: Callable, column_names: List[str] = None):
        self.column_names = column_names
        self.func = func

    def get_string_columns(self, X):
        if self.column_names is None:
            return [c for c, t in X.dtypes.items() if t == 'object']
        return self.column_names

    def transform(self, X):
        X = X.copy()
        self.column_names = self.get_string_columns(X)
        return X[self.column_names].apply(self.func)


class DuplicatesRemoval(BaseTransformer):

    def __init__(self, columns, negation=False):
        self.columns = columns
        self.negation = negation

    def transform(self, X):
        columns = X.columns[~X.columns.isin(self.columns)].tolist() if self.negation else self.columns
        return X.drop_duplicates(subset=columns)


class ElevationMerger(BaseTransformer):

    def __init__(self, left_longitude: str, left_latitude: str, elevation_data_path: str, elevation_longitude: str, elevation_latitude: str, mode: str = 'w'):
        self.left_longitude = left_longitude
        self.left_latitude = left_latitude
        self.elevation_data_path = elevation_data_path
        self.elevation_longitude = elevation_longitude
        self.elevation_latitude = elevation_latitude
        self.mode = mode

    @property
    def elevation_data(self):
        return load_elevation_data(self.elevation_data_path)

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        merged = X.merge(how='left', right=self.elevation_data, left_on=[self.left_longitude, self.left_latitude],
                         right_on=[self.elevation_longitude, self.elevation_latitude])
        not_in_elevation_mask = merged[[self.elevation_longitude, self.elevation_latitude]].isnull().any(axis=1)
        not_in_elevation_data = merged[not_in_elevation_mask]
        elevation = Elevation(df=not_in_elevation_data, batch_size=100, latitude=self.left_latitude,
                              longitude=self.left_longitude)
        retrieved = elevation.retrieve_to_df()
        print(self.elevation_data)
        print(retrieved)
        output = pd.concat([self.elevation_data, retrieved], axis=0)
        if self.mode == 'w':
            output.to_csv(self.elevation_data_path, index=False)
        return X.merge(how='left', right=output, left_on=[self.left_longitude, self.left_latitude],
                       right_on=[self.elevation_longitude, self.elevation_latitude])


def multiply(func, **kwargs):
    def func_wrapper(from_string, thousand_eq=None, million_eq=None, billion_eq=None, thousand_mlpr = 1e3, million_mlpr = 1e6, billion_mlpr = 1e9):
        from_string = str(from_string)
        if billion_eq and billion_eq in from_string:
            multiplier = billion_mlpr
        elif million_eq and million_eq in from_string:
            multiplier = million_mlpr
        elif thousand_eq and thousand_eq in from_string:
            multiplier = thousand_mlpr
        else:
            multiplier = 1
        return func(from_string, **kwargs) * multiplier
    return func_wrapper

@multiply
def extract_num(from_string, decimal_sep='.'):
    ''' Extract all numeric values from string '''
    res=[s for s in from_string if s in string.digits or s==decimal_sep]
    num_s=''.join(res).replace(decimal_sep, '.')
    return float(num_s)
    
def test_set_check(identifier, test_ratio):
    if isinstance(identifier, str):
        return crc32(identifier.encode('ascii')) / 0xffffffff < test_ratio
    else:
        return crc32(np.int64(identifier)) / 0xffffffff < test_ratio

def split_train_test_by_hash(df, test_ratio, id_column):
    ids = df[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return df.loc[~in_test_set], df.loc[in_test_set]

def calc_outlier_ratio(st_df, columns, n_sigma = 10, sigma_step = 0.25):
    total_records = len(st_df)
    ratios = []
    for sigma in np.arange(n_sigma, 1, -sigma_step):
        all_mask = np.array([True] * total_records)
        for c in columns:
            mask = (st_df[c].values <= sigma) & (st_df[c].values >= sigma * (-1))
            all_mask = all_mask & mask
        filtered_records = len(st_df[all_mask])
        ratio = 1 - (filtered_records/total_records)
        ratios.append(ratio*100)
    return np.c_[np.arange(n_sigma, 1, -sigma_step), np.array(ratios)]    
    
def create_outlier_mask(st_df, columns, sigma_threshold = 3):
    total_records = len(st_df)
    outlier_mask = np.array([True] * total_records)
    for c in columns:
        mask = (st_df[c].values <= sigma_threshold) & (st_df[c].values >= sigma_threshold * (-1))
        outlier_mask = outlier_mask & mask
    return outlier_mask


def get_address_mask(series, public_domains_fn, street_num = True):
    public_domains = prep.load_public_domain_names(public_domains_fn)
    ptrn = '|'.join(['.* {}$'.format(d) for d in public_domains])
    if street_num:
        ptrn_dot = '|'.join(['.*{}.*\d\.$'.format(d) for d in public_domains])
        ptrn_slash_num = '|'.join(['.*{}.*\d/[0-9]$'.format(d) for d in public_domains])
        ptrn_slash_letter = '|'.join(['.*{}.*\d/[A-Z]$'.format(d) for d in public_domains])
        ptrn_dash= '|'.join(['.*{}.*\d*-\d*$'.format(d) for d in public_domains])
        ptrn = '|'.join([ptrn_dot, ptrn_slash_num, ptrn_slash_letter, ptrn_dash])
    mask = series.apply(lambda a: bool(re.match(string=str(a), pattern=ptrn)))
    return mask