from typing import Any, List, Callable, Union
import re
import logging
from datetime import datetime
import pandas as pd, numpy as np
import string
from zlib import crc32
from src.base import BaseTransformer
from src.preparation import Elevation
import src.preparation as prep
from src.utils import load_stored_elevation, store_elevation


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
        X[self.column_names] = X[self.column_names].apply(self.func)
        return X


class DuplicatesRemoval(BaseTransformer):

    def __init__(self, columns: List[str], is_columns_negated: bool = False):
        self.columns = columns
        self.is_columns_negated = is_columns_negated


    def get_to_be_modified_columns(self, X):
        if self.is_columns_negated:
            return X.columns[~X.columns.isin(self.columns)].tolist()
        return self.columns

    def transform(self, X):
        to_be_modified_columns = self.get_to_be_modified_columns(X)
        unique_X = X.drop_duplicates(subset=to_be_modified_columns)
        total_records = len(X)
        duplicated_records_ratio = 1 - len(unique_X) / total_records
        print('Total number of records: {0:,}'.format(total_records))
        print('Duplicated records are {0:.3%}'.format(duplicated_records_ratio))
        return unique_X


class ElevationMerger(BaseTransformer):

    def __init__(self, left_longitude: str, left_latitude: str, stored_elevation_path: str,
                 stored_elevation_longitude: str,
                 stored_elevation_latitude: str, rounding_decimals=6):
        self.left_longitude = left_longitude
        self.left_latitude = left_latitude
        self.stored_elevation_path = stored_elevation_path
        self.stored_elevation_longitude = stored_elevation_longitude
        self.stored_elevation_latitude = stored_elevation_latitude
        self.rounding_decimals = rounding_decimals

    @property
    def stored_elevation(self):
        stored_elevation = load_stored_elevation(self.stored_elevation_path)
        return stored_elevation

    @property
    def rounded_stored_elevation(self):
        to_be_rounded = [self.stored_elevation_longitude, self.stored_elevation_latitude]
        rounded_stored_elevation = self.stored_elevation.copy()
        rounded_stored_elevation[to_be_rounded] = self.stored_elevation[to_be_rounded].round(
            decimals=self.rounding_decimals)
        return rounded_stored_elevation

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        return X.merge(how='left', right=self.rounded_stored_elevation, left_on=[self.left_longitude, self.left_latitude],
                       right_on=[self.stored_elevation_longitude, self.stored_elevation_latitude])


class ElevationInserter(ElevationMerger):

    def __init__(self, left_longitude: str, left_latitude: str, stored_elevation_path: str,
                 stored_elevation_longitude: str,
                 stored_elevation_latitude: str, rounding_decimals=6, mode: str = 'w'):
        self.mode = mode
        super().__init__(left_longitude=left_longitude, left_latitude=left_latitude,
                         stored_elevation_path=stored_elevation_path,
                         stored_elevation_longitude=stored_elevation_longitude,
                         stored_elevation_latitude=stored_elevation_latitude, rounding_decimals=rounding_decimals)

    def get_unstored_elevation(self, X: pd.DataFrame):
        merged_to_rounded_elevation = X.merge(how='left', right=self.rounded_stored_elevation,
                         left_on=[self.left_longitude, self.left_latitude],
                         right_on=[self.stored_elevation_longitude, self.stored_elevation_latitude])
        unstored_elevation_mask = merged_to_rounded_elevation[[self.stored_elevation_longitude, self.stored_elevation_latitude]].isnull().any(
            axis=1)
        unstored_elevation = merged_to_rounded_elevation[unstored_elevation_mask]
        return unstored_elevation

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X[[self.left_longitude, self.left_latitude]] = X[[self.left_longitude, self.left_latitude]].round(
            decimals=self.rounding_decimals)
        unstored_elevation = self.get_unstored_elevation(X)

        logging.info('Not in elevation data: %s', len(unstored_elevation))
        retrieved_unstored_elevation = Elevation(df=unstored_elevation, batch_size=100, latitude=self.left_latitude,
                                      longitude=self.left_longitude).retrieve_to_df()
        elevation = pd.concat([self.rounded_stored_elevation, retrieved_unstored_elevation], axis=0)
        if self.mode == 'w' and len(retrieved_unstored_elevation) > 0:
            store_elevation(elevation=elevation, file_path=self.stored_elevation_path)


class DropColumns(BaseTransformer):

    def __init__(self, columns: List[str], is_columns_negated: bool = False):
        self.columns = columns
        self.is_columns_negated = is_columns_negated

    def get_to_be_modified_columns(self, X):
        if self.is_columns_negated:
            return X.columns[~X.columns.isin(self.columns)].tolist()
        return self.columns

    def transform(self, X):
        to_be_modified_columns = self.get_to_be_modified_columns(X)
        return X.drop(columns=to_be_modified_columns)


class IdCreator(BaseTransformer):

    def __init__(self, columns: List[str], date_format: str = '%Y-%m-%d %H:%M:%S.%f', id_column_name: str = 'id'):
        self.columns = columns
        self.date_format = date_format
        self.id_column_name = id_column_name

    def get_numeric_columns(self, X):
        return [c for c in self.columns if pd.api.types.is_numeric_dtype(X[c])]

    def get_str_columns(self, X):
        return [c for c in self.columns if pd.api.types.is_string_dtype(X[c])]

    def transform(self, X):
        X[self.id_column_name] = 0
        numeric_columns = self.get_numeric_columns(X)
        str_columns = self.get_str_columns(X)
        for c in numeric_columns:
            X[self.id_column_name] = X[self.id_column_name] + X[c]
        for c in str_columns:
            X[self.id_column_name] = X[self.id_column_name] + X[c].apply(
                lambda ts: datetime.strptime(ts, self.date_format).timestamp())
        return X


class FunctionApplier(BaseTransformer):

    def __init__(self, function: Callable, columns: List[str], new_columns: List[str]):
        self.function = function
        self.columns = columns
        self.new_columns = new_columns

    def transform(self, X):
        output = X.copy()
        for col, new_col in zip(self.columns, self.new_columns):
            if col == new_col:
                logging.warning('%s is being overwritten', col)
            output[new_col] = output[col].apply(self.function)
        return output


class ColumnAdder(BaseTransformer):

    def __init__(self, left_columns: List[str], right_columns: List[str], new_columns: List[str]):
        self.left_columns = left_columns
        self.right_columns = right_columns
        self.new_columns = new_columns

    def transform(self, X):
        output = X.copy()
        for left, right, new_col in zip(self.left_columns, self.right_columns, self.new_columns):
            output[new_col] = output[left] + output[right]
        return output


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


def get_multiplier(from_string: str, thousand_eq: str = None, million_eq: str = None, billion_eq: str = None,
                    thousand_mlpr: float = 1e3, million_mlpr: float = 1e6, billion_mlpr: float = 1e9) -> float:
    """
    Extract multiplier number based on the multiplier literal in given string.
    Parameters
    ----------
    from_string: str
        String, which contains multiplier literal
    thousand_eq: str, optional
        Thousand literal
    million_eq: str, optional
        Million literal
    billion_eq: str, optional
        Billion literal
    thousand_mlpr: float, optional
        Thousand multiplier
    million_mlpr: float, optional
        Million multiplier
    billion_mlpr: float, optional
        Billion multiplier

    Returns
    -------
        Multiplier number
    """
    from_string = str(from_string)
    if billion_eq and billion_eq in from_string:
        multiplier = billion_mlpr
    elif million_eq and million_eq in from_string:
        multiplier = million_mlpr
    elif thousand_eq and thousand_eq in from_string:
        multiplier = thousand_mlpr
    else:
        multiplier = 1
    return multiplier


def is_str(input: Any) -> bool:
    """
    Checks, whether input is a string.
    Parameters
    ----------
    input
        Input, which should be tested.

    Returns
    -------
    Bool, True if it is string, False it is not a string

    """
    if not isinstance(input, str):
        if pd.isna(input):
            logging.debug('%s is not a string, but not number', input)
        else:
            logging.warning('%s is not a string', input)
        return False
    return True


def extract_num(from_string: Any, decimal_sep: str = '.', thousand_eq: str = None, million_eq: str = None,
                billion_eq: str = None, thousand_mlpr: float = 1e3, million_mlpr: float = 1e6,
                billion_mlpr: float = 1e9) -> Union[float, Any]:
    """
    Extract number from a string.
    Parameters
    ----------
    from_string: str
        String, which contains number
    decimal_sep: str, optional
        Decimal separator, by default using '.'
    Returns
    -------
        Float number from the string
    """
    if is_str(from_string):
        multiplier = get_multiplier(from_string, thousand_eq=thousand_eq, million_eq=million_eq, billion_eq=billion_eq,
                                    thousand_mlpr=thousand_mlpr, million_mlpr=million_mlpr, billion_mlpr=billion_mlpr)
        res = [s for s in from_string if s in string.digits or s == decimal_sep]
        num_s = ''.join(res).replace(decimal_sep, '.')
        return float(num_s) * multiplier
    else:
        return from_string


def check_test_set(identifier, test_ratio):
    if isinstance(identifier, str):
        return crc32(identifier.encode('ascii')) / 0xffffffff < test_ratio
    else:
        return crc32(np.int64(identifier)) / 0xffffffff < test_ratio


def split_train_test_by_hash(df, test_ratio, id_column):
    ids = df[id_column]
    in_test_set = ids.apply(lambda id_: check_test_set(id_, test_ratio))
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


if __name__ == '__main__':
    def sample_gps_data():
        a = np.array([[47.52991, 18.992949],
                      [47.54727, 19.07117],
                      [47.51102, 19.07725],
                      [47.488605, 19.075905],
                      [47.4746, 18.9899],
                      [47.42004, 19.0021],
                      [47.53713, 19.12761],
                      [47.49086, 19.136728],
                      [47.47963, 18.992636],
                      [47.478138, 19.231878]])
        df = pd.DataFrame(data=a, columns=['lat', 'lng'])
        return df

    sample_gps_data = sample_gps_data()
    dummy_elevation_path = 'tests/fixtures/dummy_elevation.csv'
    em = ElevationMerger(left_latitude='lat', left_longitude='lng', stored_elevation_path=dummy_elevation_path,
                         stored_elevation_latitude='latitude', stored_elevation_longitude='longitude')
    res = em.transform(sample_gps_data)
