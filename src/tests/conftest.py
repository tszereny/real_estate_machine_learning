import pytest
from src.utils import RealEstateData
from src.processing import ColumnRenamer, Translator
from pipeline import OLD_TO_NEW, HUN_TO_ENG, LISTING_TYPE_HUN_TO_ENG
import pandas as pd, numpy as np
import os


@pytest.fixture(scope='session')
def real_estate_data():
    FIXTURE_DIR = '../fixtures/'
    DATA_FNAME = 'sample.csv'
    red = RealEstateData(FIXTURE_DIR, DATA_FNAME)
    return red


@pytest.fixture(scope='session')
def real_estate_raw(real_estate_data):

    def _real_estate_raw(date_idx):
        dates = real_estate_data.directories
        return real_estate_data.read(dir_name='raw', date=dates[date_idx])

    return _real_estate_raw


@pytest.fixture(scope='session', params=[0, 1])
def real_estate_renamed(real_estate_raw, request):
    cr = ColumnRenamer(old_to_new=OLD_TO_NEW, hun_to_eng=HUN_TO_ENG)
    df = cr.transform(real_estate_raw(request.param))
    return df


@pytest.fixture(scope='session')
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


@pytest.fixture(scope='session')
def sample_elevation_data():
    a = np.array([[222., 47.52991, 18.992949],
                  [108., 47.54727, 19.07117],
                  [114., 47.51102, 19.07725],
                  [108., 47.488605, 19.075905],
                  [194., 47.4746, 18.9899],
                  [180., 47.42004, 19.0021],
                  [114., 47.53713, 19.12761]])
    df = pd.DataFrame(data=a, columns=['elevation', 'latitude', 'longitude'])
    return df
