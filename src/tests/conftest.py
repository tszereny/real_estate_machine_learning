from src.utils import RealEstateData
import pytest
import pandas as pd
import os


@pytest.fixture(scope='module')
def real_estate_data():
    FIXTURE_DIR = '../fixtures/'
    DATA_FNAME = 'sample.csv'
    red = RealEstateData(FIXTURE_DIR, DATA_FNAME)
    return red

