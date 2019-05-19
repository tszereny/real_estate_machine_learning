import pytest
from pandas import DataFrame
from src.utils import RealEstateData

@pytest.fixture(scope='module')
def real_estate():
    FIXTURE_DIR = '../fixtures/'
    DATA_FNAME = 'sample.csv'
    red = RealEstateData(FIXTURE_DIR, DATA_FNAME)
    return red

class TestRealEstateData:

    def test_directories(self, real_estate):
        for d in real_estate.directories:
            assert d in ['raw', 'interim', 'processed']

    @pytest.mark.parametrize('dir_name', real_estate().directories)
    def test_files(self, real_estate, dir_name):
        for f in real_estate.list_files(dir_name):
            f in ['20181101', '20190201']

    @pytest.mark.parametrize('date', real_estate().list_files('raw'))
    def test_read_data_in_raw(self, date, real_estate):
        df = real_estate.read_data('raw', date)
        assert isinstance(df, DataFrame)
        assert len(df) < 600
        assert df.applymap(lambda c: c in real_estate.NA_EQUIVALENTS).any().any() == False