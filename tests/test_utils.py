import pytest
from pandas import DataFrame


class TestRealEstateData:

    def test_directories(self, real_estate_data):
        for d in real_estate_data.directories:
            assert d in ['20181101', '20190201']

    def test_read_data_in_raw(self, real_estate_data):
        for date in real_estate_data.directories:
            df = real_estate_data.read(dir_name='raw', date=date)
            assert isinstance(df, DataFrame)
            assert len(df) < 600
            assert df.applymap(lambda c: c in real_estate_data.NA_EQUIVALENTS).any().any() == False
