import pytest
from src.preparation import *
from .conftest import IS_TEST_SKIPPED


class TestElevation:

    @pytest.mark.skipif(IS_TEST_SKIPPED, reason='slow test')
    def test_retrieve_to_df(self, sample_gps_data):
        single_sample = sample_gps_data.loc[:0, :]
        elevation = Elevation(df=single_sample, latitude='lat', longitude='lng')
        result = elevation.retrieve_to_df()
        assert result.at[0, 'elevation'] == 222
