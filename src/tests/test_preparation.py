import pytest
from src.preparation import *

@pytest.mark.skipif(True, reason='slow test')
def test_append_to_elevation_data(input_data, output_data):
    res = append_to_elevation_data(input_df=input_data, input_latitude='lat', input_longitude='lng',
                                   output_df=output_data,
                                   output_latitude='latitude', output_longitude='longitude')
    assert len(res) == 3
    assert res['elevation'].isin([130, 200, 145]).all()


class TestElevation:

    @pytest.mark.skipif(True, reason='slow test')
    def test_retrieve_to_df(self, input_data):
        single_sample = input_data.loc[:0, :]
        elevation = Elevation(df=single_sample, latitude='lat', longitude='lng')
        result = elevation.retrieve_to_df()
        assert result.at[0, 'elevation'] == 222
