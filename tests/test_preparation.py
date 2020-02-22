import pytest
from src.preparation import *


def test_get_open_topo_elevation(open_topo_input_data, open_topo_elevation_data):
    latitude, longitude = open_topo_input_data['latitude'], open_topo_input_data['longitude']
    results = get_open_topo_elevation(latitude, longitude)
    assert len(results) == 10


class TestElevation:

    @pytest.mark.api_call
    def test_retrieve_to_df(self, sample_gps_data):
        single_sample = sample_gps_data.loc[:0, :]
        elevation = Elevation(df=single_sample, latitude_alias='lat', longitude_alias='lng')
        result = elevation.retrieve_to_df()
        assert result.at[0, 'elevation'] - 219 < 1
