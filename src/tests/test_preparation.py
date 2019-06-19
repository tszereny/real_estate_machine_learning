import pytest
from src.preparation import *

@pytest.fixture
def input_data():
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


@pytest.fixture
def output_data():
    a = np.array([[222., 47.52991, 18.992949],
                  [108., 47.54727, 19.07117],
                  [114., 47.51102, 19.07725],
                  [108., 47.488605, 19.075905],
                  [194., 47.4746, 18.9899],
                  [180., 47.42004, 19.0021],
                  [114., 47.53713, 19.12761]])
    df = pd.DataFrame(data=a, columns=['elevation', 'latitude', 'longitude'])
    return df

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
