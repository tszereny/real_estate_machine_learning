import pytest
from src.utils import RealEstateData
from src.processing import ColumnRenamer
from src.pipelines import OLD_TO_NEW, HUN_TO_ENG
import pandas as pd, numpy as np

IS_TEST_SKIPPED = True
FIXTURE_DIR = 'tests/fixtures/'

@pytest.fixture(scope='session')
def real_estate_data():
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
def open_topo_input_data():
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
    df = pd.DataFrame(data=a, columns=['latitude', 'longitude'])
    return df


@pytest.fixture(scope='session')
def open_topo_elevation_data():
    a = np.array([[219.21203613, 47.52991, 18.992949],
                  [109.00254059, 47.54727, 19.07117],
                  [116.33333588, 47.51102, 19.07725],
                  [111.09939575, 47.488605, 19.075905],
                  [201.97969055, 47.4746, 18.9899],
                  [182.68031311, 47.42004, 19.0021],
                  [118.37011719, 47.53713, 19.12761],
                  [131.17958069, 47.49086, 19.136728],
                  [207.88407898, 47.47963, 18.992636],
                  [148.72012329, 47.478138, 19.231878]])
    df = pd.DataFrame(data=a, columns=['elevation', 'latitude', 'longitude'])
    return df


@pytest.fixture(scope='function')
def mock_open_topo_requests(requests_mock, open_topo_elevation_data):
    elevation, latitude, longitude = open_topo_elevation_data['elevation'].tolist(), open_topo_elevation_data[
        'latitude'].tolist(), open_topo_elevation_data['longitude'].tolist()
    locations = [f'{lat},{long}' for lat, long in zip(latitude, longitude)]
    locations_formatted = '|'.join(locations)
    API_ENDPOINT = f'https://api.opentopodata.org/v1/eudem25m?locations={locations_formatted}'
    j = {"results": [{"elevation": ele, "location": {"lat": lat, "lng": lng}} for ele, lat, lng in
                     zip(elevation, latitude, longitude)], "status": "OK"}
    requests_mock.get(API_ENDPOINT, json=j)
