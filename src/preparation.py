from typing import List
import logging
logging.basicConfig(level=logging.DEBUG)
from functools import wraps
import time
import pandas as pd, numpy as np
import json
import requests
from requests.exceptions import RequestException
from http.client import RemoteDisconnected
import overpy
from src.utils import read_txt, generate_intervals, load_stored_elevation


def retry(delay=5, retries=3, exception=Exception):
    def retry_decorator(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            opt_dict = {'retries': retries, 'delay': delay}
            while opt_dict['retries'] > 1:
                try:
                    return f(*args, **kwargs)
                except exception as e:
                    logging.debug('Exception: %s, Retrying in %d seconds...', e, delay)
                    time.sleep(opt_dict['delay'])
                    opt_dict['retries'] -= 1
            return f(*args, **kwargs)
        return f_retry

    return retry_decorator


@retry(delay=5, retries=5, exception=RequestException)
def get_open_elevation(latitude: float, longitude: float):
    TIMEOUT = 200
    API_ENDPOINT = 'https://api.open-elevation.com/api/v1/lookup'
    HEADERS = {'content-type': 'application/json', 'accept': 'application/json'}
    params = [{'locations': [{'latitude': latitude, 'longitude': longitude}]}]
    response = requests.post(url=API_ENDPOINT, timeout=TIMEOUT, data=json.dumps(params), headers=HEADERS)
    j = response.json()
    return pd.DataFrame(j['results'])


@retry(delay=5, retries=3, exception=RequestException)
def get_open_topo_elevation(latitude_list: List[float], longitude_list: List[float], timeout: int = 200):
    MAX_BATCH_SIZE = 100
    all_locations = [f'{lat},{long}' for lat, long in zip(latitude_list, longitude_list)]
    n_batches = int(np.ceil(len(all_locations) / MAX_BATCH_SIZE))
    intervals = generate_intervals(n_batches, len(all_locations))
    results = []
    for interval in intervals:
        locations = all_locations[interval.start:interval.stop]
        locations_formatted = '|'.join(locations)
        API_ENDPOINT = f'https://api.opentopodata.org/v1/eudem25m?locations={locations_formatted}'
        response = requests.get(url=API_ENDPOINT, timeout=timeout)
        if not response.ok:
            raise RequestException(response.reason)
        j = response.json()
        results.extend(j['results'])
    return results


class Elevation:

    def __init__(self, df: pd.DataFrame, latitude_alias: str, longitude_alias: str, use_only_unique: bool = True):
        self.use_only_unique = use_only_unique
        self.latitude_alias = latitude_alias
        self.longitude_alias = longitude_alias
        self.df = df

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df):
        if self.use_only_unique:
            self._df = df[[self.latitude_alias, self.longitude_alias]].drop_duplicates().reset_index(drop=True)
        else:
            self._df = df
        logging.info('Total number of unique GPS coordinates: %d', len(self._df))

    def retrieve_to_df(self):
        latitude, longitude = self.df[self.latitude_alias].tolist(), self.df[self.longitude_alias].tolist()
        data = get_open_topo_elevation(latitude_list=latitude, longitude_list=longitude)
        elevation = [d['elevation'] for d in data]
        return pd.DataFrame({'elevation': elevation, 'latitude': latitude, 'longitude': longitude}).reset_index(drop=True)


class OSM:

    def __init__(self, query):
        self.query = query
        overpass_api = overpy.Overpass()
        self.result = overpass_api.query(self.query)
    
    def nodes_to_df(self, node_attrs=['id', 'lat', 'lon']):
        nodes = [[n.__getattribute__(a) for a in node_attrs] for n in self.result.nodes]
        df = pd.DataFrame(nodes, columns=node_attrs)
        return df
    
    def to_df(self, node_attrs=['id', 'lat', 'lon'], add_tags=[]):
        data = []
        for rel in self.result.relations:
            tags = [rel.tags[t] for t in add_tags]
            for m in rel.members:
                if isinstance(m, overpy.RelationWay):
                    w = m.resolve()
                    nodes = [[n.__getattribute__(a) for a in node_attrs] + tags for n in w.nodes]
                    data.extend(nodes)
                elif isinstance(m, overpy.RelationNode):
                    try:
                        node = [m.__getattribute__(a) for a in node_attrs] + tags
                    except AttributeError:
                        rm = m.resolve()
                        node = [rm.__getattribute__(a) for a in node_attrs] + tags
                    data.append(node)
        df = pd.DataFrame(data, columns=node_attrs + add_tags)
        return df


def get_coordinates_from(geojson):
    rows = []
    for feat in geojson['features']:
        nodes = feat['geometry']['coordinates']
        for node in nodes:
            if len(node) > 2:
                for r in node:
                    rows.append(r)
            else:
                rows.append(node)
    return pd.DataFrame(rows, columns=['lng', 'lat'])


def get_public_domain_names():
    URL = 'https://ceginformaciosszolgalat.kormany.hu/download/b/46/11000/kozterulet_jelleg_2015_09_07.txt'
    response = requests.get(URL)
    txt = response.read()
    decoded_txt = txt.decode(encoding='utf-8-sig')
    return decoded_txt


def load_public_domain_names(txt_path):
    txt = read_txt(txt_path, encoding=None)
    return [line for line in txt.split('\n') if len(line)>0]


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

    t = get_open_topo_elevation(latitude_list=a[:, 0].tolist(), longitude_list=a[:, 1].tolist())
    # elevation = Elevation(df=sample_gps_data(), latitude_alias='lat', longitude_alias='lng')
    # result = elevation.retrieve_to_df()

