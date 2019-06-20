import pandas as pd, numpy as np
import urllib, json
from urllib.request import HTTPDefaultErrorHandler, HTTPError, URLError
from http.client import RemoteDisconnected
import overpy
from src.utils import read_txt, calc_intervals, load_elevation_data


class Elevation:
    HEADERS = {'content-type': 'application/json', 'accept': 'application/json'}
    API = 'https://api.open-elevation.com/api/v1/lookup'

    def __init__(self, df, latitude, longitude, batch_size=None, use_only_unique=True):
        self.only_unique = use_only_unique
        self.latitude = latitude
        self.longitude = longitude
        self.df = df
        self.batch_size = len(self.df) if batch_size is None else batch_size
        self.dfs = [df[interval.start:interval.stop] for interval in self.batch_intervals]

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df):
        self._df = df[[self.latitude, self.longitude]].drop_duplicates().reset_index(drop=True)
        print('Total number of unique GPS coordinates: {0:,}'.format(len(self._df)))

    @property
    def n_batches(self):
        n_batches = int(np.ceil(len(self.df)/self.batch_size))
        return n_batches
    
    @property
    def batch_intervals(self):
        intervals = calc_intervals(self.n_batches, len(self.df)) if self.n_batches > 1 else [range(0, len(self.df))]
        return intervals
        
    @property
    def locations_params(self):
        params = [{'locations':[{'latitude': r[self.latitude], 'longitude': r[self.longitude]} for i, r in df[[self.latitude, self.longitude]].iterrows()]} for df in self.dfs]
        return params
    
    @property
    def json_params(self):
        params_json = [json.dumps(lp).encode('utf8') for lp in self.locations_params]
        return params_json
        
    def _retrieve_to_df(self, timeout=200, batch_idx=0):
        req = urllib.request.Request(url=self.API, method='POST', data=self.json_params[batch_idx], headers=self.HEADERS)
        response_stream = urllib.request.urlopen(req, timeout=timeout)
        response = response_stream.read()
        response_stream.close()
        parsed_response = json.loads(response.decode('utf8'))
        return pd.DataFrame(parsed_response['results'])
    
    def retrieve_to_df(self):
        result = pd.DataFrame()
        for b in range(self.n_batches):
            while True:
                batch = None
                try:
                    batch = self._retrieve_to_df(batch_idx=b)
                    print('{}. success - batch retrieved!'. format(b + 1))
                except HTTPError as err:
                    pass
                except URLError as err:
                    pass
                except RemoteDisconnected as err:
                    pass
                if batch is not None:
                    break
            result = pd.concat([result, batch], axis=0)
        return result.reset_index(drop=True)

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
        nodes=feat['geometry']['coordinates']
        for node in nodes:
            if len(node) > 2:
                for r in node:
                    rows.append(r)
            else:
                rows.append(node)
    return pd.DataFrame(rows, columns=['lng', 'lat'])

def get_public_domain_names():
    URL = 'https://ceginformaciosszolgalat.kormany.hu/download/b/46/11000/kozterulet_jelleg_2015_09_07.txt'
    response = urllib.request.urlopen(URL)
    txt = response.read()
    decoded_txt = txt.decode(encoding='utf-8-sig')
    return decoded_txt

def load_public_domain_names(txt_path):
    txt = read_txt(txt_path, encoding = None)
    return [line for line in txt.split('\n') if len(line)>0]