import pandas as pd, numpy as np
import urllib, json
import overpy
from src.utils import read_txt

class Elevation:
    
    def __init__(self, df, latitude, longitude):
        self.HEADERS={'content-type': 'application/json', 'accept': 'application/json'}
        self.API = 'https://api.open-elevation.com/api/v1/lookup'
        self.df = df
        self.latitude = latitude
        self.longitude = longitude
    
    @property
    def locations_params(self):
        params = {'locations':[{'latitude': r[self.latitude], 'longitude': r[self.longitude]} for i, r in self.df[[self.latitude, self.longitude]].iterrows()]}
        return params
    
    @property
    def json_params(self):
        params_json = json.dumps(self.locations_params).encode('utf8')
        return params_json
    
    def retrieve_to_df(self, timeout=200):
        req = urllib.request.Request(url=self.API, method='POST', data=self.json_params, headers=self.HEADERS)
        response_stream = urllib.request.urlopen(req, timeout=timeout)
        response = response_stream.read()
        response_stream.close()
        parsed_response = json.loads(response.decode('utf8'))
        return pd.DataFrame(parsed_response['results'])

class OSM:

    def __init__(self, query):
        self.query = query
        overpass_api = overpy.Overpass()
        self.result = overpass_api.query(self.query)
    
    def nodes_to_df(self, node_attrs = ['id', 'lat', 'lon']):
        nodes = [[n.__getattribute__(a) for a in node_attrs] for n in self.result.nodes]
        df = pd.DataFrame(nodes, columns=node_attrs)
        return df
    
    def to_df(self, node_attrs = ['id', 'lat', 'lon'], add_tags = []):
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
        df=pd.DataFrame(data, columns=node_attrs + add_tags)
        return df
    
def get_coordinates_from(geojson):
    rows=[]
    for feat in geojson['features']:
        nodes=feat['geometry']['coordinates']
        for node in nodes:
            if len(node)>2:
                for r in node:
                    rows.append(r)
            else:
                rows.append(node)
    return pd.DataFrame(rows, columns=['lng', 'lat'])

def get_public_domain_names():
    URL='https://ceginformaciosszolgalat.kormany.hu/download/b/46/11000/kozterulet_jelleg_2015_09_07.txt'
    response=urllib.request.urlopen(URL)
    txt=response.read()
    decoded_txt=txt.decode(encoding='utf-8-sig')
    return decoded_txt

def load_public_domain_names(txt_path):
    txt=read_txt(txt_path, encoding = None)
    return [line for line in txt.split('\n') if len(line)>0]