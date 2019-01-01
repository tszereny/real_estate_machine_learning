import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import overpy
import string, pickle
import json
import smtplib, ssl, getpass
import urllib
from zlib import crc32
import sys, os
import urllib
REPO_DIR = os.path.join(os.environ['USERPROFILE'], 'repos')
REAL_ESTATE_HUN_DIR=os.path.join(REPO_DIR, 'real_estate_hungary')
sys.path.append(REAL_ESTATE_HUN_DIR)
from real_estate_hungary import RequestWithHeaders

class Email:
    
    def __init__(self, sender_email, receiver_email, port = 465, smtp_server = 'smtp.gmail.com'):
        self.sender_email = sender_email
        self.receiver_email = receiver_email
        self.port = port
        self.smtp_server = smtp_server
        self.context = ssl.create_default_context()
        self._password = getpass.getpass('Type your password and press enter: ')
        
    def send(self, msg):
        with smtplib.SMTP_SSL(self.smtp_server, self.port, context=self.context) as server:
            server.login(self.sender_email, self._password)
            server.sendmail(self.sender_email, self.receiver_email, msg)

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
        
def save_pkl(pkl_path, obj):
    with open(pkl_path, 'wb') as f:  
        pickle.dump(obj, f)
    return pkl_path

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:  
        obj = pickle.load(f)
    return obj
    
def generate_na(df, na_eq):
    gen_na=lambda x: None if x==na_eq else x
    if isinstance(df, pd.Series):
        return df.apply(gen_na)
    elif isinstance(df, pd.DataFrame):
        return df.applymap(gen_na)
        
def multiply(func, **kwargs):
    def func_wrapper(from_string, thousand_eq=None, million_eq=None, billion_eq=None):
        if thousand_eq==million_eq==billion_eq==None:
            multiplier = 1
        elif billion_eq in from_string:
            multiplier = 1e9
        elif million_eq in from_string:
            multiplier = 1e6
        elif thousand_eq in from_string:
            multiplier = 1e3
        else:
            multiplier = 1
        return func(from_string, **kwargs) * multiplier
    return func_wrapper

@multiply
def extract_num(from_string, decimal_sep='.'):
    ''' Extract all numeric values from string '''
    res=[s for s in from_string if s in string.digits or s==decimal_sep]
    num_s=''.join(res).replace(decimal_sep, '.')
    return float(num_s)

def test_set_check(identifier, test_ratio):
    if isinstance(identifier, str):
        return crc32(identifier.encode('ascii')) / 0xffffffff < test_ratio
    else:
        return crc32(np.int64(identifier)) / 0xffffffff < test_ratio

def split_train_test_by_hash(df, test_ratio, id_column):
    ids = df[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return df.loc[~in_test_set], df.loc[in_test_set]
    
def get_district_details_wikipedia():
    wiki_url='https://hu.wikipedia.org/wiki/Budapest_v%C3%A1rosr%C3%A9szeinek_list%C3%A1ja'
    r=RequestWithHeaders(url=wiki_url)
    html=r.parse_to_html()
    tbls=html.find_all('table')
    results=[]
    for tbl, i in zip(tbls, range(len(tbls)-1, -1, -1)):
        trows=[tr for tr in tbl.tbody.find_all('tr')]
        header=[th.get_text().strip() for th in trows[0].find_all('th')]
        body=[[td.get_text().strip().replace('\xa0', ' ') for td in tr.find_all('td')] for tr in trows[1:]]
        body=[rw for rw in body if len(rw)>0]
        df=pd.DataFrame(np.array(body)[:, i:], columns=header[i:])
        results.append(df)
    return pd.concat(results)

def get_public_domain_names():
    url='https://ceginformaciosszolgalat.kormany.hu/download/b/46/11000/kozterulet_jelleg_2015_09_07.txt'
    response=urllib.request.urlopen(url)
    txt=response.read()
    decoded_txt=txt.decode(encoding='utf-8-sig')
    return [line.strip() for line in decoded_txt.split('\r') if len(line)>1]
    
def load_public_domain_names(txt_path):
    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        txt=f.read()
    return [line for line in txt.split('\n') if len(line)>0]

def parse_geojson(geojson):
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

def load_json(json_path, encoding='utf8'):
    with open(json_path, encoding=encoding) as f:
        data=json.load(f)
    return data
    
def load_geoson_gps_coordinates(json_path):
    geojson=load_json(json_path)
    return parse_geojson(geojson)

def retreive_gps_overpass(overpass_query):
    overpass_api = overpy.Overpass()
    result = overpass_api.query(overpass_query)
    nodes = [(n.id, float(n.lat), float(n.lon)) for n in result.nodes]
    df = pd.DataFrame(nodes, columns=['id', 'lat', 'lng'])
    return df
    
def query_osm(q):
    overpass_api = overpy.Overpass()
    result = overpass_api.query(q)
    return result

def osm_result_to_df(result, add_tag=None):
    r=[]
    for rel in result.relations:
        if add_tag:
            col_val=[rel.tags[add_tag]]
            column_names=['id', 'lat', 'lng']+[add_tag]
        else:
            col_val=[]
            column_names=['id', 'lat', 'lng']
        for m in rel.members:
            if isinstance(m, overpy.RelationWay):
                w=m.resolve()
                for n in w.nodes:
                    rw=[n.id, float(n.lat), float(n.lon)]+col_val
                    r.append(rw)
            elif isinstance(m, overpy.RelationNode):
                rw=[n.id, float(n.lat), float(n.lon)]+col_val
                r.append(rw)
    df=pd.DataFrame(r, columns=column_names).drop_duplicates()
    return df

def osm_to_df(q, add_tag=None):
    result = query_osm(q)
    df = osm_result_to_df(result, add_tag)
    return df

def check_n_query_osm(file_p, query_p, add_tag=None, saving=True):
    if not os.path.exists(file_p):
        with open(query_p, encoding='utf8') as  f:
            osm_query=f.read()
        df = osm_to_df(osm_query, add_tag=add_tag)
        if saving: df.to_csv(file_p, index=False, encoding='utf8')
    else:
        df = pd.read_csv(file_p, encoding='utf8')
    return df

def calc_intervals(ints_n, length):
    r=[]
    strt=0
    for i in range(ints_n):
        if i==ints_n-1:
            stp=strt+(length-stp)
            r.append(range(strt, stp))
        else:
            stp=strt+int(np.ceil(length/ints_n))
            r.append(range(strt, stp))
            strt=stp
    return r

def calc_fig_size(pic_ratio, multiplier=2):
    width=10*multiplier
    height=pic_ratio*multiplier*10
    return (width, height)    

def generate_colors(df, c_map, condition):
    c=[]
    for k, s in df.items():
        c_ord=[c_map[k][0]]*s[s==condition].count() + [c_map[k][1]]*s[s!=condition].count()
        c.append(c_ord)
    return np.array(c)    
    
def plot_na_ratio(df, grouping, subplot_n=2,  dest_dir='.'):
    title='Missing values-ratio by features (%)'
    na_count=df.groupby(grouping).count()
    total=df.groupby(grouping).agg(lambda col: len(col))
    na_total_ratio=100-na_count/total*100
    groups=na_total_ratio.index.tolist()
    sorted_df=na_total_ratio.T.sort_values(groups, ascending=True)
    col_names=sorted_df.index.tolist()
    ints=calc_intervals(subplot_n, len(col_names))
    fig, axs = plt.subplots(subplot_n)
    fig.suptitle(title, fontsize=20, weight=10, y=0.93)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)

    for i, ax in enumerate(axs):
        ax_df=sorted_df.iloc[ints[i].start:ints[i].stop, :]
        ax_df.plot(kind='bar', figsize=(17,15), legend=True, ax=axs[i])
        axs[i].set_xticklabels(ax_df.index.tolist(), fontsize=14)
    fig.savefig('{0}/{1}.png'.format(dest_dir, title))
    return fig, axs
    
    
def plot_scatter_matrix(df, fs=15):
    col_n=len(df.columns)
    fig, axs = plt.subplots(col_n, col_n, figsize=(20,20))
    for c, c_k in enumerate(df):
        for r, r_k in enumerate(df):
            if c==r:
                axs[r,c].text(0.5, 0.5, s=c_k, horizontalalignment='center', verticalalignment='center', fontdict={'fontsize': fs})
                axs[r,c].xaxis.set_visible(False)
                axs[r,c].yaxis.set_visible(False)
            elif (c==0) & (r==col_n-1):
                df.plot(kind='scatter', x=c_k, y=r_k, ax=axs[r, c], s=4)
                axs[r,c].xaxis.set_visible(True)
                axs[r,c].yaxis.set_visible(True)
                axs[r,c].set_xlabel(c_k, fontdict={'fontsize': fs})
                axs[r,c].set_ylabel(r_k, fontdict={'fontsize': fs})
            elif c==0:
                df.plot(kind='scatter', x=c_k, y=r_k, ax=axs[r, c], s=4)
                axs[r,c].xaxis.set_visible(False)
                axs[r,c].yaxis.set_visible(True)
                axs[r,c].set_ylabel(r_k, fontdict={'fontsize': fs})
            elif r==col_n-1:
                df.plot(kind='scatter', x=c_k, y=r_k, ax=axs[r, c], s=4)
                axs[r,c].xaxis.set_visible(True)
                axs[r,c].set_xlabel(c_k, fontdict={'fontsize': fs})
                axs[r,c].yaxis.set_visible(False)
            elif (r==0) & (c==col_n-1):
                df.plot(kind='scatter', x=c_k, y=r_k, ax=axs[r, c], s=4)
                axs[r,c].xaxis.set_ticks_position('top')
                axs[r,c].yaxis.set_ticks_position('right')
                axs[r,c].xaxis.set_label_position('top')
                axs[r,c].yaxis.set_label_position('right')
                axs[r,c].set_xlabel(c_k, fontdict={'fontsize': fs})
                axs[r,c].set_ylabel(r_k, fontdict={'fontsize': fs})          
            elif c==col_n-1:
                df.plot(kind='scatter', x=c_k, y=r_k, ax=axs[r, c], s=4)
                axs[r,c].xaxis.set_visible(False)
                axs[r,c].set_ylabel(r_k, fontdict={'fontsize': fs})
                axs[r,c].yaxis.set_ticks_position('right')
                axs[r,c].yaxis.set_label_position('right')
            elif r==0:
                df.plot(kind='scatter', x=c_k, y=r_k, ax=axs[r, c], s=4)
                axs[r,c].xaxis.set_visible(True)
                axs[r,c].set_xlabel(c_k, fontdict={'fontsize': fs})
                axs[r,c].yaxis.set_visible(False)
                axs[r,c].xaxis.set_ticks_position('top')
                axs[r,c].xaxis.set_label_position('top')
            else:
                df.plot(kind='scatter', x=c_k, y=r_k, ax=axs[r, c], s=4)
                axs[r,c].xaxis.set_visible(False)
                axs[r,c].yaxis.set_visible(False)
            plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axs

def plot_scatter_map(figsize, x, y, c=None, colormap=None, cbar=True, c_bar_shrink=1, xlabel=None, ylabel=None, clabel=None, s=25, linewidth=0.25, alpha=1, epsg_code=4326, dpi=160):
    m = Basemap(urcrnrlat=y.max(),     # top
              urcrnrlon=x.max(),   # bottom
              llcrnrlat=y.min(),     # left
              llcrnrlon=x.min(),   # right
              epsg=epsg_code)
    width = figsize[0]
    height = figsize[1]
    fig = plt.gcf()
    fig.set_size_inches(width, height)
    dpi = dpi
    xpixels = dpi * width
    m.arcgisimage(service='Canvas/World_Light_Gray_Base', xpixels=xpixels)
    plt.scatter(x=x, y=y, c=c, cmap=colormap, s=s, alpha=alpha, linewidth=linewidth, edgecolor='Black')
    plt.xticks(np.linspace(start=x.min(), stop=x.max(), num=np.ceil(width/2).astype(int)).round(2), fontsize=15)
    plt.yticks(np.linspace(start=y.min(), stop=y.max(), num=np.ceil(height/2).astype(int)).round(2), fontsize=15)
    plt.xlabel(x.name if xlabel is None else xlabel, fontsize=20)
    plt.ylabel(y.name if ylabel is None else ylabel, fontsize=20)
    if c is not None and cbar:
        cbar = plt.colorbar(orientation='vertical', shrink=c_bar_shrink)
        cbar.set_label(c.name if clabel is None else clabel, rotation=90, fontsize=20)
        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=15)
    return plt.gcf(), plt.gca()
    
def plot_sca_hist(df, x, y, bins, xlabel=None, ylabel=None, fs=15):
    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(20,20)
    df.plot(kind='scatter', x=x, y=y, ax=axs[1,0], s=25, alpha=0.4, linewidth=0.25, edgecolor='Black', fontsize=fs)
    df.hist(x, ax=axs[0,0], bins=bins, linewidth=0.5, edgecolor='Black', grid=False, sharey=True, xlabelsize=fs, ylabelsize=fs, label=None)
    df.hist(y, ax=axs[1,1], bins=bins, linewidth=0.5, edgecolor='Black', grid=False, orientation='horizontal', sharex=True, xlabelsize=fs, ylabelsize=fs)
    axs[0,1].axis('off')
    axs[0,0].xaxis.set_ticks([])
    axs[1,1].yaxis.set_ticks([])
    axs[1,0].set_xlabel(x if xlabel is None else xlabel, fontsize=20)
    axs[1,0].set_ylabel(y if ylabel is None else ylabel, fontsize=20)
    axs[0,0].set_title('')
    axs[1,1].set_title('')
    plt.tight_layout(w_pad=-2, h_pad=-1.5)
    return fig, axs

def plot_sca_stackedhist(df, x, y, category, colors):
    FACE_C='lightgrey'
    df_by_categories=[df.loc[df[category]==cat_cond, [x, y, category]] for cat_cond in df[category].drop_duplicates()]
    x_by_cats=[df[x].values for df in df_by_categories]
    y_by_cats=[df[y].values for df in df_by_categories]
    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(20,20)
    axs[1,0].set_facecolor(FACE_C)
    axs[1,0].grid(True)
    for x, y, c in zip(x_by_cats, y_by_cats, colors):
        axs[1,0].scatter(x=x, y=y, color=c, linewidth=0.25, edgecolor='Black', alpha=0.5)
    axs[0,0].set_facecolor(FACE_C)
    axs[0,0].grid(True)
    axs[0,0].hist(x_by_cats, color=colors, histtype='barstacked', bins=80, linewidth=0.3, edgecolor='Black')
    axs[1,1].set_facecolor(FACE_C)
    axs[1,1].grid(True)
    axs[1,1].hist(y_by_cats, color=colors, histtype='barstacked', bins=80, orientation='horizontal', linewidth=0.3, edgecolor='Black')
    axs[0,1].axis('off')
    plt.tight_layout()
    return fig, axs    
    
def plot_outliers(st_df, n_std=6, figsize=(30,10)):
    n_cols=len(st_df.columns)
    fig, axs = plt.subplots(1, n_cols, figsize=figsize)
    for ax, k in zip(axs, st_df):
        ax.scatter(range(len(st_df[k])), st_df[k], marker='o', s=10)
        ax.set_title(k, fontdict={'fontsize':20})
        for std in range(-n_std, n_std+1):
            ax.axhline(std, c='r')
    return fig, axs

def create_grid(x_min, x_max, y_min, y_max, data_points, model, x_first_feature=True):
    X, Y=np.meshgrid(np.linspace(x_min, x_max, data_points),
                         np.linspace(y_min, y_max, data_points))
    Z=np.ones((data_points,data_points))
    for i in range(0, data_points):
        if x_first_feature:
            Z[i] = model.predict(X=np.stack([X[i], Y[i]], axis=1))
        else:
            Z[i] = model.predict(X=np.stack([Y[i], X[i]], axis=1))
    return X, Y, Z

def plot_3d_surface(X, Y, Z, elevation=50, rotation=45, figsize=(15,15), saving=False, dir_name=None, model_name=None):
    fig = plt.figure(figsize=(15,15))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X=X, Y=Y, Z=Z, cmap="coolwarm")
    ax.view_init(elevation, rotation)
    if saving and model_name and dir_name:
        fn='model_{0}_elev_{1}_rotat_{2}.png'.format(model_name, elevation, rotation)
        fig.savefig(os.path.join(dir_name, fn))
    return fig, ax