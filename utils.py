import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import string, json
from zlib import crc32
import sys
import urllib
REAL_ESTATE_HUN_DIR='../real_estate_hungary/'
sys.path.append(REAL_ESTATE_HUN_DIR)
from real_estate_hungary import RequestWithHeaders

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
    
def load_danube_gps_coordinates(json_path):
    geojson=load_json(json_path)
    return parse_geojson(geojson)
    
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

def plot_scatter_map(x, y, figsize, xlabel=None, ylabel=None, epsg_code=4326, dpi=160):
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
    plt.scatter(x=x, y=y, s=25, alpha=1, linewidth=0.25, edgecolor='Black')
    plt.xticks(np.linspace(start=x.min(), stop=x.max(), num=np.ceil(width/2).astype(int)).round(2), fontsize=15)
    plt.yticks(np.linspace(start=y.min(), stop=y.max(), num=np.ceil(height/2).astype(int)).round(2), fontsize=15)
    plt.xlabel(x.name if xlabel is None else xlabel, fontsize=20)
    plt.ylabel(y.name if ylabel is None else ylabel, fontsize=20)
    plt.show()