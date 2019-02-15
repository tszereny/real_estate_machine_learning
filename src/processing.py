import re
import pandas as pd, numpy as np
import string
from zlib import crc32
import src.preparation as prep

def generate_na(df, na_eqs):
    gen_na=lambda x: None if x in na_eqs else x
    if isinstance(df, pd.Series):
        return df.apply(gen_na)
    elif isinstance(df, pd.DataFrame):
        return df.applymap(gen_na)

def rename_cols_to_new(df):
    OLD_NEW={'data_id':'property_id', 'description':'desc',
    'district':'city_district','latitude': 'lat',
    'link':'property_url', 'load_time':'timestamp',
    'longitude': 'lng', 'max_list': 'max_listing',
    'price':'price_in_huf', 'quadmeter': 'area_size',
    'rooms': 'room', 'type': 'listing_type'}

    df.columns=df.columns.str.lower()
    return df.rename(columns=OLD_NEW)

def rename_cols_to_eng(df):
    HUN_ENG={'akadálymentesített':'accessibility', 'belmagasság':'ceiling_height', 'busz':'buses',
    'busz_count':'buses_count', 'bútorozott':'furnished','dohányzás':'smoking', 'emelet':'floors',
    'energiatanúsítvány': 'energy_perf_cert', 'erkély':'balcony','fürdő_és_wc':'bath_and_wc',
    'fűtés':'type_of_heating', 'gépesített':'equipped', 'hajó':'boats', 'hajó_count': 'boats_count',
    'hév':'local_railways','hév_count': 'local_railways_count', 'ingatlan_állapota':'condition_of_real_estate',
    'kertkapcsolatos':'with_entry_to_garden', 'kilátás':'view','kisállat':'pets', 'komfort':'convenience_level',
    'költözhető':'vacant', 'lakópark_neve':'residental_park_name','légkondicionáló': 'air_conditioned',
    'metró':'metro_lines', 'metró_count':'metro_lines_count', 'min._bérleti_idő': 'min_tenancy','parkolás':'parking',
    'parkolóhely_ára': 'parking_lot_price','rezsiköltség':'utilities','tetőtér':'attic',
    'troli':'trolley_buses', 'troli_count':'trolley_buses_count','tájolás':'orientation',
    'villamos':'trams','villamos_count':'trams_count','éjszakai':'all_night_services',
    'éjszakai_count':'all_night_services_count','építés_éve': 'year_built','épület_szintjei':'building_floors'}
    return df.rename(columns=HUN_ENG)
   
def translate_listing_type(df, col_n='listing_type'):
    LISTING_HUN_ENG={'elado': 'for-sale', 'kiado': 'for-rent'}
    df[col_n]=df[col_n].map(LISTING_HUN_ENG)
    return df
    
def transform_naming(df):
    df=rename_cols_to_new(df)
    df=rename_cols_to_eng(df)
    df.address=df.address.str.strip()
    df.desc=generate_na(df.desc, na_eqs=['|   |', '| |', ' '])
    df=translate_listing_type(df)
    return df

def multiply(func, **kwargs):
    def func_wrapper(from_string, thousand_eq=None, million_eq=None, billion_eq=None, thousand_mlpr = 1e3, million_mlpr = 1e6, billion_mlpr = 1e9):
        from_string = str(from_string)
        if billion_eq and billion_eq in from_string:
            multiplier = billion_mlpr
        elif million_eq and million_eq in from_string:
            multiplier = million_mlpr
        elif thousand_eq and thousand_eq in from_string:
            multiplier = thousand_mlpr
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

def calc_outlier_ratio(st_df, columns, n_sigma = 10, sigma_step = 0.25):
    total_records = len(st_df)
    ratios = []
    for sigma in np.arange(n_sigma, 1, -sigma_step):
        all_mask = np.array([True] * total_records)
        for c in columns:
            mask = (st_df[c].values <= sigma) & (st_df[c].values >= sigma * (-1))
            all_mask = all_mask & mask
        filtered_records = len(st_df[all_mask])
        ratio = 1 - (filtered_records/total_records)
        ratios.append(ratio*100)
    return np.c_[np.arange(n_sigma, 1, -sigma_step), np.array(ratios)]    
    
def create_outlier_mask(st_df, columns, sigma_threshold = 3):
    total_records = len(st_df)
    outlier_mask = np.array([True] * total_records)
    for c in columns:
        mask = (st_df[c].values <= sigma_threshold) & (st_df[c].values >= sigma_threshold * (-1))
        outlier_mask = outlier_mask & mask
    return outlier_mask
	
def get_address_mask(series, public_domains_fn, street_num = True):
    public_domains = prep.load_public_domain_names(public_domains_fn)
    ptrn = '|'.join(['.* {}$'.format(d) for d in public_domains])
    if street_num:
        ptrn_dot = '|'.join(['.*{}.*\d\.$'.format(d) for d in public_domains])
        ptrn_slash_num = '|'.join(['.*{}.*\d/[0-9]$'.format(d) for d in public_domains])
        ptrn_slash_letter = '|'.join(['.*{}.*\d/[A-Z]$'.format(d) for d in public_domains])
        ptrn_dash= '|'.join(['.*{}.*\d*-\d*$'.format(d) for d in public_domains])
        ptrn = '|'.join([ptrn_dot, ptrn_slash_num, ptrn_slash_letter, ptrn_dash])
    mask = series.apply(lambda a: bool(re.match(string=str(a), pattern=ptrn)))
    return mask