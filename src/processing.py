import pandas as pd

def generate_na(df, na_eq):
    gen_na=lambda x: None if x==na_eq else x
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
    'parkolóhely_ára': 'parking_lot_price_in_huf','rezsiköltség':'utilities','tetőtér':'attic',
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
    df.desc=generate_na(df.desc, na_eq='|   |')
    df=translate_listing_type(df)
    return df