from sklearn.pipeline import Pipeline
from src.processing import *
from src.utils import RealEstateData, load_elevation_data

OLD_TO_NEW = {'data_id': 'property_id', 'description': 'desc',
              'district': 'city_district', 'latitude': 'lat',
              'link': 'property_url', 'load_time': 'timestamp',
              'longitude': 'lng', 'max_list': 'max_listing',
              'price': 'price_in_huf', 'quadmeter': 'area_size',
              'rooms': 'room', 'type': 'listing_type'}
HUN_TO_ENG = {'akadálymentesített': 'accessibility', 'belmagasság': 'ceiling_height', 'busz': 'buses',
              'busz_count': 'buses_count', 'bútorozott': 'furnished', 'dohányzás': 'smoking', 'emelet': 'floors',
              'energiatanúsítvány': 'energy_perf_cert', 'erkély': 'balcony', 'fürdő_és_wc': 'bath_and_wc',
              'fűtés': 'type_of_heating', 'gépesített': 'equipped', 'hajó': 'boats', 'hajó_count': 'boats_count',
              'hév': 'local_railways', 'hév_count': 'local_railways_count',
              'ingatlan_állapota': 'condition_of_real_estate',
              'kertkapcsolatos': 'with_entry_to_garden', 'kilátás': 'view', 'kisállat': 'pets',
              'komfort': 'convenience_level',
              'költözhető': 'vacant', 'lakópark_neve': 'residental_park_name', 'légkondicionáló': 'air_conditioned',
              'metró': 'metro_lines', 'metró_count': 'metro_lines_count', 'min._bérleti_idő': 'min_tenancy',
              'parkolás': 'parking',
              'parkolóhely_ára': 'parking_lot_price', 'rezsiköltség': 'utilities', 'tetőtér': 'attic',
              'troli': 'trolley_buses', 'troli_count': 'trolley_buses_count', 'tájolás': 'orientation',
              'villamos': 'trams', 'villamos_count': 'trams_count', 'éjszakai': 'all_night_services',
              'éjszakai_count': 'all_night_services_count', 'építés_éve': 'year_built',
              'épület_szintjei': 'building_floors'}

LISTING_TYPE_HUN_TO_ENG = {'elado': 'for-sale', 'kiado': 'for-rent'}

COMPOSITE_ID = ['property_id', 'timestamp']
TECHNICAL_COLUMNS = ['property_url', 'cluster_id', 'batch_num', 'page_num', 'max_page', 'max_listing', 'is_ad_active']

INPUT_DIR = '../../real_estate_hungary/output/'
DATE = '20181101'

ELEVATION_PATH = '../data/ext/dummy_elevation.csv'
ELEVATION_MAP = dict(latitude='latitude', longitude='longitude')


def build_preprocess():
    pass


if __name__ == '__main__':
    real_estate_data = RealEstateData(data_dir=INPUT_DIR, file_name='raw.csv')
    raw = real_estate_data.read(dir_name='data', date=DATE)
    elevation_data = load_elevation_data(ELEVATION_PATH)
    raw_elevation_map = {k:v for k, v in OLD_TO_NEW.items() if k in ELEVATION_MAP.keys()}

    pipeline = Pipeline([('column_name_standardisation',
                          ColumnRenamer(old_to_new=OLD_TO_NEW, hun_to_eng=HUN_TO_ENG)),
                         ('listing_type_translation',
                          Translator(column_name='listing_type', hun_eng_map=LISTING_TYPE_HUN_TO_ENG)),
                         ('to_lower_case', StringStandardizer(func=lambda s: s.str.lower())),
                         ('drop_duplicates', DuplicatesRemoval(negation=True, columns=COMPOSITE_ID + TECHNICAL_COLUMNS)),
                         ('add_elevation',
                          ElevationMerger(left_longitude=raw_elevation_map['longitude'],
                                          left_latitude=raw_elevation_map['latitude'],
                                          elevation_data_path=ELEVATION_PATH,
                                          elevation_longitude=ELEVATION_MAP['longitude'],
                                          elevation_latitude=ELEVATION_MAP['latitude'],
                                          rounding_decimals=16, mode='w'))
                         ])
    pro = pipeline.transform(raw)
    print(raw.shape, pro.shape)
