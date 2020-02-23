import os
from src.base import BASE_DIR
from src.processing import *

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

ELEVATION_PATH = os.path.join(BASE_DIR, 'data', 'ext', 'elevation.csv')
ELEVATION_MAP = dict(latitude='latitude', longitude='longitude')
raw_elevation_map = {k: v for k, v in OLD_TO_NEW.items() if k in ELEVATION_MAP.keys()}

preprocessing_steps = [('column_name_standardisation',
                        ColumnRenamer(old_to_new=OLD_TO_NEW, hun_to_eng=HUN_TO_ENG)),
                       ('listing_type_translation',
                   Translator(column_name='listing_type', hun_eng_map=LISTING_TYPE_HUN_TO_ENG)),
                       ('to_lower_case', StringStandardizer(func=lambda s: s.str.lower())),
                       ('property_id_to_number', FunctionApplier(function=lambda x: extract_num(x), columns=['property_id'],
                                                            new_columns=['property_id'])),
                       ('cluster_id_to_number',
                        FunctionApplier(function=lambda x: extract_num(x), columns=['cluster_id'],
                                        new_columns=['cluster_id'])),
                       ('basic_ids_to_int',
                        ColumnCaster(columns=['property_id', 'cluster_id'], dtypes=['int64', 'int64'])),
                       ('drop_duplicates',
                   DuplicatesRemoval(are_columns_negated=True, columns=COMPOSITE_ID + TECHNICAL_COLUMNS)),
                       ('new_records_to_elevation', ElevationInserter(left_longitude=raw_elevation_map['longitude'],
                                                                 left_latitude=raw_elevation_map['latitude'],
                                                                 stored_elevation_path=ELEVATION_PATH,
                                                                 stored_elevation_longitude=ELEVATION_MAP[
                                                                     'longitude'],
                                                                 stored_elevation_latitude=ELEVATION_MAP[
                                                                     'latitude'],
                                                                 rounding_decimals=6, mode='w')),
                       ('merge_elevation',
                   ElevationMerger(left_longitude=raw_elevation_map['longitude'],
                                   left_latitude=raw_elevation_map['latitude'],
                                   stored_elevation_path=ELEVATION_PATH,
                                   stored_elevation_longitude=ELEVATION_MAP['longitude'],
                                   stored_elevation_latitude=ELEVATION_MAP['latitude'],
                                   rounding_decimals=6)),
                       ('drop_duplicated_columns',
                   DropColumns(columns=[ELEVATION_MAP['longitude'], ELEVATION_MAP['latitude']])),
                       ('create_id',
                   IdCreator(columns=COMPOSITE_ID + list(raw_elevation_map.values()), date_format='%Y-%m-%d %H:%M:%S.%f',
                             fallback_date_format='%Y-%m-%d %H:%M:%S',
                             id_column_name='id')),
                       ('price_to_number', FunctionApplier(
                      function=lambda x: extract_num(from_string=x, thousand_eq='ezer', million_eq='millió',
                                                     billion_eq='milliárd'), columns=['price_in_huf'],
                      new_columns=['price_in_huf'])),
                       ('area_to_number',
                   FunctionApplier(function=lambda x: extract_num(from_string=x), columns=['area_size'],
                                   new_columns=['area_size'])),
                       ('room_to_lt_12_sqm',
                   FunctionApplier(function=lambda x: extract_num(x.split('+')[1]) if '+' in x else 0,
                                   columns=['room'], new_columns=['room_lt_12_sqm'])),
                       ('room_to_ge_12_sqm',
                   FunctionApplier(function=lambda x: extract_num(x.split('+')[0]), columns=['room'],
                                   new_columns=['room_ge_12_sqm'])),
                       ('room_total',
                   ColumnAdder(left_columns=['room_ge_12_sqm'], right_columns=['room_lt_12_sqm'],
                               new_columns=['room_total'])),
                       ('rooms_to_int', ColumnCaster(columns=['room_ge_12_sqm', 'room_ge_12_sqm', 'room_total'],
                                                     dtypes=['int64', 'int64', 'int64'])),
                       ('balcony_to_number',
                   FunctionApplier(function=lambda x: extract_num(from_string=x), columns=['balcony'],
                                   new_columns=['balcony'])),
                       ('parking_lot_in_huf',
                   ParkingFunctionApplier(function=lambda x: extract_num(x, million_eq='m'), monthly_alias='/hó',
                                          euro_alias='€', monthly_flag=False, euro_flag=False,
                                          column='parking_lot_price', new_column='parking_lot_in_huf')),
                       ('parking_lot_in_eur',
                   ParkingFunctionApplier(function=lambda x: extract_num(x), monthly_alias='/hó',
                                          euro_alias='€', monthly_flag=False, euro_flag=True,
                                          column='parking_lot_price', new_column='parking_lot_in_eur')),
                       ('parking_lot_in_huf_monthly',
                   ParkingFunctionApplier(function=lambda x: extract_num(x), monthly_alias='/hó',
                                          euro_alias='€', monthly_flag=True, euro_flag=False,
                                          column='parking_lot_price', new_column='parking_lot_in_huf_monthly')),
                       ('parking_lot_in_eur_monthly',
                   ParkingFunctionApplier(function=lambda x: extract_num(x), monthly_alias='/hó',
                                          euro_alias='€', monthly_flag=True, euro_flag=True,
                                          column='parking_lot_price', new_column='parking_lot_in_eur_monthly')),
                       ('utilities_to_number',
                   FunctionApplier(function=lambda x: extract_num(from_string=x), columns=['utilities'],
                                   new_columns=['utilities'])),
                       # TODO: Issue with utilities that, people are too lazy to type 25000 instead they type only 25.0
                       ('mininum_tenancy_to_number',
                   MinTenancyFunctionApplier(column='min_tenancy', new_column='min_tenancy', year_alias='év',
                                             monthly_alias='hónap', no_alias='nincs')),
                       # TODO: As a next step, data type of the columns should be checked and if it is not right, fix it
                       ('add_price_per_sqm', ColumnDivider(left_columns=['price_in_huf'], right_columns=['area_size'],
                                                      new_columns=['price_per_sqm'])),
                       ('drop_original_columns', DropColumns(columns=['room', 'parking_lot_price'])),
                       ]


def build_preprocess():
    pass
