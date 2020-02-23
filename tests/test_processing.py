import pytest
import os
from copy import deepcopy
from src.base import SlicedPipeline
from src.processing import *
from src.pipelines import preprocessing_steps, ELEVATION_PATH, ELEVATION_MAP, OLD_TO_NEW, HUN_TO_ENG, LISTING_TYPE_HUN_TO_ENG, \
    COMPOSITE_ID, raw_elevation_map


@pytest.mark.parametrize('s, expected', [('4.3 milliárd', 4.3e9), ('33.5 millió', 33.5e6), ('33.5 million', 33.5),
                                         ('180 ezer', 180e3), ('150 000', 150000), (None, None), (np.nan, np.nan)])
def test_extract_num(s, expected):
    assert extract_num(from_string=s, thousand_eq='ezer', million_eq='millió',
                       billion_eq='milliárd') == expected or np.isnan(s)

class TestStringStandardizer:

    def fill_with_none(self, s):
        return s.apply(lambda x: None)

    def test_transform(self, real_estate_renamed):
        assert isinstance(real_estate_renamed, pd.DataFrame)
        ss = StringStandardizer(func=lambda s: self.fill_with_none(s))
        assert isinstance(ss.get_string_columns(real_estate_renamed), list)
        result = ss.transform(real_estate_renamed)
        assert isinstance(result, pd.DataFrame)
        assert result[ss.column_names].isnull().all().all() == True


class TestColumRenamer:

    def test_transform(self, real_estate_raw):
        cr = ColumnRenamer(old_to_new=OLD_TO_NEW, hun_to_eng=HUN_TO_ENG)
        date_0 = cr.transform(real_estate_raw(0)).columns.tolist()
        date_1 = cr.transform(real_estate_raw(1)).columns.tolist()
        assert set(date_0) - set(date_1) == {'batch_num', 'is_ad_active'}


class TestTranslator:

    def test_wrong_column(self, real_estate_renamed):
        t = Translator(column_name='wrong_column', hun_eng_map={})
        with pytest.raises(KeyError):
            t.transform(real_estate_renamed)

    def test_listing_type(self, real_estate_renamed):
        to_be_translated = 'listing_type'
        t = Translator(column_name=to_be_translated, hun_eng_map=LISTING_TYPE_HUN_TO_ENG)
        translated_listing_type = t.transform(real_estate_renamed)[to_be_translated]
        assert len(translated_listing_type.unique()) == 2
        assert translated_listing_type.apply(lambda s: s not in ('elado', 'kiado')).all() == True


class TestElevationTransformers:

    def test_retrieving(self, sample_gps_data):
        dummy_elevation_path = 'tests/fixtures/not_existing_elevation_data.csv'
        em = ElevationInserter(left_latitude='lat', left_longitude='lng', stored_elevation_path=dummy_elevation_path,
                             stored_elevation_latitude='latitude', stored_elevation_longitude='longitude')
        em.transform(sample_gps_data[-3:])
        stored_elevation = pd.read_csv(dummy_elevation_path)
        stored_elevation['elevation'] = stored_elevation['elevation'].apply(int)
        assert len(stored_elevation) == 3
        assert stored_elevation['elevation'].isin([131, 207, 148]).all()
        if os.path.exists(dummy_elevation_path):
            os.remove(dummy_elevation_path)

    def test_merging(self, sample_gps_data):
        dummy_elevation_path = 'tests/fixtures/dummy_elevation.csv'
        em = ElevationMerger(left_latitude='lat', left_longitude='lng', stored_elevation_path=dummy_elevation_path,
                             stored_elevation_latitude='latitude', stored_elevation_longitude='longitude',
                             rounding_decimals=6)
        res = em.transform(sample_gps_data[-3:])
        assert len(res) == 3
        assert res['elevation'].isin([130, 200, 145]).all()

    def test_transform(self, real_estate_renamed):
        sp = SlicedPipeline(steps=deepcopy(preprocessing_steps), stop_step='merge_elevation')
        preprocessed_data = sp.transform(real_estate_renamed)
        assert not 'elevation' in real_estate_renamed.columns
        em = ElevationMerger(left_longitude=raw_elevation_map['longitude'],
                        left_latitude=raw_elevation_map['latitude'],
                        stored_elevation_path=ELEVATION_PATH,
                        stored_elevation_longitude=ELEVATION_MAP['longitude'],
                        stored_elevation_latitude=ELEVATION_MAP['latitude'],
                        rounding_decimals=6)
        emr = em.transform(preprocessed_data)
        assert 'elevation' in emr.columns
        assert emr['elevation'].isnull().sum() == 0


class TestFunctionApplier:

    def test_property_id_to_number(self, real_estate_renamed):
        sp = SlicedPipeline(steps=deepcopy(preprocessing_steps), stop_step='to_lower_case')
        preprocessed_data = sp.transform(real_estate_renamed)
        fa = FunctionApplier(function=lambda x: extract_num(x), columns=['property_id'],
                        new_columns=['property_id_extracted'])
        far = fa.transform(preprocessed_data)
        assert far['property_id_extracted'].dtype in ('float64', 'int64')
        assert far['property_id_extracted'].isnull().any() == False

    def test_price_to_number(self, real_estate_renamed):
        fa = FunctionApplier(function=lambda x: extract_num(from_string=x, thousand_eq='ezer', million_eq='millió',
                                                       billion_eq='milliárd'), columns=['price_in_huf'],
                        new_columns=['price_in_huf_extracted'])
        far = fa.transform(real_estate_renamed)
        assert far['price_in_huf_extracted'].dtype == 'float64'
        assert far['price_in_huf_extracted'].isnull().any() == False
        assert far['price_in_huf_extracted'].min() / 1e6 <= 1
        assert far['price_in_huf_extracted'].max() / 1e11 <= 1

    def test_area_to_number(self, real_estate_renamed):
        fa = FunctionApplier(function=lambda x: extract_num(from_string=x), columns=['area_size'],
                        new_columns=['area_size_extracted'])
        far = fa.transform(real_estate_renamed)
        assert far['area_size_extracted'].dtype == 'float64'
        assert far['area_size_extracted'].isnull().any() == False

    def test_room_to_lt_12_sqm(self, real_estate_renamed):
        fa = FunctionApplier(function=lambda x: extract_num(x.split('+')[1]) if '+' in x else 0, columns=['room'],
                        new_columns=['room_lt_12_sqm'])
        far = fa.transform(real_estate_renamed)
        assert far['room_lt_12_sqm'].dtype == 'float64'
        assert far['room_lt_12_sqm'].isnull().any() == False

    def test_room_to_ge_12_sqm(self, real_estate_renamed):
        fa = FunctionApplier(function=lambda x: extract_num(x.split('+')[0]), columns=['room'],
                        new_columns=['room_ge_12_sqm'])
        far = fa.transform(real_estate_renamed)
        assert far['room_ge_12_sqm'].dtype == 'float64'
        assert far['room_ge_12_sqm'].isnull().any() == False

    def test_balcony_to_number(self, real_estate_renamed):
        fa = FunctionApplier(function=lambda x: extract_num(x), columns=['balcony'],
                        new_columns=['balcony_extracted'])
        far = fa.transform(real_estate_renamed)
        assert far['balcony_extracted'].dtype == 'float64'
        assert far['balcony_extracted'].isnull().any() == True, 'Balcony contains Nan values'

    def test_utilities_to_number(self, real_estate_renamed):
        fa = FunctionApplier(function=lambda x: extract_num(x), columns=['utilities'],
                        new_columns=['utilities_extracted'])
        far = fa.transform(real_estate_renamed)
        assert far['utilities_extracted'].dtype == 'float64'
        assert far['utilities_extracted'].isnull().any() == True, 'Utilities can contain Nan values'
        assert np.sum(far.loc[far['utilities_extracted'].notnull(), 'utilities_extracted'] > 1000) > 100
        assert np.all(far.loc[far['utilities_extracted'].notnull(), 'utilities_extracted'] >= 25), 'Somebody was lazy to type 25000 Huf'


class TestColumnCaster:

    def test_transform(self, real_estate_renamed):
        sp = SlicedPipeline(steps=deepcopy(preprocessing_steps), stop_step='drop_duplicates')
        preprocessed_data = sp.transform(real_estate_renamed)
        for col in ['property_id', 'cluster_id']:
            assert preprocessed_data[col].dtype == 'int64'

class TestColumnsAdder:

    def test_total_rooms(self, real_estate_renamed):
        sp = SlicedPipeline(steps=deepcopy(preprocessing_steps), stop_step='room_total')
        assert sp.named_steps['to_lower_case'].column_names == None
        preprocessed_data = sp.transform(real_estate_renamed)
        ca = ColumnAdder(left_columns=['room_ge_12_sqm'], right_columns=['room_lt_12_sqm'], new_columns=['room_total'])
        car = ca.transform(preprocessed_data)
        assert car['room_total'].dtype == 'float64'
        assert car['room_total'].isnull().any() == False
        assert np.all(car['room_total'] > 0)


class TestIdCreator:

    def test_transform(self, real_estate_renamed):
        sp = SlicedPipeline(steps=deepcopy(preprocessing_steps), stop_step='create_id')
        preprocessed_data = sp.transform(real_estate_renamed)
        ic = IdCreator(columns=COMPOSITE_ID + list(raw_elevation_map.values()), date_format='%Y-%m-%d %H:%M:%S.%f',
                  fallback_date_format='%Y-%m-%d %H:%M:%S',
                  id_column_name='id')
        icr = ic.transform(preprocessed_data)
        assert icr['id'].dtype == 'float64'
        assert len(icr['id'].tolist()) == len(icr['id'].unique())


class TestParkingFunctionApplier:

    def test_transform(self, real_estate_renamed):
        sp = SlicedPipeline(steps=deepcopy(preprocessing_steps), stop_step='utilities_to_number')
        spr = sp.transform(real_estate_renamed)
        assert {'parking_lot_in_huf', 'parking_lot_in_eur', 'parking_lot_in_huf_monthly',
                    'parking_lot_in_eur_monthly'} - set(spr.columns.tolist()) == set()
        assert np.all(spr.loc[spr['parking_lot_in_huf'].notnull(), 'parking_lot_in_huf'] > 1e5)
        assert np.all(spr.loc[spr['parking_lot_in_huf_monthly'].notnull(), 'parking_lot_in_huf_monthly'] > 1e3)
        assert spr.loc[spr['parking_lot_price'].str.contains('m').fillna(False),
                       ['parking_lot_in_huf', 'parking_lot_in_huf_monthly']].notnull().any(axis=1).all()
        assert spr.loc[spr['parking_lot_price'].str.contains('€').fillna(False),
                       ['parking_lot_in_eur', 'parking_lot_in_eur_monthly']].notnull().any(axis=1).all()


class TestMinTenancyFunctionApplier:

    def test_transform(self, real_estate_renamed):
        sp = SlicedPipeline(steps=deepcopy(preprocessing_steps), stop_step='add_price_per_sqm')
        spr = sp.transform(real_estate_renamed)
        assert spr['min_tenancy'].dtype == 'float64'
        assert np.all(spr.loc[spr['min_tenancy'].notnull(), 'min_tenancy'] >= 0)


class TestColumnDivider:

    def test_transform(self, real_estate_renamed):
        sp = SlicedPipeline(steps=deepcopy(preprocessing_steps), stop_step='drop_original_columns')
        spr = sp.transform(real_estate_renamed)
        assert 'price_per_sqm' in spr.columns
        assert np.all(spr['price_per_sqm'] >= 0)
        assert np.all(spr['price_per_sqm'] == spr['price_in_huf'] / spr['area_size'])


class TestPipeline:

    def test_transform(self, real_estate_renamed):
        sp = SlicedPipeline(steps=deepcopy(preprocessing_steps), stop_step=None)
        spr = sp.transform(real_estate_renamed)
        interval_or_ratio = ['lat', 'lng',
                             'elevation', 'price_in_huf', 'price_per_sqm',
                             'area_size', 'room_total',
                             'balcony', 'parking_lot_in_huf',
                             'parking_lot_in_eur', 'parking_lot_in_huf_monthly',
                             'parking_lot_in_eur_monthly',
                             'utilities', 'min_tenancy',
                             'metro_lines_count', 'trams_count',
                             'trolley_buses_count', 'buses_count',
                             'boats_count', 'local_railways_count',
                             'all_night_services_count']
        for c in interval_or_ratio:
            assert pd.api.types.is_numeric_dtype(spr[c])
