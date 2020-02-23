import pytest
import os
from copy import deepcopy
from sklearn.pipeline import Pipeline
import pandas as pd
from src.base import SlicedPipeline
from src.processing import *
from pipeline import pipeline_steps, OLD_TO_NEW, HUN_TO_ENG, LISTING_TYPE_HUN_TO_ENG


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


class TestFunctionApplier:

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


class TestColumnsAdder:

    def test_total_rooms(self, real_estate_renamed):
        sp = SlicedPipeline(steps=deepcopy(pipeline_steps), stop_step='sum_of_rooms')
        assert sp.named_steps['to_lower_case'].column_names == None
        preprocessed_data = sp.transform(real_estate_renamed)
        ca = ColumnAdder(left_columns=['room_ge_12_sqm'], right_columns=['room_lt_12_sqm'], new_columns=['room_total'])
        car = ca.transform(preprocessed_data)
        assert car['room_total'].dtype == 'float64'
        assert car['room_total'].isnull().any() == False
