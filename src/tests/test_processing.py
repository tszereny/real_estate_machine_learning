import pytest
import os
from sklearn.pipeline import Pipeline
from src.processing import ColumnRenamer, Translator, StringStandardizer, ElevationMerger
from pipeline import OLD_TO_NEW, HUN_TO_ENG, LISTING_TYPE_HUN_TO_ENG


class TestStringStandardizer:

    def test_transform(self, real_estate_raw):
        df = real_estate_raw(0)
        ss = StringStandardizer(func=lambda s: None)
        assert ss.transform(df)[ss.column_names].isnull().all().all() == True

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


class TestElevationMerger:

    @pytest.mark.skipif(False, reason='slow test')
    def test_transform(self, sample_gps_data):
        dummy_elevation_path = './dummy_elevation.csv'
        em = ElevationMerger(left_latitude='lat', left_longitude='lng', elevation_data_path=dummy_elevation_path,
                             elevation_latitude='latitude', elevation_longitude='longitude')
        res = em.transform(sample_gps_data[-3:])
        if os.path.exists(dummy_elevation_path):
            os.remove(dummy_elevation_path)
        assert len(res) == 3
        assert res['elevation'].isin([130, 200, 145]).all()
