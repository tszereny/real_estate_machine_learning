import pytest
from src.utils import RealEstateData
from src.processing import ColumnRenamer, Translator
from pipeline import OLD_TO_NEW, HUN_TO_ENG, LISTING_TYPE_HUN_TO_ENG
import pandas as pd
import os


@pytest.fixture(scope='module')
def real_estate_data():
    FIXTURE_DIR = '../fixtures/'
    DATA_FNAME = 'sample.csv'
    red = RealEstateData(FIXTURE_DIR, DATA_FNAME)
    return red


@pytest.fixture(scope='module')
def real_estate_raw(real_estate_data):

    def _real_estate_raw(date_idx):
        dates = real_estate_data.directories
        return real_estate_data.read(dir_name='raw', date=dates[date_idx])

    return _real_estate_raw

@pytest.fixture(scope='module', params=[0, 1])
def real_estate_renamed(real_estate_raw, request):
    cr = ColumnRenamer(old_to_new=OLD_TO_NEW, hun_to_eng=HUN_TO_ENG)
    df = cr.transform(real_estate_raw(request.param))
    return df

# class TestColumRenamer:
#
#     def test_transform(self, real_estate_data):
#
#         date_0 = cr.transform(dates[0]).columns.tolist()
#         date_1 = cr.transform(dates[1]).columns.tolist()
#         assert set(date_0) - set(date_1) == {'batch_num', 'is_ad_active'}
#
#
# class TestTranslator:
#
#     @pytest.mark.parametrize('idx', [0, 1])
#     def test_wrong_column(self, idx, real_estate_data):
#         date = real_estate_data.directories[idx]
#         cr = ColumnRenamer(old_to_new=OLD_TO_NEW, hun_to_eng=HUN_TO_ENG)
#         df = cr.transform(real_estate_data.read(dir_name='raw', date=date))
#         t = Translator(column_name='wrong_column', hun_eng_map={})
#         with pytest.raises(KeyError):
#             t.transform(df)
#         to_be_translated = 'listing_type'
#         t = Translator(column_name=to_be_translated, hun_eng_map=LISTING_TYPE_HUN_TO_ENG)
#         translated_listing_type = t.transform(df)[to_be_translated]
#         assert len(translated_listing_type.unique()) == 2
#         assert translated_listing_type.apply(lambda s: s not in ('elado', 'kiado')).all() == True

