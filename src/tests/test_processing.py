import pytest
from sklearn.pipeline import Pipeline
from src.processing import ColumnRenamer, Translator
from pipeline import OLD_TO_NEW, HUN_TO_ENG


class TestColumRenamer:

    def test_transform(self, real_estate_data):
        cr = ColumnRenamer(old_to_new=OLD_TO_NEW, hun_to_eng=HUN_TO_ENG)
        dates = [real_estate_data.read(dir_name='raw', date=d) for d in real_estate_data.directories]
        date_0 = cr.transform(dates[0]).columns.tolist()
        date_1 = cr.transform(dates[1]).columns.tolist()
        assert set(date_0) - set(date_1) == {'batch_num', 'is_ad_active'}


class TestTranslator:

    @pytest.mark.parametrize('idx', [0, 1])
    def test_wrong_column(self, idx, real_estate_data):
        date = real_estate_data.directories[idx]
        cr = ColumnRenamer(old_to_new=OLD_TO_NEW, hun_to_eng=HUN_TO_ENG)
        df = cr.transform(real_estate_data.read(dir_name='raw', date=date))
        t = Translator(column_name='wrong_column', hun_eng_map={})
        with pytest.raises(KeyError):
            t.transform(df)
