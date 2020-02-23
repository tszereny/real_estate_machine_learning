import os
from src.base import SlicedPipeline, BASE_DIR
from src.utils import RealEstateData, load_stored_elevation
from src.pipelines import preprocessing_steps


INPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'real_estate_hungary', 'output')
DATE = '20181101'


if __name__ == '__main__':
    real_estate_data = RealEstateData(data_dir=INPUT_DIR, file_name='raw.csv')
    raw = real_estate_data.read(dir_name='data', date=DATE)
    preprocessing_pipeline = SlicedPipeline(stop_step=None, steps=preprocessing_steps)
    pro = preprocessing_pipeline.transform(raw)