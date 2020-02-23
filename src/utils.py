import logging
import numpy as np, pandas as pd
import json
import smtplib, ssl, getpass
import pickle
import os
from pandas import read_csv


def load_stored_elevation(file_path: str):
    if os.path.exists(file_path):
        return read_csv(file_path, float_precision='%.6f')
    logging.warning('file not exist {}, empty Dataframe initialized for elevation dataset'.format(file_path))
    return pd.DataFrame({'elevation': [], 'longitude': [], 'latitude': []})


def store_elevation(elevation: pd.DataFrame, file_path: str):
    elevation.to_csv(file_path, index=False, float_format='%.6f')


class RealEstateData:
    NA_EQUIVALENTS = ['nincs megadva', '|   |', '| |', ' ']

    def __init__(self, data_dir, file_name):
        self.data_dir = data_dir
        self.file_name = file_name

    @property
    def directories(self):
        return [_ for _ in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, _))]

    def list_files(self, dir_name):
        p = os.path.join(self.data_dir, dir_name)
        return os.listdir(p)

    def read(self, dir_name, date):
        p = os.path.join(self.data_dir, date, dir_name, self.file_name)
        df = read_csv(p, encoding='utf8', sep=',', na_values=self.NA_EQUIVALENTS)
        return df


class Email:
    
    def __init__(self, sender_email, receiver_email, port = 465, smtp_server = 'smtp.gmail.com'):
        self.sender_email = sender_email
        self.receiver_email = receiver_email
        self.port = port
        self.smtp_server = smtp_server
        self.context = ssl.create_default_context()
        self._password = getpass.getpass('Type your password and press enter: ')
        
    def send(self, msg):
        with smtplib.SMTP_SSL(self.smtp_server, self.port, context=self.context) as server:
            server.login(self.sender_email, self._password)
            server.sendmail(self.sender_email, self.receiver_email, msg)


def load_json(json_path, encoding='utf8'):
    with open(json_path, encoding=encoding) as f:
        data=json.load(f)
    return data


def read_txt(txt_path, encoding='utf8'):
    with open(txt_path, 'r', encoding=encoding) as  f:
        txt=f.read()
    return txt


def save_txt(txt_path, text):
    with open(txt_path, 'w') as f:
        f.write(text)


def save_pkl(pkl_path, obj):
    with open(pkl_path, 'wb') as f:  
        pickle.dump(obj, f)
    return pkl_path


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:  
        obj = pickle.load(f)
    return obj


def generate_intervals(ints_n, length):
    r = []
    strt = 0
    stp = 0
    for i in range(ints_n):
        if i == ints_n - 1:
            stp = strt + (length - stp)
            r.append(range(strt, stp))
        else:
            stp = strt + int(np.ceil(length / ints_n))
            r.append(range(strt, stp))
            strt = stp
    return r


if __name__ == '__main__':
    # DIR = '../../real_estate_hungary/output/'
    # data = RealEstateData(data_dir=DIR, file_name='raw.csv')
    # df = data.read(dir_name='data', date='20180301')
    ELEVATION_PATH = './data/ext/elevation.csv'
    df = load_stored_elevation(ELEVATION_PATH)

