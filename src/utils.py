import numpy as np
import json
import smtplib, ssl, getpass
import pickle

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
        
def calc_intervals(ints_n, length):
    r=[]
    strt=0
    for i in range(ints_n):
        if i==ints_n-1:
            stp=strt+(length-stp)
            r.append(range(strt, stp))
        else:
            stp=strt+int(np.ceil(length/ints_n))
            r.append(range(strt, stp))
            strt=stp
    return r