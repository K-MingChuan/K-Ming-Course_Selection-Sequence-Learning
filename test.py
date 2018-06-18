import json
from data_preprocessing import *
import numpy

if __name__ == '__main__':
    with open('lv1_training.json', 'r', encoding='utf-8') as fr:
        raw = json.load(fr)
        sequences = raw['data']
        labels = raw['labels']
        print(len(sequences), len(labels))
