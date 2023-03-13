import math
import os
import uuid
import time

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from IPython.display import HTML
import itertools
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2
from waymo_open_dataset.protos import scenario_pb2


def main():
    FILENAME = '/media/PJLAB\maosong/data/scenario/training/training.tfrecord-00000-of-01000'

    # read data
    dataset = tf.data.TFRecordDataset(FILENAME)
    scenario_data = []

    dataset1 = list(dict())
    for idx, data in enumerate(dataset):
        proto_string = data.numpy()
        proto = scenario_pb2.Scenario()
        proto.ParseFromString(proto_string)
        scenario_data.append(proto)

        data1 = {}
        data1['scenario_id'] = proto.scenario_id
        data1['map_feature'] = proto.map_features
        data1['tracks_to_predict'] = proto.tracks_to_predict
        data1['current_time_index'] = proto.current_time_index
        


    print(scenario_data[0].current_time_index)


if __name__ == '__main__':
    main()
