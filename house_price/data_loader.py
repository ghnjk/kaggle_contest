#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Author: ghnjk
# @Create: 2018-06-03
import os
import pandas as pd


class HousePriceData(object):

    def __init__(self, data_dir="data/download"):
        self.train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        self.test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
