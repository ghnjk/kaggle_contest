#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Author: ghnjk
# @File: preview_data.py
# @Create: 2018-06-03
from data_loader import HousePriceData


def main():
    data = HousePriceData()
    print(data.train_df.info())
    print(data.test_df.info())


if __name__ == '__main__':
    main()
