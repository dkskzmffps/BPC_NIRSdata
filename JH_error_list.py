# -*- coding: utf-8 -*-
# from __future__ import print_function


class Error_dataset_name(Exception):
    def __init__(self):
        super().__init__('Please check dataset name. It should be NIH_Covid')

class Error_nor(Exception):
    def __init__(self):
        super().__init__('Please check normalize state')

class Error_net_name(Exception):
    def __init__(self):
        super().__init__('Please check network name')


class Error_mometum_weight(Exception):
    def __init__(self):
        super().__init__('Please check network name for set momentum weight')