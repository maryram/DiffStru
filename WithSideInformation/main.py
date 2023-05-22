#!/usr/bin/env python
# coding: utf-8

# In[31]:


"""
Tests for the BNMF Gibbs sampler.
"""

import sys, os
from _ast import In
from pathlib import Path
import Init
import utils
import sys
from distributions.algebra import sigmoid

import numpy, math, itertools # pytest,
import numpy as np
from DiffStru import bnmf_gibbs
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from distributions import algebra
from performance_metrics import print_metrics
import configparser
import argparse

from model import Model

param_file = 'params.ini'

if __name__ == '__main__':

    params = {}
    config = configparser.ConfigParser()
    config.read(param_file)
    params['iterations'] = config['general']['iterations']
    params['initial_burn_in'] = config['general']['initial_burn_in']
    params['zero_time'] = float(config['general']['zero_time'])
    params['compute_perf'] = bool(config['general']['compute_perf'])


    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--dataset_path', action='store', type=str, required=True)
    parser.add_argument('--burn', action='store', type=int)
    parser.add_argument('--thinning', action='store', type=int)
    parser.add_argument('-d', '--dim', action='store', type=int, required=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cascade', action='store_true')
    parser.add_argument("--e_threshold", action="store", type=float, default=0)
    args = parser.parse_args()
    
    model = Model(args, zero_time=params['zero_time'])
    if args.train:
        model.train(params)
    elif args.test:
        e_threshold = float(args.e_threshold)
        burn_in, thinning = args.burn, args.thinning
        model.test(burn_in, thinning, e_threshold)
    elif args.cascade:
        burn_in, thinning = args.burn, args.thinning
        model.print_cascades(burn_in, thinning)
    else:
        print("Specify --train or --test or --cascade")
    exit()
