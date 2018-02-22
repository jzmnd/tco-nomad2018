#!/usr/bin/env python
# encoding: utf-8

"""
Functions for loading data from csv files
"""

import os
import numpy as np
import pandas as pd
import pymatgen as mg


def load_features(data_dir, with_ext=True, with_geo=True, expectO=0.6):
    """
    Function to load data from data_dir and output train and test dataframes.
    Also adds features:
        - cell volume
        - atomic and mass density
        - unit cell angles in radians
        - a combined spacegroup and natoms category
        - number and fraction of O atoms
    """
    # Load basic features
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'),
                        names=['id', 'spacegroup', 'natoms', 'al',
                               'ga', 'in', 'a', 'b', 'c',
                               'alpha', 'beta',
                               'gamma', 'E0',
                               'bandgap'],
                        header=0,
                        sep=',')
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'),
                       names=['id', 'spacegroup', 'natoms', 'al',
                              'ga', 'in', 'a', 'b', 'c',
                              'alpha', 'beta',
                              'gamma'],
                       header=0,
                       sep=',')

    # Load extra features from xyz files and element properties
    if with_ext:
        train_ext = pd.read_csv(os.path.join(data_dir, 'train_ext.csv'),
                                header=0,
                                sep=',')
        test_ext = pd.read_csv(os.path.join(data_dir, 'test_ext.csv'),
                               header=0,
                               sep=',')

        train = train.merge(train_ext, on='id')
        test = test.merge(test_ext, on='id')

    # Load geometry features from xyz files processed from crystal graph
    if with_geo:
        train_geo = pd.read_csv(os.path.join(data_dir, 'train_geo.csv'),
                                header=0,
                                sep=',')
        test_geo = pd.read_csv(os.path.join(data_dir, 'test_geo.csv'),
                               header=0,
                               sep=',')

        train = train.merge(train_geo, on='id')
        test = test.merge(test_geo, on='id')

    # Add the spacegroup_natoms category
    train['spacegroup_natoms'] = train['spacegroup'].astype(str) +\
        '_' + train['natoms'].astype(int).astype(str)
    test['spacegroup_natoms'] = test['spacegroup'].astype(str) +\
        '_' + test['natoms'].astype(int).astype(str)

    # Add the cell volume and calculate atomic and mass densities
    train['cellvol'] = train.apply(lambda r: mg.Lattice.from_parameters(
        r['a'], r['b'], r['c'], r['alpha'], r['beta'], r['gamma']).volume,
                                   axis=1)
    test['cellvol'] = test.apply(lambda r: mg.Lattice.from_parameters(
        r['a'], r['b'], r['c'], r['alpha'], r['beta'], r['gamma']).volume,
                                 axis=1)

    train['atom_density'] = train['natoms'] / train['cellvol']
    test['atom_density'] = test['natoms'] / test['cellvol']

    if with_ext or with_geo:
        train['mass_density'] = train['avg_mass'] / train['cellvol']
        test['mass_density'] = test['avg_mass'] / test['cellvol']

    # Convert angles to radians
    train['alpha_r'] = np.radians(train['alpha'])
    train['beta_r'] = np.radians(train['beta'])
    train['gamma_r'] = np.radians(train['gamma'])
    test['alpha_r'] = np.radians(test['alpha'])
    test['beta_r'] = np.radians(test['beta'])
    test['gamma_r'] = np.radians(test['gamma'])

    if with_ext or with_geo:
        # Check O fraction is the expected value for all
        train['o_fraction'] = train['o_cnt'] / train['natoms']
        test['o_fraction'] = test['o_cnt'] / test['natoms']
        assert (train['o_fraction'] == expectO).all()
        assert (test['o_fraction'] == expectO).all()

    return train, test
