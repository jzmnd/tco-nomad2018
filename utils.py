#!/usr/bin/env python
# encoding: utf-8

"""
Utility functions
"""
import os
import numpy as np
import pandas as pd
import pymatgen as mg


ELEMENTS = {'Al': mg.Element('Al'),
            'Ga': mg.Element('Ga'),
            'In': mg.Element('In'),
            'O': mg.Element('O')}


def get_size(ser, idx):
    """
    Gets value from pd.Series but returns 0 if not found
    """
    try:
        s = ser[idx]
    except KeyError:
        s = 0
    return s


def get_element_properties(data_dir):
    """
    Function to get the element properties from their .csv files (electron
    affinity, ionization potential, max radius of s, p and d orbitals)

    Inputs:
        data_dir - location of data directory containing element_properties
    Outputs:
        df_merged - dataframe of element properties
    """
    elem_dir = 'element_properties'

    ea = pd.read_csv(os.path.join(data_dir, elem_dir, 'EA.csv'),
                     header=None, names=['element', 'ea'])
    ip = pd.read_csv(os.path.join(data_dir, elem_dir, 'IP.csv'),
                     header=None, names=['element', 'ip'])
    rs = pd.read_csv(os.path.join(data_dir, elem_dir, 'rs_max.csv'),
                     header=None, names=['element', 'rs_max'])
    rp = pd.read_csv(os.path.join(data_dir, elem_dir, 'rp_max.csv'),
                     header=None, names=['element', 'rp_max'])
    rd = pd.read_csv(os.path.join(data_dir, elem_dir, 'rd_max.csv'),
                     header=None, names=['element', 'rd_max'])

    dflist = [ea, ip, rs, rp, rd]
    df_merged = reduce(lambda l, r: pd.merge(l, r,
                                             on=['element'],
                                             how='outer'), dflist)
    return df_merged


def rmsle(h, y):
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y

    Inputs:
        h - numpy array containing predictions
        y - numpy array containing targets
    """
    return np.sqrt(np.square(np.log1p(h) - np.log1p(y)).mean())


def get_xyz_data(filename):
    """
    Gets the XYZ Cartesian coordinates from file and extracts relevant
    information

    Inputs:
        filename - name of .xyz file to open
    Outputs:
        pos_data_df - dataframe of element positions
        lat_data    - lattice vector array
        natoms      - dataframe of element counts
        avg_elec    - average electronegativity
        avg_mass    - average atomic mass
    """
    pos_data = []
    lat_data = []

    with open(filename) as f:
        for line in f.readlines():
            x = line.strip().split()
            if x[0] == 'atom':
                pos_data.append([float(x[1]),
                                 float(x[2]),
                                 float(x[3]),
                                 x[4],
                                 ELEMENTS[x[4]].X,
                                 ELEMENTS[x[4]].atomic_mass,
                                 ELEMENTS[x[4]].average_ionic_radius])
            elif x[0] == 'lattice_vector':
                lat_data.append(np.array(x[1:4], dtype=np.float))

    pos_data_df = pd.DataFrame(pos_data,
                               columns=['x', 'y', 'z', 'element',
                                        'electroneg', 'atomic_mass',
                                        'r_ionic'])

    natoms = pos_data_df.groupby('element').size()
    avg_elec = pos_data_df['electroneg'].mean()
    avg_mass = pos_data_df['atomic_mass'].mean()

    return pos_data_df, np.array(lat_data), natoms, avg_elec, avg_mass
