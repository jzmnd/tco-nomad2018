#!/usr/bin/env python
# encoding: utf-8

"""
1. Function for retrieving element properties.

2. Extension of the Element class in pymatgen to include more properties.
   Note: cannot inherit from Enum therefore building new class.
"""
import os
import pandas as pd
import pymatgen as mg


# Default data directory (location of element_properties)
DATA_DIR = './data'


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


class ElementExtended():
    """
    An extension of the pymatgen Element class including the following
    additional properties:

        ip     - ionization potential
        ea     - electron affinity
        rs_max - max radius of s orbitals
        rp_max - max radius of p orbitals
        rd_max - max radius of d orbitals

    """
    def __init__(self, symbol, data_dir=DATA_DIR):
        self.symbol = symbol

        self.element = mg.Element(symbol)

        self.Z = self.element.Z
        self.X = self.element.X
        self.atomic_mass = self.element.atomic_mass
        self.average_ionic_radius = self.element.average_ionic_radius

        data = get_element_properties(data_dir).set_index('element')

        try:
            self.ip = data.loc[symbol]['ip']
            self.ea = data.loc[symbol]['ea']
            self.rs_max = data.loc[symbol]['rs_max']
            self.rp_max = data.loc[symbol]['rp_max']
            self.rd_max = data.loc[symbol]['rd_max']
        except KeyError:
            self.ip = None
            self.ea = None
            self.rs_max = None
            self.rp_max = None
            self.rd_max = None

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return "Element {}".format(self.symbol)

    def __hash__(self):
        return self.Z
