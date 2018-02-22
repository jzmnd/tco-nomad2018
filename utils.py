#!/usr/bin/env python
# encoding: utf-8

"""
Utility functions
"""
import numpy as np
import pandas as pd
import networkx as nx

from numpy.linalg import inv
from numpy.linalg import norm

from properties import ElementExtended

# ELement properties from the pymatgen database
ELEMENTS = {'Al': ElementExtended('Al'),
            'Ga': ElementExtended('Ga'),
            'In': ElementExtended('In'),
            'O': ElementExtended('O')}

# Create grid of possible (-1, 0, +1) lmn combinations
pm = np.arange(-1., 2.)
LMN_GRID = np.array(np.meshgrid(pm, pm, pm)).T.reshape(-1, 3)


def get_size(ser, idx):
    """
    Gets indexed value from pd.Series but returns 0 if index not found.
    """
    try:
        s = ser[idx]
    except KeyError:
        s = 0
    return s


def rmsle(h, y):
    """
    Compute the Root Mean Squared Log Error for hypothesis h and targets y.

    Inputs:
        h - numpy array containing predictions
        y - numpy array containing targets
    """
    return np.sqrt(np.square(np.log1p(h) - np.log1p(y)).mean())


def get_xyz_data(filename):
    """
    Gets the XYZ Cartesian coordinates from file and extracts relevant
    information.

    Inputs:
        filename - name of .xyz file to open
    Outputs:
        pos_data_df - dataframe of element positions
        lat_data    - lattice vector array
        natoms      - dataframe of element counts
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
                                 ELEMENTS[x[4]].average_ionic_radius,
                                 ELEMENTS[x[4]].ea,
                                 ELEMENTS[x[4]].ip,
                                 ELEMENTS[x[4]].rs_max,
                                 ELEMENTS[x[4]].rp_max,
                                 ELEMENTS[x[4]].rd_max])
            elif x[0] == 'lattice_vector':
                lat_data.append(np.array(x[1:4], dtype=np.float))

    pos_data_df = pd.DataFrame(pos_data,
                               columns=['x', 'y', 'z', 'element',
                                        'electroneg', 'atomic_mass',
                                        'r_ionic', 'ea', 'ip',
                                        'rs_max', 'rp_max', 'rd_max'])

    natoms = pos_data_df.groupby('element').size()

    return pos_data_df, np.array(lat_data), natoms


def convert_to_red(R, lat, tol=1e-9):
    """
    Converts atomic coordinates to reduced coordinate system.
    Rounds values less than tol to zero.
    """
    B = inv(lat.T)
    R_red = np.matmul(B, R.T).T
    R_red[abs(R_red) < tol] = 0
    return R_red


def get_shortest_distances(R_red, lat):
    """
    Gets the shortest distances between atom pairs in a periodic crystal.

    Inputs:
        R_red - matrix of reduced coordinates
        lat   - lattice vector array
    Outputs:
        dists   - matrix of distances between atom pairs
        Rij_min - min vector between pairs
    """
    A = lat.T
    natom = len(R_red)
    dists = np.zeros((natom, natom))
    Rij_min = np.zeros((natom, natom, 3))

    # Loop through all atom pairs
    for i in xrange(natom):
        for j in xrange(i):
            # Separation within unit cell
            rij = R_red[i] - R_red[j]

            # All possible separations allowing for crystal periodicity
            r = rij + LMN_GRID

            # Convert back to real space
            R = np.matmul(A, r.T).T
            d = norm(R, axis=1)

            # Find the index of vector that has the minimum separation
            min_idx = np.argmin(d)

            dists[i, j] = d[min_idx]
            dists[j, i] = dists[i, j]
            Rij_min[i, j] = R[min_idx]
            Rij_min[j, i] = -Rij_min[i, j]

    return dists, Rij_min


def get_factor(spacegroup, gamma):
    """
    Determines bond length multiplication factor.
    Depends on spacegroup and gamma angle.
    """
    if spacegroup == 12:
        return 1.4
    elif spacegroup == 33:
        return 1.4
    elif spacegroup == 167:
        return 1.5
    elif spacegroup == 194:
        return 1.3
    elif spacegroup == 206:
        return 1.5
    elif spacegroup == 227:
        if gamma < 60:
            return 1.4
        else:
            return 1.5
    else:
        msg = 'get_factor does not support the spacegroup: {}'
        raise NameError(msg.format(spacegroup))


def get_crytal_graph(R_red, atomlist, dists, factor=1.5):
    """
    Gets the crystal graph (CG) from the reduced atomic coordinates.

    Inputs:
        R_red    - atomic positions in reduced coordinates
        atomlist - list of atomic symbols for each position
        dists    - matrix of distances between atom pairs
        factor   - bond length multiplication factor
    Outputs:
        G - crystal graph
    """
    natom = len(atomlist)
    G = nx.MultiGraph()

    # Loop through all atom pairs
    for i in xrange(natom):
        sym_i = atomlist[i]

        for j in xrange(i):
            sym_j = atomlist[j]

            # Look for M-O bonds only
            if (sym_i != sym_j) and ((sym_i == 'O') or (sym_j == 'O')):
                node_i = '{}_{}'.format(sym_i, i)
                node_j = '{}_{}'.format(sym_j, j)

                # Add nodes (element) if not already in graph
                if node_i not in G:
                    G.add_node(node_i,
                               symbol=sym_i,
                               electroneg=ELEMENTS[sym_i].X,
                               atomic_mass=ELEMENTS[sym_i].atomic_mass)
                if node_j not in G:
                    G.add_node(node_j,
                               symbol=sym_j,
                               electroneg=ELEMENTS[sym_j].X,
                               atomic_mass=ELEMENTS[sym_j].atomic_mass)

                # Calculate max atomic separation that could be called a bond
                R_max = (ELEMENTS[sym_i].average_ionic_radius +
                         ELEMENTS[sym_j].average_ionic_radius) * factor

                if dists[i, j] < R_max:
                    sym_metal = sym_i if not 'O' else sym_j

                    # Add edges (bonds)
                    G.add_edge(node_i, node_j,
                               symbol='{}-O'.format(sym_metal),
                               bond_length=dists[i, j])

    # Add M-O or O-M coordination numbers to each atom
    for i in xrange(natom):
        sym_i = atomlist[i]
        node_i = '{}_{}'.format(sym_i, i)

        crdn_i = list(G.neighbors(node_i))
        cn_i = len(crdn_i)

        G.node[node_i]['cn'] = cn_i

    return G
