# NOMAD2018 Transparent Conductive Oxide Materials Predictive Models

Using XGBoost models to predict the bandgap and formation energy of In-Al-Ga-O materials.
These models use the following features:

- Basic compositional information and unit cell parameters
- Elemental properties averaged over the atoms in the cell e.g. electronegativity, ionization potential, orbital radii
- Average coordination numbers of the elements
- Statistics of M-O bond lengths
- Averaged feature vectors obtained from a crystal graph representation of the unit cell

For information on the dataset and challenge see: https://www.kaggle.com/c/nomad2018-predict-transparent-conductors
