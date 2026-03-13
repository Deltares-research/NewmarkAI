# GNNs to improve Newmark integration scheme

The objective is to provide a GNN algorithm that can replace newmark integration scheme.
Once the model is trained it will be faster to use the GNN-Newmark to solve the equation of motion than the Newmark integration scheme.

## process running

1.  create the FEM analysis

    To do this we use run_scatter.py (I edited the solver locally to write the results)

1. create the data loader

