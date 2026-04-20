This code is based on the repository:

https://github.com/jiechenjiechen/GNP

The following changes has been made:

- An example has been added where matrices as retrieved from Scatter are read
- Cg linear solver has been added, as this is more efficient for symmetric positive definite (spd) matrices, which is the case for sool structure interaction FEM.
