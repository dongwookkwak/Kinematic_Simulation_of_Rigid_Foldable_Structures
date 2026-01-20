# Kinematic Simulation of Rigid Foldable Structures

This repository provides a unified kinematic simulation framework for rigid-foldable structures based on screw theory and graph-based constraint formulation.  
The framework is designed for both engineers and general users who aim to design rigid-foldable patterns or perform kinematic simulations without manually handling complex loop-closure constraints.

The full pipeline is supported through a unified data schema: once a rigid-foldable pattern is defined, desired joint angles and stiffness parameters can be specified, and the framework automatically computes feasible folding trajectories that satisfy all kinematic constraints.  
The resulting motion is then visualized through integrated visualization utilities, enabling intuitive inspection of folding behavior in multi-sheet and multi-layer structures.


## Features
- Unified data schema for multi-sheet rigid origami
- Graph-based loop detection using a minimum cycle basis
- Screw-theory-based Pfaffian constraint matrix generation
- Lagrange multiplier methods for computing feasible crease angle trajectories satisfying loop-closure constraints
- Visualization utilities


## Relation to Paper

The methodology and implementation in this repository are presented in:

Kwak, D., Cho, G., Chung, J., and Yang, J.,  
*A Unified Framework for Kinematic Simulation of Rigid Foldable Structures*,  
*arXiv preprint*, 2026.  
Available at: https://arxiv.org/abs/2601.10225


In addition, the algorithm used to compute feasible crease-angle trajectories
satisfying loop-closure constraints is implemented based on the following
work:

- Hu, Y., and Liang, H., *Folding simulation of rigid origami with Lagrange multiplier method*,  
  **International Journal of Solids and Structures**, 202, 552–561, 2020.  
  https://doi.org/10.1016/j.ijsolstr.2020.06.016



## Contributes

- Dongwook Kwak, Geonhee Cho, Jiook Chung

