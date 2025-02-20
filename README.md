This is a program for geometry optimizations of saturated hydrocarbons both in internal and cartesian coordinates using a force field.

# Theory
Geometry optimization is the process of finding atomic coordinates that minimize the potential energy of a molecular system. This corresponds to solving: 
<img src="https://latex.codecogs.com/svg.latex?\frac{\partial%20V}{\partial%20r_i}%20=%200" alt="Equation: partial V / partial r_i = 0">
where V(r) is the potential energy as a function of atomic coordinates $r$.

## Optimization algorithms
### 1. Steepest descent
   The Steepest descent method is a first-order optimization technique that moves the system in the direction of the negative gradient of the potential energy function:
<img src="https://latex.codecogs.com/svg.latex?r_{k+1}%20=%20r_k%20-%20\alpha\nabla%20V(r_k)" alt="r_{k+1} = r_k - alpha grad V(r_k)">
where α is a step size parameter.
The parameter α is used so that V(r_k + αp_k) reaches a minimum along p_k. This is known as performing a line search. A commonly-used requirement for a rough line search is that α should satisfy the so-called "Wolfe rules". In this code it is only implemented the first rule, which ensures that the energy in the new geometry becomes significantly lower:

<img src="https://latex.codecogs.com/svg.latex?V(r_k+\alpha%20p_k)%20%3C=%20V(r_k)%20+%20c_1%20\alpha%20p_k%20\cdot%20\nabla%20V(r_k)" alt="Wolfe condition">

### 3. BFGS Quasi-Newton Method
   The Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm is a quasi-Newton method that improves optimization efficiency by approximating the Hessian matrix (second derivatives of the energy function) iteratively. The Hessian update follows:
   
<img src="https://latex.codecogs.com/svg.latex?B_{k+1}%20=%20B_k%20+%20\frac{y_k%20\otimes%20y_k}{y_k%20\cdot%20s_k}%20-%20\frac{w_k%20\otimes%20w_k}{s_k%20\cdot%20w_k}" alt="BFGS update equation">

where:

<img src="https://latex.codecogs.com/svg.latex?s_k%20=%20r_{k+1}%20-%20r_k" alt="s_k = r_{k+1} - r_k"> is the displacement vector.
<img src="https://latex.codecogs.com/svg.latex?y_k%20=%20\nabla%20V(r_{k+1})%20-%20\nabla%20V(r_k)" alt="y_k = gradient difference"> is the gradient difference.
<img src="https://latex.codecogs.com/svg.latex?w_k%20=%20B_ks_k" alt="w_k = B_k s_k"> is an intermediate update term.
<img src="https://latex.codecogs.com/svg.latex?B_k" alt="B_k"> is the approximate inverse Hessian.

### Wilson B matrix
To work with redudant coordinate sets, it is necessary to compute the Wilson B matrix, which relates small displacements in internal coordinates to Cartesian displacements. It is defined as: 

<img src="https://latex.codecogs.com/svg.latex?\mathbf{B}%20=%20\frac{\partial%20q}{\partial%20r}" alt="B = partial q / partial r">

where  represents internal coordinates (e.g., bond lengths, angles, torsions) and  represents Cartesian coordinates. The B matrix is particularly useful in combination with the BFGS algorithm, as it allows second-derivative information to be approximated in internal coordinates, leading to better convergence properties compared to Cartesian optimization.

# Implementation 

## Features 
- Geometry optimization using Steepest Descent and BFGS.
- Gradient-based optimization in Cartesian and Internal Coordinates.
- Efficient Hessian approximation with BFGS updates.
- Energy and gradient computation based on molecular mechanics.

## How to use it

1. Install dependencies:
   `pip install numpy scipy`
2. Run optimazation:
   `python3 programming-project-final.py`

# References
- The Tinker software package, from which the ‘tiny’ forcefield used here is taken, can be found
here: https://dasher.wustl.edu/tinker/
- Using Redundant Internal Coordinates to Optimize Equilibrium Geometries and
Transition States, Journal of Computational Chemistry, 1996, 17, pp. 49 – 56


   
