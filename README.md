This is a program for geometry optimizations both in internal and cartesian coordinates.

#Theory
Geometry optimization is the process of finding atomic coordinates that minimize the potential energy of a molecular system. This corresponds to solving: 
\begin{equation}
\frac{\partial V}{\partial r_i} = 0
\end{equation}
where V(r) is the potential energy as a function of atomic coordinates $r$.

## Optimization algorithms
1. Steepest descent
   The Steepest descent method is a first-order optimization technique that moves the system in the direction of the negative gradient of the potential energy function: $r_{k+1} = r_k - \alpha\grad V(r_k) $ where $\alpha $ is a step size parameter.
The parameter $\alpha$ is used so that $V(r_k + \alpha p_k )$ reaches a minimum along $p_k$. This is known as performing a line search. A commonly-used requirement for a rough line search is that $\alpha$ should satisfy the so-called "Wolfe rules". In this code it is only implemented the first rule, which ensures that the energy in the new geometry becomes significantly lower. 
\begin{equation}
V(r_k+\alpha p_k) <= V(r_k) + c_1 \alpha p_k \cdot \grad V(r_k)
\end{equation}
2. BFGS Quasi-Newton Method
   The Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm is a quasi-Newton method that improves optimization efficiency by approximating the Hessian matrix (second derivatives of the energy function) iteratively. The Hessian update follows:
\begin{equation}
B_{k+1} = B_k + \frac{y_k \otimes y_k}{y_k \cdot s_k} - \frac{w_k \otimes w_k}{s_k \cdot w_k}
\end{equation}
