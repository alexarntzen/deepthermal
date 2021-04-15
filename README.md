# FiniteElementProject
Project for 401-4656-21L Deep Learning in Scientific Computing FS2021 at ETH Zürich. 

This group members for this project are: 
 * Alexander Johan Arntzen 

The source files for the solver and quadrature methods are placed in the `deepthermal/` directory. 
The tests and figure generating methods are placed in `test/` directory. 

To run all the test run: 

```
python3 -m unittest
```
from in the top level directory. 
<!---

Alternatively  append `test.test_quadrature.TestQuadratureMethods`, `test.test_solver.TestHomogeneousDirichlet`,  `test.test_meshing.TestMeshing`
or `test.test_solver.TestSolverNeumann` to run the test corresponding to each task. These test will compare the numerical solutions with the analytical solution as required by the project description. 
For example: 
```
python3 -m unittest test.test_quadrature.TestQuadratureMethods
```
Then the figures will be generated into the `figures/` directory. 

To test the methods developed for part 2 of the project run 
```
python3 -m unittest test.test_stress_recovery.TestStressRecovery
```
,  
```
python3 -m unittest test.test_elasticity_solver.TestElasticHomogeneousDirichlet
```
or 
```
python3 -m unittest test.test_elasticity_solver.TestElasticSolverPerformance
```
.
-->