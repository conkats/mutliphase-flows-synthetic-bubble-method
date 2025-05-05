# A synthetic-bubble-method for mutliphase-flows
This repository contains a modified model used to define a transient inlet boundary condition for Volume-Of-Fluid (VOF) simulations in StarCCM+ flow solver, based on the "Synthetic Bubble Model" (SBM) of De Moerloose PhD published in 2020 at Ghent University. You can find the full text [here](https://lib.ugent.be/catalog/rug01:002978914) and following the approach in the  github repo [here]:https://github.com/ldmoerlo/InletModelling_VOF. 

Here the main python script has been extended to generate a .csv of the variables in a table(xyz, time) format that StarCCM+ accepts for the transient inlet boundary conditions. The model constructs a virtual pre-domain in front of the actual domain's inlet based on the defined inlet geometry and subsequently fills the pre-domain - initially filled with liquid - with gas bubbles of arbitrary shape and of random size, at a randomly chosen position in time and space. 

This work forms part of European Project on multiphase flow and FSI modelling.

