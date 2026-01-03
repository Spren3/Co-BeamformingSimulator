# Coordinated Beamforming Simulator
Implementation of a Monte Carlo Simulator of IEEE 802.11 Networks Supporting Coordinated Beamforming <br><br>
Project was implemented in Python, because it supports rapid prototyping and supports multiple libraries. Of these libraries, the following were used: 
- NumPy
- SciPy
- MatPlotLib

This simulator is based on beam pattern created in purpose of 

Basic expermients with one or two APs are proceeded in [basic_scenarios.py](/basic_scenarios.py) file. Also, there are stored functions responsible for creating plots with various metrics, depicted and explained in thesis. <br>
In main simulator file -- [simulator.py](/simulator.py), where <b>NetworkNode</b> class is created. There two types of topologies, called openspace and multiroom are generated, simulation rounds are proceeded and neccessary statistics are returned.
