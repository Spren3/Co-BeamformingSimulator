# Coordinated Beamforming Simulator
Implementation of a Monte Carlo Simulator of IEEE 802.11 Networks Supporting Coordinated Beamforming <br><br>
Project was implemented in Python, because it supports rapid prototyping and supports multiple libraries. Of these libraries, the following were used: 
- NumPy
- SciPy
- MatPlotLib

This simulator is based on beam pattern created in purpose of 

Basic expermients with one or two APs are proceeded in [basic_scenarios.py](/basic_scenarios.py) file. Also, there are stored functions responsible for creating plots with various metrics, depicted and explained in thesis. <br>
In main simulator file -- [simulator.py](/simulator.py), where <b>NetworkNode</b> class is created. There two types of topologies, called openspace and multiroom are generated, simulation rounds are proceeded and neccessary statistics are returned.

## **Installation**
1. **Clone the repository:**
    ```bash
    git clone https://github.com/Spren3/Co-BeamformingSimulator.git
    ```
2. **Proceed to the project directory:**
    ```bash
    cd Co-BeamformingSimulator
    ```
3. **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate #or
    source .venv/bin/activate #if on linux
    ```
4. **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
## **Usage**
Basic_scenarios file contains set of configurations, if you want to test one, uncomment execution line, e.g.:
```python
def fifth_scenario_4STA_2AP_line(d1=10, d2_range=None):
    if d2_range is None:
        d2_range = [2*d1, 3*d1, 4*d1, 5*d1, 6*d1, 7*d1, 8*d1]
    # main part of the code has been omitted in this description
    plt.show()

# fifth_scenario_4STA_2AP_line(d1=10, d2_range=None)

```
and run in terminal:
```bash
python basic_scenarios.py
```

For the simulator code:
```python
def round_sim(num_simulations: int, pattern_type: str, ap_selection: str, seed: int, topology_seed: int):
    np.random.seed(seed)
    f= 2.4 #GHz
    Tx_PWR = 20 # dBm
    noise=0.0000000004
    Bp=10 # meters
    total_thr=0
    generator = TopologyGenerator()
### Change grid and room size parameters or comment out the generation of multiroom topology and uncomment openspace
    topology = generator.generate_multiroom_topology(
        topo_seed=topology_seed,
        grid_size=(2, 2), 
        room_size=20.0
    )
    # topology = generator.generate_open_space_topology()
### most of the round simulation function omitted
    return float(np.mean(sim_totals)),sim_totals,per_station
### after last line of this function type for example:
your_variable=round_sim(100, "beam", "all", 37,1)[0]
```
After the closing round bracket, entering 0 in square brackets allows to save the average aggregate throughput per round for the selected configuration to a variable. Entering 1 in square brackets allows you to save a list of $n$ total throughputs to a variable for later use in a function that displays averages with confidence intervals, where $n$ is the number of rounds. Writing 2 in square brackets allows to save a list of throughputs per-station to a variable in order to pass it as an argument to the function showing the cumulative distribution function (CDF).
