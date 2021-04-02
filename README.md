# Documentation
### Dependencies
 - [py_mesa_reader](https://github.com/wmwolf/py_mesa_reader)
 - [scipy](https://www.scipy.org/)
 - [matplotlib](https://matplotlib.org/stable/index.html)

### Usage
NOTE: Currently debuging and trying to get evaporation to work inside of `DM_evap_MESA.py`, to run do:
```
./DM_evap_MESA.py -D Ilie4_ii -p 700 -e
```
One I get reasonable evap I'll reorganize and get all the flags working again.

Execute from the command line, in the directory that contains MESA log directories like so:
```
├── DM_evap_MESA.py
└── mesa_1
    ├── history_1.data
    ├── profiles.index
    ├── profile1.index
    ...
    └── profile31.index
```

Specify the directory using the `-D` flag, then the index of profile file you wish to plot with the `-p` flag:

```./DM_evap_MESA.py -D mesa_1 -p 700```.

Finally, pass the `-T`, `-t`, `-V`, or `-n` flags to specify which plot (or plots) to show.

These indices can be found inside the `profiles.index` file.

The current command line arguments are:
```
usage: evap_snapshot.py [-h] [-D DIREC] [-p PROFILE] [-T] [-t] [-V] [-v] [-n]

optional arguments:
  -h, --help            show this help message and exit
  -D DIREC, --direc DIREC
                        directory containing MESA profile and history files
  -p PROFILE, --profile PROFILE
                        index of the profile to use
  -T, --TchiMchi        solve for and plot DM temperature vs DM mass
  -M, --MESA            plot stellar parameters from MESA
  -P, --poly            plot stellar parameters for N=3 polytope
  -e, --evap            plot DM evap rate from MESA data files
```
These can be printed with `./DM_evap_MESA.py -h`.

# Issues/To Do
 - [X] Issues with the calculated potential from MESA's gravitational acceleration?
 - [X] Add comparison plots with N=3 poltropes.
 - [X] Code Goulde 3.11 R(w|v) function.
 - [X] Get MESA Evap. rates.
 - [X] Fix polytrope number density!!!
 - [X] How to get polytrope central temp???
 - [X] Compare Goulde 3.11 R(w|v) function against Caleb's.
 - [X] Two different cetral densities for polytrope?
 - [ ] Boltzmann constant in alpha, beta, gamma and mpf(gibberish)?
 - [ ] Why is MESA's Gould 3.11 R(w|v) 10^9 larger than caleb's?
 - [ ] Why is v_esc so different for MESA and N=3?
 - [ ] Check V_esc within the star is being calculated right.
 - [ ] Check normalization factor.
 - [ ] Check input functions to DM temp (i.e. boltzman constant).
 - [ ] Check DM temp lines up with expected.
 - [ ] Run evap code using polytrope stellar functions, is it still bad?
 - [ ] Precision issues in scipy quad integrals?
 - [ ] Discrepency in central density between MESA and N=3?
 - [ ] Automate plotting multiple models and/or multiple stars.
 - [ ] Specify name of output CSV and plots from command line.

# Plots
![density](./plots/Ilie4_700_density.png)
![number density](./plots/Ilie4_700_np.png)
![potential](./plots/Ilie4_700_phi.png)
![temperature](./plots/Ilie4_700_temp.png)
![escape velocity](./plots/Ilie4_700_temp.png)
![Eq. 3.11](./plots/Ilie4_700_R.png)
![evaporation](./plots/Ilie4_700_evap.png)
