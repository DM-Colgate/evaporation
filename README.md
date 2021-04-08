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

Specify the directory using the `-D` flag, then the index of profile file you wish to use with the `-p` flag:

```./DM_evap_MESA.py -D mesa_1 -p 700```.

The CSV files `TM4_***.CSV` are data files containing DM temperature verus DM mass. These are calculated for the Ilie4 star. The way it is set up now, the program will automatically pick the correct profile based on the profile index pass it. As of now you still have to manually change the `TM4` to something else if you wish to calculate for another star.

These indices can be found inside the `profiles.index` file.

The current command line arguments are:
```
usage: DM_evap_MESA.py [-h] [-D DIREC] [-p PROFILE] [-T TCHIMCHI] [-M] [-P] [-e]
                       [-R] [-H]

optional arguments:
  -h, --help            show this help message and exit
  -D DIREC, --direc DIREC
                        directory containing MESA profile and history files
  -p PROFILE, --profile PROFILE
                        index of the profile to use
  -T TCHIMCHI, --TchiMchi TCHIMCHI
                        name of csv file to store T_chi data in after solving
                        with Eq. 4.10 from Spergel and Press 1985
  -M, --MESA            plot stellar parameters from MESA
  -P, --poly            plot stellar parameters for N=3 polytope
  -e, --evap            plot DM evap rate from MESA data files
  -R, --G311            plot Gould 3.11 equation
  -H, --heatmap         plot heatmaps for alpha, beta, and gamma
usage: evap_snapshot.py [-h] [-D DIREC] [-p PROFILE] [-T] [-t] [-V] [-v] [-n]
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
 - [ ] Why is potential from MESA wrong by a factor of 0.5???
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

### quad error
```
IntegrationWarning: The maximum number of subdivisions (50) has been achieved. If increasing the limit yields no improvement it is advised to analyze the integrand in order to determine the difficulties. If the position of a local difficulty can be determined (singularity, discontinuity) one will probably gain from splitting up the interval and calling the integrator on the subranges. Perhaps a special-purpose integrator should be used.
IntegrationWarning: The occurrence of roundoff error is detected, which prevents the requested tolerance from being achieved. The error may be underestimated.
```

# Plots
![density](./plots/Ilie4_700_density.png)
![number density](./plots/Ilie4_700_np.png)
![potential](./plots/Ilie4_700_phi.png)
![temperature](./plots/Ilie4_700_temp.png)
![escape velocity](./plots/Ilie4_700_vesc.png)
![Eq. 3.11](./plots/Ilie4_700_R.png)
![evaporation](./plots/Ilie4_700_evap.png)
