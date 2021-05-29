# Background
- [Weakly Interacting Massive Particle Distribution in and Evaporation from the Sun](https://ui.adsabs.harvard.edu/abs/1985ApJ...294..663S/abstract)
- [Effect of hypothetical, weakly interacting, massive particles on energy transport in the solar interior](https://ui.adsabs.harvard.edu/abs/1987ApJ...321..560G/abstract)

# Documentation
### Dependencies
 - [py_mesa_reader](https://github.com/wmwolf/py_mesa_reader)
 - [scipy](https://www.scipy.org/)
 - [matplotlib](https://matplotlib.org/stable/index.html)

### Usage
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

```./DM_evap_MESA.py -D mesa_1 -p 725 -E```.

The CSV files `TM4_***.CSV` are data files containing DM temperature verus DM mass. These are calculated for the Ilie4 star. The way it is set up now, the program will automatically pick the correct profile based on the profile index pass it. As of now you still have to manually change the `TM4` to something else if you wish to calculate for another star.

These indices can be found inside the `profiles.index` file.

The current command line arguments are:
```
usage: DM_evap_MESA.py [-h] [-D DIREC] [-p PROFILE] [-T] [-M MESA] [-E] [-P] [-e] [-R R311] [-H]

optional arguments:
  -h, --help            show this help message and exit
  -D DIREC, --direc DIREC
                        directory containing MESA profile and history files
  -p PROFILE, --profile PROFILE
                        index of the profile to use
  -T, --TchiMchi        use MESA data files to solve for DM temperature with Eq. 4.10 from Spergel and Press 1985
  -M MESA, --MESA MESA  mass of DM in GeV to use, plot stellar params from MESA
  -E, --Evap            calculate and plot DM evaporation rate using MESA data files
  -P POLY, --poly POLY  calc the evap polytrope, give the CSV file holding the tau vs. mu data for an N=3 polytrope
  -S STAR, --star STAR  100, 300, or 1000
  -e, --evapcsv         plot DM evaporation rate using previously calculated csv file
  -R R311, --R311 R311  mass of DM in GeV to use, plot Goulde Eq. 3.11
  -H, --heatmap         plot heatmaps for alpha, beta, and gamma
```
These can be printed with `./DM_evap_MESA.py -h`.

# Issues/To Do
 <!-- - [X] Issues with the calculated potential from MESA's gravitational acceleration? -->
 <!-- - [X] Add comparison plots with N=3 poltropes. -->
 <!-- - [X] Code Goulde 3.11 R(w|v) function. -->
 <!-- - [X] Get MESA Evap. rates. -->
 <!-- - [X] Fix polytrope number density!!! -->
 <!-- - [X] How to get polytrope central temp??? -->
 <!-- - [X] Two different cetral densities for polytrope? -->
 <!-- - [X] Boltzmann constant in alpha, beta, gamma and mpf(gibberish)? -->
 <!-- - [X] Why is v_esc so different for MESA and N=3? -->
 <!-- - [X] Check V_esc within the star is being calculated right. -->
 <!-- - [X] Check normalization factor. -->
 <!-- - [X] Check input functions to DM temp (i.e. boltzman constant). -->
 <!-- - [X] Check DM temp lines up with expected. -->
 <!-- - [X] Run evap code using polytrope stellar functions, is it still bad? -->
 <!-- - [X] Discrepency in central density between MESA and N=3? -->
 <!-- - [X] Check against Caleb's rates across the mass range! -->
 <!-- - [X] E_e and E_c dependence on r??? -->
 - [ ] Precision issues in scipy quad integrals?
 - [ ] Automate plotting multiple models and/or multiple stars.

### quad error
```
IntegrationWarning: The maximum number of subdivisions (50) has been achieved. If increasing the limit yields no improvement it is advised to analyze the integrand in order to determine the difficulties. If the position of a local difficulty can be determined (singularity, discontinuity) one will probably gain from splitting up the interval and calling the integrator on the subranges. Perhaps a special-purpose integrator should be used.
IntegrationWarning: The occurrence of roundoff error is detected, which prevents the requested tolerance from being achieved. The error may be underestimated.
```

# ZAMS
The closest profiles to zero age main sequence are
```
4 -> 725
5 -> 400
6 -> 400
```

# Plots
![evaporation](./plots/E_5_400.png)
![Eq. 3.11](./plots/R311_4_725_-1.png)
![normalization factor](./plots/norm_5_400.png)
![density](./plots/rho_5_400.png)
![number density](./plots/np_5_400.png)
![potential](./plots/np_5_400.png)
![temperature](./plots/T_5_400.png)
![escape velocity](./plots/vesc_5_400.png)
