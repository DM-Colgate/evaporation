# Documentation
### Dependencies:
 - [py_mesa_reader](https://github.com/wmwolf/py_mesa_reader)
 - scipy
 - matplotlib

### Usage:
Execute from the command line, in the directory that contains MESA log directories like so:
```
├── evap_snapshot.py
└── mesa_1
    ├── history_1.data
    ├── profiles.index
    ├── profile1.index
    ...
    └── profile31.index
```

Specify the directory using the `-D` flag, then the index of profile file you wish to plot with the `-p` flag:

```./evap_snapshot.py -D mesa_1 -p 700```.

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
  -T, --TchiMchi        plot DM temperature vs DM mass
  -t, --taumu           plot DM dimensionless temperature vs DM dimensionless mass
  -V, --phi             plot radial graviation potential from MESA data files
  -v, --phipoly         plot radial graviation potential for N=3 polytrope
  -n, --np              plot proton number denisty from MESA data files
```
These can be printed with `./evap_snapshot.py -h`.


# Note on Branches
Please feel free to make any commits you want to the 'testing' branch. To switch branches simply use the 'checkout' command:

`git checkout testing` or `git checkout master`

Just make sure to comment your code and add an explanitory commit message.  


# Issues
 - [X] Issues with the calculated potential from MESA's gravitational acceleration?
 - [ ] Implement MESA Evap. rates.
 - [ ] Implement N=3 polytrope Evap. rates.
 - [ ] Implement closed form Evap. approx.
 - [ ] Simultaneously plot multiple models and/or multiple stars.
