# Documentation
### Dependencies:
 - [py_mesa_reader](https://github.com/wmwolf/py_mesa_readerhttps://github.com/wmwolf/py_mesa_reader)
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

Specify the directory as the first argument, then the indecies of profile files you wish to plot as the 2nd thru Nth arguments.
```./evap_snapshot.py mesa_1 400 555 500```

These indices can be found inside the `profiles.index` file.
