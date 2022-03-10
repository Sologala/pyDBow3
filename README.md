# Pyfbow , A python wrapper of [DBow3](https://github.com/rmsalinas/DBow3) with pybind11 

# Build and Install

1. Clone this repo 
```shell
git clone --recurse-submodules  https://github.com/Sologala/pyDBow3.git
cd pyDBow3
```
2. install with python
```shell
python3 setup.py develop
```
You can also install `pyDBow3` into python's site-packages by
```shell
python3 setup.py install
```

# How to uninstall

After installed by python , A manifest file named `files.txt` will be created.
```shell
xargs rm -rf < files.txt
```

Or you can uninstall pyDBow3 with `pip`

# Quick Start 

Please ref to [this notebook](tests/test_wrap.ipynb)