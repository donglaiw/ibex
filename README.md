IbexHelper:
-------------

Helper functions for [Ibex Package](https://github.com/bmatejek/ibex) for the manipulation of skeleton and its simplified graph.

Install
===
- setup environment
```
conda create -n ibexHelper python=2.7
source activate ibexHelper
conda install cython numpy scipy networkx
```

- compilation
```
cd ibex/skeletionization/
python setup.py build_ext --inplace
cd ../../ibex/transforms/
python setup.py build_ext --inplace
```


Main Contributors
==================
- @bmatejek: mesh to skeleton conversion
- @srujanm: skeleton to graph conversion
- @abhimanyutalwar: graph simplification, notebook 
- @donglaiw: graph simplification, repo organization
