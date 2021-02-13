# Examples

## Skeletonization (test_skel.py)
- `python test_skel.py 0 PATH_SEGMENT_H5_FILE`

## ERL Evaluation (test_erl.py)
- install [funlib.evaluate](https://github.com/funkelab/funlib.evaluate)
```
conda env create -n erl-eval
source activate erl-eval
conda install -c conda-forge -c ostrokach-forge -c pkgw-forge graph-tool
pip install -r requirements.txt
pip install --editable .
```
- `python test_erl.py 0 PATH_SKEL_PICKLE_FILE PATH_SEGMENT_H5_FILE`
