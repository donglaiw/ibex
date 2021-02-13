
ERL Evaluation
---
- install [funlib.evaluate](https://github.com/funkelab/funlib.evaluate)
```
conda env create -n erl-eval
source activate erl-eval
conda install -c conda-forge -c ostrokach-forge -c pkgw-forge graph-tool
pip install -r requirements.txt
pip install --editable .
```
- `python test_erl.py`
