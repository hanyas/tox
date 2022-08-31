# Trajectory Optimization in JAX 

A toolbox for trajectory optimization and model-predictive control based on [JAX](https://github.com/google/jax)


## Installation (Conda)

```bash
git clone git@github.com:hanyas/tox.git
cd tox
conda env create --file env_tox.yml
conda activate tox
pip install -e .
```

## Sample Run

```bash
conda activate tox
cd tox
python -B examples/bsp_ilqr/point.py
```