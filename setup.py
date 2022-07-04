from setuptools import setup

setup(
    name='tox',
    version='0.0.1',
    author='Hany Abdulsamad',
    author_email='hany@robot-learning.de',
    description='A toolbox for trajectory optimization based on JAX',
    install_requires=['jax', 'flax', 'matplotlib'],
    zip_safe=False,
)
