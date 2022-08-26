from setuptools import setup

setup(
    name="gym_pybullet_drones",
    version="1.0.0",
    py_modules=[],
    install_requires=[
        "numpy",
        "Pillow",
        "matplotlib",
        "cycler",
        "hydra",
        "hydra-core",
        "tensorboard",
        "omegaconf",
        "gym",
        "pybullet",
        "stable_baselines3",
        "ray[rllib]",
    ],
)
