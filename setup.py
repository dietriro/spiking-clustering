#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="spikingclustering",
    version="0.1",
    description="Package for unsupervised spiking clustering with RBF neurons",
    url="https://gitlab.lrz.de/robind/spiking_clustering/",
    author="Technical University of Munich. Informatik VI",
    packages=find_packages(exclude=["examples", "evaluation"]),
    install_requires=[
    ],
)
