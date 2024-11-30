from setuptools import setup, find_packages
import sys
import os

setup(
    name='BikeSharePrediction',
    version='0.1',
    author='Alexander Busch',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torch-geometric',
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
        'tqdm',
        'optuna',
        'folium',
        'geopy',
        'wget',
        'pyproj',
        'tensorboard'
    ],
    entry_points={
        'console_scripts': [
            'train = bikesharepred.run_training:main',
            'test = bikesharepred.test:main',
            'test_training = bikesharepred.test_training:main'
        ]
    },
    python_requires='>=3.6')

# add folder to system library path to make all scripts directly accessible

sys.path.append(os.path.join(os.path.dirname(__file__), 'bikesharepred'))