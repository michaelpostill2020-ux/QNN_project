# QNN_project



This repository contains a modular, configuration-driven Quantum Machine Learning (QML) framework designed for use in providing a quantum boost to machine learning piplines. This boost is designed to take the 3 best training variables from your ML pipline, entangle them together then train a quantum neural network in a regression task. This learns addtional correlations from the features. The ouput feature of the QNN is then injected into a classical network that trains on the 3 orignal feature plus this quantum feature. We also train a classical network here without the quantum boost to show the difference in performance. 

In testing this method has shown to reduce the MAE by 60% between the quantum hybrid NN and the classical NN. 

To make this framework accessible to external collaborators and CI/CD environments, it features a fully decoupled architecture, YAML-driven configurations, and a synthetic ROOT generator that enables end-to-end testing without requiring access to private ATLAS data or high-performance computing clusters.







# To use the QNN:

Execute setup script (creates virtual environment 'qml_env')

source setup_env.sh

Activate the virtual environment
source qml_env/bin/activate



Use the config.yaml to edit hyperparamters.

You can create synthetic data to play around with:
python generate_mock_data.py --outdir ./mock_data


convert files into h5
python run_regression.py --convert --config config.yaml


training both the quantum and classical networks
python run_regression.py --train --config config.yaml

training will automatically produce plots of train/val loss with each epoch for each k-fold. 

Evaluate with Pearson correlations and feature importance. Also compares model with and without quantum boost.
python run_regression.py --evaluate --config config.yaml



# Run unit and integration tests
python -m pytest -v