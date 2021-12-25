import os

if 'chi-hong' in os.getcwd():
    PROJECT_PATH = '/home/chi-hong/Desktop/RICHARD/Decentralized-Multi-Robot-Exploration/decentralized_exploration/dme_drl'
else:
    PROJECT_PATH = '/Users/richardren/VisualStudioCodeProjects/Decentralized-Multi-Robot-Exploration/decentralized_exploration/dme_drl'

CONFIG_PATH = PROJECT_PATH + '/assets/config.yaml'
MODEL_DIR = PROJECT_PATH + '/model/'