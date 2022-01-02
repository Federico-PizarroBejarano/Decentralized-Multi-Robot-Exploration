# Setup the paths
import os

if 'chi-hong' in os.getcwd():
    PROJECT_PATH = '/home/chi-hong/Desktop/RICHARD/Decentralized-Multi-Robot-Exploration/decentralized_exploration/dme_drl'
else:
    PROJECT_PATH = '/Users/richardren/VisualStudioCodeProjects/Decentralized-Multi-Robot-Exploration/decentralized_exploration/dme_drl'

CONFIG_PATH = PROJECT_PATH + '/assets/config.yaml'
MODEL_DIR = PROJECT_PATH + '/model/'

MANUAL_CHECK_PATH = PROJECT_PATH + '/manual_check/'
os.makedirs(MANUAL_CHECK_PATH, exist_ok=True)

RESET_WORLD_PATH = MANUAL_CHECK_PATH + '/reset_world/'
os.makedirs(RESET_WORLD_PATH, exist_ok=True)
RESET_ROBOT_PATH = MANUAL_CHECK_PATH + '/reset_robot/'
os.makedirs(RESET_ROBOT_PATH, exist_ok=True)

STEP_WORLD_PATH = MANUAL_CHECK_PATH + '/step_world/'
os.makedirs(STEP_WORLD_PATH, exist_ok=True)
STEP_ROBOT_PATH = MANUAL_CHECK_PATH + '/step_robot/'
os.makedirs(STEP_ROBOT_PATH, exist_ok=True)


# Setup the render
render_world = False
render_robot_map = False

manual_check = True

ID_TO_COLOR = {0:'red', 1:'green', 2:'blue'}

ACTION_TO_NAME = ['s-se', 'se,e', 'e-ne', 'ne-n', 'n-nw', 'nw-w', 'w-sw', 'sw-s']
