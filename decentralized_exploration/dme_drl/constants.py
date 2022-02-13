# Setup the paths
import os
import shutil

if 'chi-hong' in os.getcwd():
    PROJECT_PATH = '/home/chi-hong/Desktop/RICHARD/Decentralized-Multi-Robot-Exploration/decentralized_exploration/dme_drl'
else:
    PROJECT_PATH = '/Users/richardren/VisualStudioCodeProjects/Decentralized-Multi-Robot-Exploration/decentralized_exploration/dme_drl'


CONFIG_PATH = PROJECT_PATH + '/assets/config.yaml'
MODEL_DIR = PROJECT_PATH + '/model/'

manual_check = False

MANUAL_CHECK_DIR = PROJECT_PATH + '/manual_check/'
shutil.rmtree(MANUAL_CHECK_DIR)
os.makedirs(MANUAL_CHECK_DIR, exist_ok=True)

RESET_WORLD_DIR = MANUAL_CHECK_DIR + '/reset_world/'
os.makedirs(RESET_WORLD_DIR, exist_ok=True)
RESET_ROBOT_DIR = MANUAL_CHECK_DIR + '/reset_robot/'
os.makedirs(RESET_ROBOT_DIR, exist_ok=True)

STEP_WORLD_DIR = MANUAL_CHECK_DIR + '/step_world/'
os.makedirs(STEP_WORLD_DIR, exist_ok=True)
STEP_ROBOT_DIR = MANUAL_CHECK_DIR + '/step_robot/'
os.makedirs(STEP_ROBOT_DIR, exist_ok=True)


# Setup the render
render_world = False
render_robot_map = False


ID_TO_COLOR = {0:'red', 1:'green', 2:'blue'}

ACTION_TO_NAME = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw']
