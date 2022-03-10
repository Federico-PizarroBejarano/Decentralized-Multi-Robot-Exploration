# Setup the paths
import os
import shutil

if 'chi-hong' in os.getcwd():
    PROJECT_PATH = '/home/chi-hong/Desktop/RICHARD/Decentralized-Multi-Robot-Exploration/decentralized_exploration/dme_drl'
elif 'asblab' in os.getcwd():
    PROJECT_PATH = '/home/asblab/Desktop/RICHARD/Decentralized-Multi-Robot-Exploration/decentralized_exploration/dme_drl'
else:
    PROJECT_PATH = '/Users/richardren/VisualStudioCodeProjects/Decentralized-Multi-Robot-Exploration/decentralized_exploration/dme_drl'

CONFIG_PATH = PROJECT_PATH + '/assets/config.yaml'
MODEL_DIR = PROJECT_PATH + '/model/'

manual_check = False

MANUAL_CHECK_PATH = PROJECT_PATH + '/manual_check/'
shutil.rmtree(MANUAL_CHECK_PATH)
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
render_world = True
render_robot_map = False


ID_TO_COLOR = {0:'red', 1:'green', 2:'blue'}

ACTION_TO_NAME = ['n-ne', 'ne,e', 'e-se', 'se-s', 's-sw', 'sw-w', 'w-nw', 'nw-n']
