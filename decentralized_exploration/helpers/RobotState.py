class RobotState:
    """
    A class used to represent the state of a robot
    
    Instance Attributes
    -------------------
    pixel_position (int): a 2-element array of integer pixel coordinates
    orientation (int): an integer representing the orientation of the robot
    """

    def __init__(self, pixel_position, orientation):
        self.pixel_position = pixel_position
        self.orientation = orientation
