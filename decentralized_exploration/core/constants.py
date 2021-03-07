class Actions():
    FORWARD = 'forward'
    COUNTER_CLOCKWISE = 'counter_clockwise'
    CLOCKWISE = 'clockwise'


class Orientation():
    """
    Note y axis is inverted
        1 == facing directly towards y==0 line
        2 == rotate 1 by 60 degrees counter-clockwise
        3 == rotate 2 by 60 degrees counter-clockwise
        4 == rotate 3 by 60 degrees counter-clockwise s.t. it faces the y=inf line
        5 == rotate 4 by 60 degrees counter-clockwise
        6 == rotate 5 by 60 degrees counter-clockwise
    """

    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6

    def __init__(self, orientation):
        self.orientation = orientation
        
        if orientation > 6 or orientation < 1: 
            print("Malformed orientation")
    
    def __eq__(self, other):
        return self.orientation == other.orientation


    def rotate_clockwise(self, in_place=False):
        new_orientation = self.orientation - 1 if self.orientation > 1 else 6
        
        if in_place:
            self.orientation = new_orientation
        else:
            return Orientation(new_orientation)
    

    def rotate_counter_clockwise(self, in_place=False):
        new_orientation = self.orientation + 1 if self.orientation < 6 else 1

        if in_place:
            self.orientation = new_orientation
        else:
            return Orientation(new_orientation)
    

    def distance_to_orientation(self, other_orientation):
        """
        Given two orientations, determines the minimum number of 60 degree rotations necessary to get
        from one orientation to the other, called orientation distance

        Parameters
        ----------
        start_orientation (Orientation): an Orientation object representing the starting orientation
        end_orientation (Orientation): an Orientation object representing the ending orientation

        Returns
        -------
        orientation_distance (int): an int representing the orientation distance
        """

        clockwise_distance = (self.orientation - other_orientation.orientation) % 6
        counter_clockwise_distance = (other_orientation.orientation - self.orientation) % 6

        orientation_distance = min(clockwise_distance, counter_clockwise_distance)

        if clockwise_distance > counter_clockwise_distance:
            is_clockwise = False
        else:
            is_clockwise = True

        return orientation_distance, is_clockwise
        