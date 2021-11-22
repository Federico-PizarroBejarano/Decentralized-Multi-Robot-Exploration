from decentralized_exploration.core.robots.utils.field_of_view import field_of_view


class RangeFinder:
    '''
    A class used to represent a range_finder of any kind

    Instance Attributes
    -------------------
    full_range (int): maximum distance in meters the RangeFinder can reliably detect distance
    frequency (int): the frequency the RangeFinder scans

    Public Methods
    --------------
    scan(world): object to scans the given world
    '''

    def __init__(self, full_range, frequency):
        self._full_range = full_range
        self._frequency = frequency

    @property
    def full_range(self):
        return self._full_range

    @property
    def frequency(self):
        return self._frequency

    # Public Methods
    def scan(self, world, position):
        '''
        Scans the given world

        Parameters
        ----------
        world (World): a World object that the RangeFinder will scan
        position (list): a 2-element list of integer pixel coordinates

        Returns
        -------
        all_free_points (list): a list of pixel coordinates representing free points
        all_occupied_points (list): a list of pixel coordinates representing occupied points
        '''

        return field_of_view(world_map=world.world_map, robot_pos=position)
