import numpy as np
import matplotlib.pyplot as plt

from ..helpers.grid import Grid
from ..core.constants import probability_of_failed_scan


def field_of_view(world_map, robot_pos):
    '''
    Given a world map and the position of the robot, returns all free and occupied pixels in its field of view

    Parameters
    ----------
    world_map (numpy.ndarray): numpy array of pixels representing the map. 
        0  == free
        1  == occupied
    robot_pos (list): a 2-element list of pixel coordinates 

    Returns
    -------
    all_free_points (list): a list of pixel coordinates representing free points
    all_occupied_points (list): a list of pixel coordinates representing occupied points
    '''

    world_size = world_map.shape
    radius = Grid.radius
    y, x = robot_pos

    all_free_points = set()
    all_occupied_points = set()

    for yi in (max(y-radius, 0), min(y+radius, world_size[0]-1)):
        for xi in range(max(x-radius, 0), min(x+radius, world_size[1]-1)+1):
            all_points = bresenham(start=robot_pos, end=[yi, xi], world_map=world_map)
            all_free_points = all_free_points.union(set(all_points[:-1]))
            if world_map[all_points[-1][0], all_points[-1][1]] == 1: 
                all_occupied_points.add(all_points[-1])
            else:
                all_free_points.add(all_points[-1])
        
    for yi in range(max(y-radius, 0), min(y+radius, world_size[0]-1)+1):
        for xi in (max(x-radius, 0), min(x+radius, world_size[1]-1)):
            all_points = bresenham(start=robot_pos, end=[yi, xi], world_map=world_map, )
            all_free_points = all_free_points.union(set(all_points[:-1]))
            if world_map[all_points[-1][0], all_points[-1][1]] == 1: 
                all_occupied_points.add(all_points[-1])
            else:
                all_free_points.add(all_points[-1])
    
    all_occupied = list(all_occupied_points)
    all_free = list(all_free_points)

    keep_occ = [np.random.randint(100) > probability_of_failed_scan for i in range(len(all_occupied))]
    keep_free = [np.random.randint(100) > probability_of_failed_scan for i in range(len(all_free))]
    all_occupied_points = set([all_occupied[p] for p in range(len(all_occupied)) if keep_occ[p]])
    all_free_points = set([all_free[p] for p in range(len(all_free)) if keep_free[p]])

    return all_occupied_points, all_free_points


def bresenham(start, end, world_map=np.array([])):
    '''
    Given a world map, a starting pixel coordinate, and an end pixel coordinate returns all pixels 
    in line of sight using bresenham's algorithm

    Parameters
    ----------
    start(list): a 2-element list of pixel coordinates representing starting pixel
    end(list): a 2-element list of pixel coordinates representing ending pixel
    world_map(numpy.ndarray): numpy array of pixels representing the map. 
        0  == free
        1  == occupied

    Returns
    -------
    points (list): a list of pixel coordinates representing line of sight, with the last pixel being 
        occupied and all others being free
    '''

    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()

    if world_map.shape[0] != 0:
        for p in range(len(points)):
            point = points[p]
            if world_map[point[0]][point[1]] == 1:
                return points[:p] + [points[p]]
    
    return points

