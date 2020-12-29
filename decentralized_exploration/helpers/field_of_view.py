import numpy as np


def field_of_view(world_map, robot_pos, start_orientation, end_orientation):
    """
    Given a world map and the position of the robot, returns all free and occupied pixels in its field of view

    Parameters
    ----------
    world_map(numpy.ndarray): numpy array of pixels representing the map. 
        0  == free
        1  == occupied
    robot_pos(array): a 2-element array of pixel coordinates 
    start_orientation(int): an int 1-6 representing the orientation of the robot
        1 == facing directly towards y==0 line
        2 == rotate 1 by 60 degrees clockwise
        3 == rotate 2 by 60 degrees clockwise
        4 == rotate 3 by 60 degrees clockwise s.t. it faces the y=inf line
        5 == rotate 4 by 60 degrees clockwise
        6 == rotate 5 by 60 degrees clockwise
    end_orientation(int): an int 1-6 representing the desired orientation

    Returns
    ----------
    list, list: two lists of pixel coordinates representing occupied and free points
    """

    world_size = world_map.shape
    all_free_points = set()
    all_occupied_points = set()

    start_edge = get_edge_point(robot_pos, start_orientation, world_size)
    end_edge = get_edge_point(robot_pos, end_orientation, world_size)

    curr_point = start_edge

    if (start_orientation < end_orientation and not (start_orientation == 1 and end_orientation == 6)) or (start_orientation == 6 and end_orientation == 1):
        clockwise = False
    else:
        clockwise = True

    while curr_point != end_edge:
        if (curr_point == [0, 0] and clockwise) or (curr_point == [world_size[0] - 1, 0] and not clockwise):
            curr_point[1] += 1
        elif (curr_point == [0, 0] and not clockwise) or (curr_point == [0, world_size[1] - 1] and clockwise):
            curr_point[0] += 1
        elif (curr_point == [world_size[0] - 1, 0] and clockwise) or (curr_point == [world_size[0] - 1, world_size[1] - 1] and not clockwise):
            curr_point[0] -= 1
        elif (curr_point == [0, world_size[1] - 1] and not clockwise) or (curr_point == [world_size[0] - 1, world_size[1] - 1] and clockwise):
            curr_point[1] -= 1
        elif curr_point[0] != 0 and curr_point[0] != world_size[0] - 1:
            if (clockwise and curr_point[1] == 0) or (not clockwise and curr_point[1] == world_size[1]-1):
                curr_point[0] -= 1
            else:
                curr_point[0] += 1
        elif curr_point[1] != 0 and curr_point[1] != world_size[1] - 1:
            if (clockwise and curr_point[0] == 0) or (not clockwise and curr_point[0] == world_size[0]-1):
                curr_point[1] += 1
            else:
                curr_point[1] -= 1

        all_points = bresenham(world_map, start=robot_pos, end=curr_point)
        all_free_points = all_free_points.union(set(all_points[:-1]))
        all_occupied_points.add(all_points[-1])

    return all_occupied_points, all_free_points


def bresenham(world_map, start, end):
    """
    Given a world map, a starting pixel coordinate, and an end pixel coordinate returns all pixels 
    in line of sight using bresenham's algorithm

    Parameters
    ----------
    world_map(numpy.ndarray): numpy array of pixels representing the map. 
        0  == free
        1  == occupied
    start(array): a 2-element array of pixel coordinates representing starting pixel
    end(array): a 2-element array of pixel coordinates representing ending pixel

    Returns
    ----------
    list: a list of pixel coordinates representing line of sight, with the last pixel being 
        occupied and all others being free
    """

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

    for p in range(len(points)):
        point = points[p]
        if world_map[point[0]][point[1]] == 1:
            return points[:p] + [points[p]]
    return points


def get_edge_point(robot_pos, orientation, world_size):
    """
    Given the robot position and orientation, finds the point on the edge of the map 
    where the robot would intersect if it continued on its trajectory

    Parameters
    ----------
    robot_pos (tuple): tuple of integer pixel coordinates
    orientation (int): an int representing the current orientation
    world_size (tuple): a tuple of the size of the world_map

    Returns
    ----------
    edge_point (tuple): tuple of integer pixel coordinates of the edge
    """

    robot_pos = robot_pos[::-1]
    world_size = world_size[::-1]

    if orientation == 1:
        edge_point = [robot_pos[0], 0]
    elif orientation == 4:
        edge_point = [robot_pos[0], world_size[1] - 1]
    elif orientation == 2:
        slope = 1/(3**0.5)
        y_intercept = robot_pos[1] - slope*robot_pos[0]

        if y_intercept >= 0:
            edge_point = [0, y_intercept]
        else:
            y = 0
            x = (y - y_intercept)/slope
            edge_point = [x, y]
    elif orientation == 5:
        slope = 1/(3**0.5)
        y_intercept = robot_pos[1] - slope*robot_pos[0]

        x = world_size[0]-1
        y = slope*x + y_intercept

        if y <= world_size[1]-1:
            edge_point = [x, y]
        else:
            y = world_size[1] - 1
            x = (y - y_intercept)/slope
            edge_point = [x, y]
    
    elif orientation == 3:
        slope = -1/(3**0.5)
        y_intercept = robot_pos[1] - slope*robot_pos[0]

        if y_intercept <= world_size[1] - 1:
            edge_point = [0, y_intercept]
        else:
            y = world_size[1] - 1
            x = (y - y_intercept)/slope
            edge_point = [x, y]
    
    elif orientation == 6:
        slope = -1/(3**0.5)
        y_intercept = robot_pos[1] - slope*robot_pos[0]

        x = world_size[0]-1
        y = slope*x + y_intercept

        if y >= 0:
            edge_point = [x, y]
        else:
            y = 0
            x = (y - y_intercept)/slope
            edge_point = [x, y]
    
    return [round(edge_point[1]), round(edge_point[0])]