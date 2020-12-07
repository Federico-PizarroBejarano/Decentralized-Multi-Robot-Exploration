import numpy as np

def field_of_view(world_map, robot_pos):
    world_size = world_map.shape
    all_free_points = set()
    all_occupied_points = set()

    for y in range(world_size[1]):
        x = 0

        all_points = bresenham(world_map, start = robot_pos, end = (x, y))
        all_free_points = all_free_points.union(set(all_points[:-1]))
        all_occupied_points.add(all_points[-1])

        x = world_size[0]-1

        all_points = bresenham(world_map, start = robot_pos, end = (x, y))
        all_free_points = all_free_points.union(set(all_points[:-1]))
        all_occupied_points.add(all_points[-1])

    for x in range(world_size[0]):
        y = 0

        all_points = bresenham(world_map, start = robot_pos, end = (x, y))
        all_free_points = all_free_points.union(set(all_points[:-1]))
        all_occupied_points.add(all_points[-1])

        y = world_size[1]-1

        all_points = bresenham(world_map, start = robot_pos, end = (x, y))
        all_free_points = all_free_points.union(set(all_points[:-1]))
        all_occupied_points.add(all_points[-1])
    
    return all_occupied_points, all_free_points


def bresenham(world_map, start, end):
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

