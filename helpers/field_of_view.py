import matplotlib.pyplot as plt
import numpy as np

def field_of_view(world_map, robot_pos, world_size):
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


if __name__ == "__main__":
    I = np.load('./maps/map_1_small.npy')
    occupied_points, free_points = field_of_view(I, (30, 77), I.shape)
    unknown_I = np.ones(I.shape)

    occ_rows, occ_cols = [p[0] for p in occupied_points], [p[1] for p in occupied_points] 
    free_rows, free_cols = [p[0] for p in free_points], [p[1] for p in free_points]

    unknown_I[occ_rows, occ_cols] = 2
    unknown_I[free_rows, free_cols] = 0

    plt.imshow(unknown_I, cmap='gray')
    plt.show()