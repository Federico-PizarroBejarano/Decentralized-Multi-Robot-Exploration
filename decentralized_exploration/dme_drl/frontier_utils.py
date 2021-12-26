import numpy as np

def is_in_map(_map, y, x):
	return 0 <= y < _map.shape[0] and 0 <= x < _map.shape[1]


def get_neighbours(_map, coords, radius=1):
	neighbours = []
	y, x = coords[0], coords[1]

	for dy in range(-radius, radius + 1):
		for dx in range(-radius, radius + 1):
			neighbour_y = y + dy
			neighbour_x = x + dx
			if is_in_map(_map, neighbour_y, neighbour_x):
				neighbours.append((neighbour_y, neighbour_x))
	return neighbours


def is_frontier(_map, coords, config):
	neighbours = get_neighbours(_map, coords)
	for neighbour in neighbours:
		if _map[neighbour] == config['color']['uncertain']:
			return True
	return False


def cleanup_frontier(_map, frontier, config):
	coords_to_be_removed = set()

	for coords in frontier:
		if not is_frontier(_map, coords, config):
			coords_to_be_removed.add(coords)

	frontier -= coords_to_be_removed
	return frontier

def update_frontier(_map, frontier, config):
	free_points = np.argwhere(_map == config['color']['free'])
	for free_point in free_points:
		if is_frontier(_map, free_point, config):
			frontier.add((free_point[0], free_point[1]))
	return cleanup_frontier(_map, frontier, config)

def update_frontier_and_remove_pose(_map, frontier, pose, config):
	updated_frontier = update_frontier(_map, frontier, config)
	return remove_pose_from_frontier(updated_frontier, pose)

def merge_frontiers(_map, frontier1, frontier2, config):
	union_of_frontiers = frontier1 | frontier2
	return cleanup_frontier(_map, union_of_frontiers, config)

def merge_frontiers_and_remove_pose(_map, frontier1, frontier2, pose, config):
	merged_frontiers = merge_frontiers(_map, frontier1, frontier2, config)
	return remove_pose_from_frontier(merged_frontiers, pose)

def remove_pose_from_frontier(frontier, pose):
	if pose in frontier:
		frontier.remove(pose)
	return frontier
