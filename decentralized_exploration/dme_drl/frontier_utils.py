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


def cleanup_frontier(_map, frontier, pose, config):
	coords_to_be_removed = set()
	coords_to_be_removed.add(pose)

	for coords in frontier:
		if not is_frontier(_map, coords, config):
			coords_to_be_removed.add(coords)

	frontier -= coords_to_be_removed
	return frontier

def update_frontier_after_scan(_map, frontier, free_points, pose, config):
	for free_point in free_points:
		if is_frontier(_map, free_point, config):
			frontier.add(free_point)

	return cleanup_frontier(_map, frontier, pose, config)

def merge_frontiers(_map, frontier1, frontier2, pose, config):
	merged_frontiers = frontier1 | frontier2
	return cleanup_frontier(_map, merged_frontiers, pose, config)

