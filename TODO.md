# TODO List 

This is a list of tasks that need to be done as well as possible enhancements

### Single Robot Exploration

- [x] Implement proper reinforcement learning algorithm
    - [x] Add reward propogation into hex_grid
    - [x] Define list of robot actions
    - [x] Change Robot.choose_next_pose to choose based on rewards 
- [ ] Allow for robots of varying sizes
- [ ] Add RangeFinder that works more realistically
- [ ] Asynchronously scan and update map while choosing actions in another thread

### Multi-Robot Exploration

- [ ] Synchronously create multiple robots with perfect communication
    - [x] Design map merging technique
    - [ ] Design basic collision avoidance
    - [ ] Update Markov Decision Process for all robots to operate synchronously
- [ ] Allow robots to operate asynchronously
- [ ] Modify MDP to allow for imperfect communication
    - [ ] Create Voronoi diagram of map
    - [ ] Modify MDP using distributed value functions for imperfect communication

### Enhancements

- [ ] Add comprehensive unit testing
- [ ] Add comprehensive integration testing
- [ ] Make dynamically growing map i.e. no previous knowledge of map size
- [ ] Improve efficiency