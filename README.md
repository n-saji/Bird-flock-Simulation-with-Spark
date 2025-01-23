## 1. Task
### 1.1 Edit Distance
A script for loading text data and computing the edit distance is provided in [`edit_dist.py`](https://github.com/UB-CSE587/homework_2/blob/main/edit_dist.py), which includes a basic implementation using a `for` loop. 
- A Spark version
- A multi-process version

### 1.2 Flock Move Simulation using Spark
In this task, each bird’s position is represented by a point in 3D space. Each bird follows movement rules based on flock dynamics

1. **Alignment**: Birds attempt to stay close to the leader bird, which follows a determined path with uniform velocity.
2. **Separation**: Birds maintain distance from nearby neighbors. If a bird gets too close to a neighbor (within a threshold), it moves away.
3. **Cohesion**: Birds strive to stay with the flock. If a bird is too far from its nearest neighbor (beyond a threshold), it moves closer.
4. **Velocity Constraints**: Flying speed is restricted to a certain range—too slow and the bird risks falling; too fast, and it exceeds physical limits.

A demonstration of the bird flock movement is shown below:
![bird](bird_simulation.gif)
---
