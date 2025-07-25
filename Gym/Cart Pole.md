# CartPole Environment in Machine Learning

The **CartPole** environment is a classic control problem widely used in reinforcement learning (RL) research. It provides a foundational testbed for developing and evaluating RL algorithms.

## Problem Description

In the CartPole problem, a pole is attached to a cart moving along a frictionless track. The pole starts upright and the goal is to prevent it from falling over by applying force to move the cart left or right. The agent must learn a policy that balances the pole as long as possible.

## Dynamics

- The system is governed by Newtonian physics.
- The agent can apply a force of fixed magnitude either to the left or right.
- The episode terminates when:
  - The pole angle exceeds a threshold (typically ±12 degrees).
  - The cart position exceeds a certain range on the x-axis (e.g., ±2.4 units).
  - The episode reaches a maximum number of steps (usually 500 in `CartPole-v1`).

## State Space

The environment provides a continuous 4-dimensional state space:
1. `cart position` (x)
2. `cart velocity` (x_dot)
3. `pole angle` (theta)
4. `pole angular velocity` (theta_dot)

Example observation:

## Action Space

The agent can take one of two discrete actions:
- `0`: Push cart to the left.
- `1`: Push cart to the right.

This makes it a **discrete action space** problem.

## Reward

- The agent receives a reward of `+1` for every timestep that the pole remains upright.
- Maximum episode return is equal to the number of steps before failure or time limit.

## Objective

The objective is to maximize the total reward, i.e., to keep the pole balanced for as long as possible.


## CartPole Variants

- `CartPole-v0`: Episode ends after 200 timesteps.
- `CartPole-v1`: Episode ends after 500 timesteps.

Use `CartPole-v1` for more challenging and longer training.



## Dependencies

To run the CartPole environment, install:

```bash
pip install gymnasium[classic_control]
```


## Sample Code (Random Agent)

```python
import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")  # For visualization, use "human"

observation, info = env.reset(seed=42)

for step in range(500):
    env.render()
    
    # Sample random action (0 or 1)
    action = env.action_space.sample()
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## Sample Code (Simple Policy)

```python
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset(seed=42)

def simple_policy(observation):
    _, _, angle, _ = observation
    return 0 if angle < 0 else 1  # Move to counteract the pole angle

for step in range(500):
    env.render()
    action = simple_policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Training with a Reinforcement Learning Agent (e.g., DQN)
While the above examples use random or heuristic policies, CartPole is commonly used to train RL agents like:

Q-Learning

Deep Q-Networks (DQN)

Policy Gradient methods (REINFORCE)

Actor-Critic algorithms


## Observation Space in CartPole Environment

The **observation space** in the CartPole environment is a 4-dimensional continuous space that represents the state of the system at each time step. This state information is crucial for decision-making by reinforcement learning agents.

## Environment

- Environment Name: `CartPole-v1`
- Framework: Gymnasium (formerly OpenAI Gym)


## Observation Space Details

The observation is a NumPy array of 4 floating-point values, representing the physical state of the cart and pole system:

| Index | Feature                  | Description                                                             | Unit           | Typical Range                     |
|-------|---------------------------|-------------------------------------------------------------------------|----------------|-----------------------------------|
| 0     | `cart position`           | Position of the cart on the track                                       | meters         | ~ -4.8 to +4.8                    |
| 1     | `cart velocity`           | Velocity of the cart                                                    | meters/second  | ~ -inf to +inf                    |
| 2     | `pole angle`              | Angle of the pole with respect to the vertical (0 = perfectly upright) | radians        | ~ -0.418 to +0.418 (~±24 degrees) |
| 3     | `pole angular velocity`   | Angular velocity of the pole                                            | radians/second | ~ -inf to +inf                    |

Note: While velocity values are unbounded theoretically, the environment often clips or ends episodes when these values grow too large or unstable.



## Observation Space (Code Example)

The observation space can be programmatically inspected as follows:

```python
import gymnasium as gym

env = gym.make("CartPole-v1")

# Print observation space
print("Observation Space:", env.observation_space)

# Sample an observation
sample_obs = env.observation_space.sample()
print("Sample Observation:", sample_obs)

# Reset environment and inspect first observation
obs, info = env.reset()
print("Initial Observation from reset():", obs)
```

## Accessing Features
```python
cart_position = obs[0]
cart_velocity = obs[1]
pole_angle = obs[2]
pole_angular_velocity = obs[3]
```

## Use in RL Algorithms
The observation vector is typically fed directly into neural networks or policy functions. For example, in Deep Q-Networks (DQN), this 4-dimensional vector serves as the input layer of the model.

## Important Notes

The observations are not normalized by default.

You may need to scale or normalize them depending on your algorithm.

Continuous state representation makes CartPole a good candidate for function approximation methods in RL.


## # Action Space in CartPole Environment

The **action space** in the CartPole environment defines the set of all possible actions that an agent can take at each time step. In the CartPole control task, the action space is **discrete** and consists of only two possible actions.

## Environment

- Environment Name: `CartPole-v1`
- Framework: Gymnasium (previously OpenAI Gym)

---

## Action Space Details

The action space is a **Discrete(2)** space. This means the agent can choose between two discrete actions:

| Action | Description               | Effect                             |
|--------|---------------------------|-------------------------------------|
| 0      | Push cart to the **left** | Applies a fixed negative force      |
| 1      | Push cart to the **right**| Applies a fixed positive force      |

- Both actions apply a horizontal force to the cart.
- The magnitude of the force is constant and environment-defined (usually `10.0` units).
- The action is applied for a fixed duration at each step (usually 0.02 seconds).

---

## Action Space (Code Example)

The action space can be programmatically inspected and used as follows:

```python
import gymnasium as gym

env = gym.make("CartPole-v1")

# Print the action space
print("Action Space:", env.action_space)

# Print number of discrete actions
print("Number of actions:", env.action_space.n)

# Sample a random action
action = env.action_space.sample()
print("Sampled Action:", action)

# Take one step using the sampled action
observation, reward, terminated, truncated, info = env.step(action)
print("Observation after action:", observation)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
```


## Using Action Space in a Loop
```python
obs, info = env.reset()
done = False

while not done:
    # Example: random action (replace with policy later)
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated
```

## Summary
The CartPole environment uses a discrete action space with two possible actions: push left or push right.

This simple setup makes CartPole ideal for learning and testing foundational reinforcement learning algorithms like:

Q-Learning

Deep Q-Networks (DQN)

Policy Gradient Methods


## State Features in CartPole Environment

The CartPole environment provides a 4-dimensional continuous state (observation) at every time step. These four values describe the current physical condition of the cart-pole system and are used by reinforcement learning agents to decide which action to take.


## 1. Position of the Cart

- **Definition**: The horizontal position of the cart on the track.
- **Index in Observation**: `observation[0]`
- **Unit**: meters
- **Range**: Approximately `-2.4` to `2.4`
- **Explanation**: The cart starts near the center (`0.0`). If it moves too far left (`< -2.4`) or right (`> 2.4`), the episode terminates. This value tracks how far the cart has deviated from the origin.


## 2. Velocity of the Cart

- **Definition**: The rate of change of the cart's position (i.e., how fast the cart is moving).
- **Index in Observation**: `observation[1]`
- **Unit**: meters per second
- **Range**: Unbounded in theory, but usually remains within `-inf` to `+inf` under normal operation
- **Explanation**: A positive value indicates the cart is moving right; a negative value indicates it is moving left. High speeds make control more difficult.


## 3. Angle of the Pole

- **Definition**: The angle between the pole and the vertical axis.
- **Index in Observation**: `observation[2]`
- **Unit**: radians
- **Range**: Approximately `-0.418` to `0.418` (about ±24 degrees)
- **Explanation**: The goal is to keep the pole angle close to zero. If the pole tilts more than ~24 degrees, the episode terminates. Negative angle means the pole leans left; positive means right.


## 4. Rotation Rate of the Pole (Angular Velocity)

- **Definition**: The rate of change of the pole's angle (angular velocity).
- **Index in Observation**: `observation[3]`
- **Unit**: radians per second
- **Range**: Unbounded in theory (`-inf` to `+inf`)
- **Explanation**: Indicates how quickly the pole is rotating. A large rotation rate means the pole is falling fast, and the agent needs to act quickly to correct it.

## Sample Code to Display State Features

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=42)

# Extract features
cart_position = obs[0]
cart_velocity = obs[1]
pole_angle = obs[2]
pole_angular_velocity = obs[3]

# Print them
print(f"Cart Position: {cart_position:.4f} m")
print(f"Cart Velocity: {cart_velocity:.4f} m/s")
print(f"Pole Angle: {pole_angle:.4f} rad")
print(f"Pole Angular Velocity: {pole_angular_velocity:.4f} rad/s")

env.close()
```

## Summary Table

| Index | Feature               | Unit           | Approximate Range | Role in Environment                    |
| ----- | --------------------- | -------------- | ----------------- | -------------------------------------- |
| 0     | Cart Position         | meters         | -2.4 to +2.4      | Horizontal location on track           |
| 1     | Cart Velocity         | meters/second  | -inf to +inf      | Speed and direction of cart            |
| 2     | Pole Angle            | radians        | -0.418 to +0.418  | Deviation from vertical upright        |
| 3     | Pole Angular Velocity | radians/second | -inf to +inf      | Speed and direction of pole's rotation |


## Notes
All four features are continuous and updated every time step.

They are essential for policy design, Q-value function inputs, and training neural networks.

The default environment does not normalize these values.
