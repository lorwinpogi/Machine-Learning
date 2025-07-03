# Simulation Environment 

## What is a Simulation Environment?

A **simulation environment** in machine learning is a virtual world or system where agents (typically controlled by algorithms) can interact with a simulated environment to learn and make decisions. It is commonly used in:

- Reinforcement Learning (RL)
- Robotics and autonomous systems
- Game AI development
- Operations research
- Control systems

These environments provide controlled, repeatable scenarios where models can be trained and evaluated safely and efficiently.

---

## Key Components of a Simulation Environment

1. **Agent**  
   The entity (model) that takes actions based on the environment's state.

2. **Environment**  
   The system in which the agent operates. It returns an observation, reward, and sometimes a done flag.

3. **State**  
   A snapshot of the environment at a given time.

4. **Action**  
   A move made by the agent that can change the state of the environment.

5. **Reward**  
   A signal from the environment used to evaluate the agent's action.

6. **Episode**  
   A sequence of states, actions, and rewards ending in a terminal state.

---

## Popular Simulation Libraries in Python

### 1. **OpenAI Gym**
```bash
pip install gym
```

```python
import gym

env = gym.make("CartPole-v1")
obs = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # random action
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
```

### 2. **PyBullet**
```bash
pip install pybullet
```
Used for physics-based simulations (e.g., robotics).

```python
import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
robot = p.loadURDF("r2d2.urdf")

for _ in range(1000):
    p.stepSimulation()
    time.sleep(1./240)

p.disconnect()

```
### 3.  **Unity ML-Agents**
Unity’s toolkit for training intelligent agents in realistic 3D environments.

Requires Unity installation and Python package:

```bash

pip install mlagents

```

Agents communicate via a gRPC interface between Unity and Python.


Custom Simulation Environment (Using gym.Env)
You can create your own environment by subclassing gym.Env.

```python
import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=10, shape=(1,))
        self.action_space = spaces.Discrete(2)

    def reset(self):
        self.state = np.array([5])
        return self.state

    def step(self, action):
        if action == 0:
            self.state -= 1
        else:
            self.state += 1

        reward = 1.0 if self.state == 10 else -0.1
        done = bool(self.state >= 10 or self.state <= 0)

        return self.state, reward, done, {}

    def render(self):
        print(f"Current State: {self.state}")
```


## Agent in Simulation Environment (Machine Learning with Python)


In machine learning—particularly in **Reinforcement Learning (RL)** and simulation environments—an **Agent** is an entity that **interacts with an environment** by:

1. **Observing** the current state
2. **Taking actions** based on a policy
3. **Receiving rewards** or penalties
4. **Learning** from experience to improve its future performance

The agent's goal is typically to **maximize cumulative reward** over time.



## Core Responsibilities of an Agent

- **Perceive** the environment (observe the state)
- **Decide** what action to take (policy or model)
- **Act** to change the state of the environment
- **Learn** from the results of its actions (through training)



## Agent-Environment Interaction Loop

The basic interaction between an agent and an environment follows this loop:

```text
1. Environment provides state (observation)
2. Agent selects an action
3. Environment transitions to new state
4. Agent receives a reward
5. Repeat...


## Types of Agents

1. **Random Agent**  
   Takes actions randomly, useful as a baseline.

2. **Rule-Based Agent**  
   Follows fixed if-else logic.

3. **Learning Agent**  
   Learns from experience using algorithms like:
   
   - **Q-Learning**
   - **Deep Q-Networks (DQN)**
   - **Policy Gradient**
   - **Actor-Critic Methods**



**Agent Example in OpenAI Gym**
```python
import gym

env = gym.make("CartPole-v1")
obs = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # Random Agent
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
```

**Building a Custom Agent Class**

```python
import numpy as np
import gym

class QLearningAgent:
    def __init__(self, action_space_size, state_space_size, learning_rate=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.action_space_size = action_space_size

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_size)  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state, done):
        future_q = np.max(self.q_table[next_state])
        target = reward + self.gamma * future_q * (not done)
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])

        if done:
            self.epsilon *= self.epsilon_decay

```

To use this agent:
```python
env = gym.make("Taxi-v3")
agent = QLearningAgent(env.action_space.n, env.observation_space.n)

for episode in range(1000):
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
```

## Environment 


In machine learning—particularly in **Reinforcement Learning (RL)**—an **Environment** is the simulated world or system with which an agent interacts. It serves as the **interface** between the agent and the task that needs to be learned.

The environment:

- **Provides observations** to the agent (states)
- **Receives actions** from the agent
- **Returns rewards** and next states based on those actions
- **Determines when an episode ends**


## Components of an Environment

| Component        | Description |
|------------------|-------------|
| **Observation (State)** | The current situation of the environment, seen by the agent |
| **Action Space** | The set of all possible actions the agent can take |
| **Reward** | A signal returned after each action to guide learning |
| **Transition** | The change in state based on the agent’s action |
| **Episode** | A complete sequence from start to terminal state |


## Example: Environment Flow

Here’s a basic sequence of how an environment behaves:

```text
1. Environment is initialized (reset)
2. Environment gives initial state (observation)
3. Agent takes action
4. Environment returns:
   - next state
   - reward
   - done (is episode over?)
   - additional info (optional)
5. Repeat until done
```

### **Python Libraries for Simulation Environments**


### 1.  **OpenAI Gym**

```bash
pip install gym
```
```python
import gym

env = gym.make("CartPole-v1")
obs = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
```

### 2.  **Custom Environments with gym.Env**
You can create your own environment by subclassing gym.Env and defining the following methods:

__init__(self)

reset(self)

step(self, action)

render(self, mode='human') (optional)

close(self) (optional)

Example:

```python
import gym
from gym import spaces
import numpy as np

class SimpleEnv(gym.Env):
    def __init__(self):
        super(SimpleEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=10, shape=(1,))
        self.action_space = spaces.Discrete(2)
        self.state = np.array([5])

    def reset(self):
        self.state = np.array([5])
        return self.state

    def step(self, action):
        if action == 0:
            self.state -= 1
        else:
            self.state += 1

        reward = 1 if self.state == 10 else -0.1
        done = bool(self.state <= 0 or self.state >= 10)

        return self.state, reward, done, {}

    def render(self, mode="human"):
        print(f"Current state: {self.state}")

```

Usage:

```python
env = SimpleEnv()
state = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    env.render()
```


###  Types of Environments

- **Deterministic**  
  Same action in the same state always gives the same result.

- **Stochastic**  
  Actions can produce different outcomes even in the same state (adds realism).

- **Discrete vs Continuous**
  - **Discrete**: Limited set of actions/states.
  - **Continuous**: Actions and/or states are floats (e.g., robot arm control).

- **Single-Agent vs Multi-Agent**
  - **Single-Agent**: Most Gym environments are single-agent.
  - **Multi-Agent**: Used in games, robotics, and coordination problems.


### Summary

The **Environment** is the backbone of a simulation setup in machine learning. It:

- Defines the rules and dynamics of the system  
- Interfaces with the agent via observations and rewards  
- Can be reused, extended, or customized for different tasks  

Python libraries like **OpenAI Gym**, **PyBullet**, and **Unity ML-Agents** provide powerful environments for research and production-level machine learning applications.


