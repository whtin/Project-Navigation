# Report
## Learning Algorithm

A deep Q-learning with an experience replay mechanism is used in this project.

### Action-value function Q
- The action-value function Q is modeled using a deep neural network with two hidden fully connected layers, each of which has 64 nodes.
- The first two fully connected layers are followed by a ReLU activation function.
- The outputs of the third layer are directly used (without an activation function) as estimations of the action-values.

### Experience replay
- A "replay buffer" object type is defined to add and sample replay memories (i.e. the state=$s_t$, action=$a_t$, reward=$r_t$, next_state=$r_{t+1}$, done=$d_{t+1}$) for each step.
- The replay buffer has a finite memory size (default=10000) and earlier experiences are removed from the buffer as new experiences are added.

###  Training algorithms:

1. Initialize replay memory $D$ with capacity = $N$
2. Initialize action-value function $Q$ with random weights $\theta$
3. Initialize target action-value function $\hat{Q}$ with weights $\hat{\theta}=\theta$
4. For episode = 1 to n_episodes (default=1000):
- Reset environment
-  For t=1 to max_t (default=1000): 
   - With probability $\epsilon$ select a random action at $a_t$
   - Otherwise select $a_t = argmax_a G \left(s_t, a; \theta\right)$
   - Agent execute action $a_t$ and observe $r_t$ and $s_{t+1}$
	- Store transition ($s_t$, $a_t$, $r_t$, $a_{t+1}$, $d_{t+1}$) in $D$
	- For every C (default=4) steps: 
		- Sample random minibatch (default=64) of transitions from $D$
		- Set $$ y_j = \begin{cases} r_j, & \text{if } d_{t+1}=1 \\ r_j + \max \hat Q(s_{j+1}, a'; \theta), & \text{otherwise} \end{cases} $$
		- Perform a gradient descent step on $\left(y_j - Q(s_j, a_j, \theta)^2 \right)$ with respect to network parameters $\theta$
		- Do a soft update $\hat{\theta} = τ * θ + (1 - τ) * \hat{\theta}$

## Plot of Rewards

### Deep Q-Learning Algorithm

The agent is able to receive an average reward (over 100 episodes) of at least +13 after training for 800 episodes.

| Episodes | Average Score |
| --- | --- |
| 100 | 0.97 |
| 200 | 4.52 |
| 300 | 6.69 |
| 400 | 8.29 |
| 500 | 10.55 |
| 600 | 11.86 |
| 700 | 12.68 |
| 800 | 13.18 |
| 900 | 14.39 |
| 1000 | 14.42 |

![dqn scores plot](/dqn_scores_plot.png)

### Double Deep Q-Learning Algorithm

Apart from the plain vanilla DQN, I also introduced a small modification to allow $\hat{\theta}$ and $\theta$ to be used randomly (with equal probabilities) for greedy policies evaluation for state $s_{t+1}$ and the other for action values estimation.

According to [Double DQN paper](https://arxiv.org/pdf/1509.06461.pdf), this modification would decompose the max operation in the target into action selection and action evaluation, and could potentially yields more accurate value estimates and higher scores on several games.

However, in this game (Banana Collector), using the DDQN mechanism does not seem to have a significant impact on the result, only improving the average score from 14.42 to 14.88 after 1000 episodes. Maybe the improvement would be more pronounced after more episodes.

| Episodes | Average Score |
| --- | --- |
| 100 | 0.90 |
| 200 | 4.06 |
| 300 | 6.68 |
| 400 | 9.14 |
| 500 | 11.01 |
| 600 | 11.82 |
| 700 | 12.84 |
| 800 | 13.04 |
| 900 | 14.63 |
| 1000 | 14.88 |

![dqn scores plot](/ddqn_scores_plot.png)

## Ideas for Future Work

- Prioritized Experience Replay: Assign priorities to each experience tuple based on the TD error so that experiences are sampled based on their "importance". This could improve the efficiency of training.
- Learning from Pixels: Currently, the agent learns from state information such as its velocity and ray-based perception of objects around its forward direction. To learn directly from pixels (i.e. the agent's first-person view of the environment), a convolutional neural network has to be adopted as the DQN architecture.
