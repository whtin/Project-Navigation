# README
## Project Description
This project is the development and training of a deep reinforcement learning agent with the aim of maximizing the scores obtained in the "Banana Collector" environment. It the first

## The Environment
[The environment](https://github.com/udacity/Value-based-methods#dependencie) is similar to, but not identical to the Banana Collector environment on the [Unity ML-Agents GitHub](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#banana-collector).
- Reward: A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.
- Goal: Collect as many yellow bananas as possible while avoiding blue bananas.
- State space: 37 dimensions which contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.
- Actions space:
  - 0 - move forward
  - 1 - move backward
  - 2 - turn left
  - 3 - turn right

## The Objective
The environment is considered solved when an average score of +13 over 100 consecutive episodes is obtained.

## Getting Started
To set up your python environment to run the code in this Jupyter notebook, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.
	- __Linux__ or __Mac__:
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__:
	```bash
	conda create --name drlnd python=3.6
	activate drlnd
	```

2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
    ```bash
    git clone https://github.com/udacity/Value-based-methods.git
    cd Value-based-methods/python
    pip install .
    ```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

## Instructions
- There are two ways to interact with this Jupyter notebook:
    - Train agents from scratch: Just run all the codes in the notebook. The DQN agent and Double DQN agent will train for 1000 episodes respectively, and the scores will be plotted. Feel free to change the training parameters.
    - Load existing DNN states and scores: Both agents have already been trained for 1000 episodes using the default parameters, and the saved states (and scores) are in the local directories. Please set `load_state==True` in the training function if you want to load the files and perform further training. You should also set `n_episodes` to a value higher than 1000 as the number of episodes already trained are deducted in the training loops.
