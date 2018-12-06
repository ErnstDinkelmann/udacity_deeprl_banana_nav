# coding: utf-8
# Author: Ernst Dinkelmann

# We begin by importing some necessary packages.  If the code cell below returns an error, please revisit the project
# instructions to double-check that you have installed
# [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and
# [NumPy](http://www.numpy.org/).

# Also confirm that you have activated your environment with the necessary packages

from unityagents import UnityEnvironment  # The environment
import numpy as np
from agent import Agent  # Our agent class, which will interact in and learn from the environment
from collections import deque
import matplotlib.pyplot as plt
import torch
import os
import inspect
from argparse import ArgumentParser


# training our agent
def train(n_episodes=1000, max_n_steps=300, eps_start=1.0, eps_end=0.01, eps_decay=0.995, strCheckpointFile='checkpoint.pth'):
    """
    Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_n_steps (int): maximum number of steps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        strCheckpointFile (str): full path to where the model weights should be saved, checkpoint file
    """

    global env
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    num_saves = 0
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score
        last_t = max_n_steps
        for t in range(max_n_steps):
            action = agent.act(state, eps)  # agent returns an epsilon-greedy action based on state
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)  # records experience and learns (depending on settings)
            state = next_state
            score += reward
            if done:
                last_t = t + 1
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tNum steps: {}\tAverage Score: {:.2f}'.format(i_episode, last_t, np.mean(scores_window)))
        # if i_episode % 100 == 0:
        #     print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13:  # win condition in course
            if num_saves == 0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, np.mean(scores_window)))
                print('\nTraining will continue and the checkpoint will be overwritten every 100 episodes')
                print('\nSaving a checkpoint now, you may interrupt code execution with eg Ctrl+C')
                torch.save(agent.qnetwork_local.state_dict(), strCheckpointFile)
            else:
                if i_episode % 100 == 0:
                    print('\nSaving another checkpoint now, you may interrupt code execution with eg Ctrl+C')
                    torch.save(agent.qnetwork_local.state_dict(), strCheckpointFile)
            num_saves += 1

    env.close()

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    # plt.show()
    plt.savefig('training_score_by_episode.png')
    return scores

def view(strCheckpointFile='checkpoint.pth'):
    """
    Load a saved 'checkpoint.pth' file to view a trained agent perform.

    Params
    ======
        strCheckpointFile (str): full path to the saved model weights, checkpoint file
    """

    global env

    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load(strCheckpointFile))
    # watch an trained agent
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score
    while True:
        action = agent.act(state)  # select an action
        # print(action)
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break

    print("Score: {}".format(score))

    env.close()


if __name__ == '__main__':
    # Set up argument parsing for execution from the command line
    # There is only a single argument --mode, which can be either 'train' or 'view'
    # If running in console, comment out the parser lines and uncomment the manual setting of the args dictionary
    parser = ArgumentParser()
    parser.add_argument('--mode', dest='mode', help='train, view', metavar='MODE', default='train')
    args = parser.parse_args()
    # args = parser.parse_args('--mode train'.split())

    # Automatic detection of the directory within which this py file is located will only work if the py file is
    # executed from command line. When executed within a console within an IDE for example, the context is different
    # In that case, just manually specify the strHomeDir as the full absolute directory within which this py file
    # is located.
    strHomeDir = str(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) + '/'
    # strHomeDir = '/home/ernst/Projects/Udacity/DeepRL/deep-reinforcement-learning/p1_navigation/'

    # We also set the full path the Banana Environment file
    # By default it assumes the applicable banana environment files lies within a directory 'Banana_Env', which
    # is located in the directory of the current py file
    #
    # You may download the banana environment as per instructions in the readme file
    # - **Mac**: `"Banana.app"`
    # - **Windows** (x86): `"path/to/Banana_Windows_x86/Banana.exe"`
    # - **Windows** (x86_64): `"path/to/Banana_Windows_x86_64/Banana.exe"`
    # - **Linux** (x86): `"path/to/Banana_Linux/Banana.x86"`
    # - **Linux** (x86_64): `"path/to/Banana_Linux/Banana.x86_64"`
    # - **Linux** (x86, headless): `"path/to/Banana_Linux_NoVis/Banana.x86"`
    # - **Linux** (x86_64, headless): `"path/to/Banana_Linux_NoVis/Banana.x86_64"`
    strBananaEnvFile = strHomeDir + 'Banana_Env/Banana.x86'

    # We also set the directory where to save and read the saved network weights.
    # By default we save/read in strHomeDir
    strCheckpointFile = strHomeDir + 'checkpoint.pth'

    # Change the working directory to the above directory - this just helps for potential relative path imports
    # Again, this will not necessarily work in the case of console execution as a different context potentially applies
    os.chdir(strHomeDir)

    if args.mode == "train":
        print('--mode train initiated')
        # Start the environment.
        # In the case of training, this is set with no_graphics=True to avoid visual gui elements, to speed up training.
        env = UnityEnvironment(
            file_name=strBananaEnvFile,
            no_graphics=True,
        )

        # Environments contain **_brains_** which are responsible for deciding the actions of their associated agents.
        # Here we check for the first brain available, and set it as the default brain we will be controlling from
        # Python.
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        # Instantiate our agent with the implied state and action space sizes
        agent = Agent(state_size=brain.vector_observation_space_size, action_size=brain.vector_action_space_size,
                      seed=0)

        scores = train(strCheckpointFile=strCheckpointFile)

    elif args.mode == 'view':
        print('--mode view initiated')

        # Start the environment.
        # In the case of viewing, this is set with no_graphics=False to see visual gui elements.
        env = UnityEnvironment(
            file_name=strBananaEnvFile,
            no_graphics=False,
        )

        # Environments contain **_brains_** which are responsible for deciding the actions of their associated agents.
        # Here we check for the first brain available, and set it as the default brain we will be controlling from Python.
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        # Instantiate our agent with the implied state and action space sizes
        agent = Agent(state_size=brain.vector_observation_space_size, action_size=brain.vector_action_space_size,
                      seed=0)

        view(strCheckpointFile=strCheckpointFile)
