from collections import deque
import numpy as np
import torch

def dqn(n_episodes, max_t, eps_start, eps_end, eps_decay, env, agent, brain_name, saver):

    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                                             # list containing scores from each episode
    scores_window = deque(maxlen=100)                       # last 100 scores
    eps = eps_start                                         # initialize epsilon
    previous_scores = 10.0
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0

        for t in range(max_t):
            action = agent.act(state, eps)
            action = action.astype(np.int32)
            env_info = env.step(action)[brain_name]         # send the action to the environment
            next_state = env_info.vector_observations[0]    # get the next state
            reward = env_info.rewards[0]                    # get the reward
            done = env_info.local_done[0]                   # see if episode has finished


            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)                         # save most recent score
        scores.append(score)                                # save most recent score
        eps = max(eps_end, eps_decay*eps)                   # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)> previous_scores:
            previous_scores = np.mean(scores_window)
            agent.save('saved/' + saver)
            best_scores = previous_scores
        if i_episode==n_episodes:
            print('\nAgent learning for {:d} episodes!\tBest Average Score: {:.2f}'.format(i_episode, best_scores))
            break
    return scores
