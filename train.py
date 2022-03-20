from collections import deque

import numpy as np
import torch

from ddpg_agent import MADDPGAgent


def ddpg(
    env, brain_name, state_size, action_size, num_agents, random_seed, n_episodes=2000
):

    scores = []
    scores_deque = deque(maxlen=100)
    average_scores = []
    agent = MADDPGAgent(state_size, action_size, num_agents, random_seed)

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        episode_score = np.zeros(num_agents)

        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            episode_score += rewards
            if np.any(dones):
                break

        mean_episode_score = np.mean(episode_score)
        scores_deque.append(mean_episode_score)
        scores.append(mean_episode_score)
        average = np.mean(scores_deque)
        average_scores.append(average)

        if i_episode % 100 == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}\tEpisode Score: {:.2f}".format(
                    i_episode, average, mean_episode_score
                ),
                end="\n",
            )

        if np.mean(scores_deque) >= 0.5:
            print(
                f"Enviroment solved in episode={i_episode} avg_score={average:.2f}".format(
                    i_episode=i_episode, avg=average
                )
            )

            torch.save(agent.actor_local.state_dict(), "checkpoint_actor.pth")
            torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth")

            return scores, average_scores

    return scores, average_scores
