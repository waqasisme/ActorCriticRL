import gym
import numpy as np
from actor_critic import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = Agent(alpha=1e-5, n_actions=env.action_space.n)
    num_episodes = 1800

    filename = f'cartpole-alpha-{agent.alpha}-episodes-{num_episodes}.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(num_episodes):
        observation, done, score = env.reset(), False, 0
        
        while not done:
            action = agent.choose_actions(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'best score %.2f' % best_score)

    x = [i+1 for i in range(num_episodes)]
    plot_learning_curve(x, score_history, figure_file)
            