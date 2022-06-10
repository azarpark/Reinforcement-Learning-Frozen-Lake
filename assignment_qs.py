from frozen_lake_environment import FrozenLake, _printoptions
from tabular_model_based import *
from tabular_model_free import *
from non_tabular_model_free import *


def main():
    seed = 0

    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)

    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations, assignment_q=True)
    env.render(policy, value)

    print('')

    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations, assignment_q=True)
    env.render(policy, value)
    print('')

    print(policy_evaluation(env, policy, gamma, theta, max_iterations))

    # print('# Model-free algorithms')
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5

    print('')

    print('## Sarsa')
    policy, value, q_per_episode = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    episode = 0
    for i in range(max_episodes):
        policy_per_episode = q_per_episode[i].argmax(axis=1)
        if np.all(policy_per_episode == policy):
            episode = i + 1
            break

    print(f"Sarsa finds the optimal policy on episode: {episode}")
    value_from_policy_eval = policy_evaluation(env, policy, gamma, theta, max_iterations)
    env.render(policy, value)
    print('')
    print(f" Value function found by policy evaluation using sarsa's optimal policy:\n {value_from_policy_eval} ")
    print('')

    print('')

    print('## Q-learning')
    policy, value, q_per_episode = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    episode = 0
    for i in range(max_episodes):
        policy_per_episode = q_per_episode[i].argmax(axis=1)
        if np.all(policy_per_episode == policy):
            episode = i + 1
            break

    print(f"Q-learning finds the optimal policy on episode: {episode}")
    value_from_policy_eval = policy_evaluation(env, policy, gamma, theta, max_iterations)
    env.render(policy, value)
    print('')
    print(
        f" Value function found by policy evaluation when using q-learning's optimal policy:\n {value_from_policy_eval} ")
    print('')


def testing_big_lake():
    seed = 0

    # big lake
    lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '#', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '#', '#', '.', '.', '.', '#', '.'],
            ['.', '#', '.', '.', '#', '.', '#', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=64, seed=seed)

    # print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

    # print('# Model-free algorithms')
    max_episodes = 20000
    eta = 0.40
    epsilon = 0.80

    print('')

    print('## Sarsa')
    policy, value, q_per_episode = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)

    print("Hyper-parameters:")
    print(f"max episodes: {max_episodes}, learning rate: {eta}, exploration factor: {epsilon}")
    value_from_policy_eval = policy_evaluation(env, policy, gamma, theta, max_iterations)
    env.render(policy, value)
    print('')
    print("Value function found by policy evaluation using sarsa's optimal policy:")
    with _printoptions(precision=3, suppress=True):
        print(value_from_policy_eval[:-1].reshape(np.array(lake).shape))
        print('')
    print('')

    print('')

    print('# Model-free algorithms')
    max_episodes = 30000
    eta = 0.50
    epsilon = 0.9

    print('## Q-learning')
    policy, value, _ = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    print("Hyper-parameters:")
    print(f"max episodes: {max_episodes}, learning rate: {eta}, exploration factor: {epsilon}")
    value_from_policy_eval = policy_evaluation(env, policy, gamma, theta, max_iterations)
    env.render(policy, value)
    print('')
    print("Value function found by policy evaluation using q learning's optimal policy:")
    with _printoptions(precision=3, suppress=True):
        print(value_from_policy_eval[:-1].reshape(np.array(lake).shape))
        print('')
    print('')


main()
#testing_big_lake()
