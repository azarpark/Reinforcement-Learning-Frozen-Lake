import numpy as np


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))
    q_per_episode = np.zeros((max_episodes, env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        epsilon_i = epsilon[i]
        a = get_epsilon_greedy_action(env, q, epsilon_i, s, random_state)
        done = False
        while not done:
            next_state, r, done = env.step(a)
            # select the next action based on an epsilon greedy policy
            next_action = get_epsilon_greedy_action(env, q, epsilon_i, next_state, random_state)
            td_error = r + (gamma * q[next_state][next_action]) - q[s][a]
            q[s][a] = q[s][a] + eta[i] * td_error
            s = next_state
            a = next_action

        q_per_episode[i] = q

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value, q_per_episode


def get_epsilon_greedy_action(env, q, epsilon, state, random_state):
    if random_state.uniform(0, 1) < epsilon:
        action = random_state.choice(env.n_actions)
    else:
        # arbitrarily break ties where there are actions which result in the same q value for a given state
        q_values = q[state]

        actions = np.where(q_values == q_values.max())[0]
        action = random_state.choice(actions)

    return action


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))
    q_per_episode = np.zeros((max_episodes, env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        done = False
        epsilon_i = epsilon[i]
        while not done:
            a = get_epsilon_greedy_action(env, q, epsilon_i, s, random_state)
            next_state, r, done = env.step(a)

            q_values = q[next_state]
            # greedily select the next action
            next_action = q_values.argmax()

            td_error = r + (gamma * q[next_state][next_action]) - q[s][a]
            q[s][a] = q[s][a] + eta[i] * td_error
            s = next_state
        q_per_episode[i] = q

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value, q_per_episode
