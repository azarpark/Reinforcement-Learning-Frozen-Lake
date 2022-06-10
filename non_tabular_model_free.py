import numpy as np


class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()  # feature vector of state

        q = features.dot(theta)  # q value for s

        epsilon_i = epsilon[i]
        a = get_epsilon_greedy_action(env, q, epsilon_i, random_state) #choose the action with the best predicted q e-greedily
        done = False

        while not done:
            next_state, r, done = env.step(a)
            delta = r - q[a] #the actual return of executing a - predicted return of a

            q = next_state.dot(theta)
            next_action = get_epsilon_greedy_action(env, q, epsilon_i, random_state)

            delta = delta + gamma * q[next_action] #the error + discounted return
            theta = theta + (eta[i] * delta * features[a]) #update theta for a in the given state in the direction of error.
            features = next_state
            a = next_action

    return theta


def get_epsilon_greedy_action(env, q, epsilon, random_state):
    q_values = q
    if random_state.uniform(0, 1) < epsilon:
        action = random_state.choice(env.n_actions)
    else:
        # arbitrarily break ties where there are actions which result in the same q value for a given state
        actions = np.where(q_values == q_values.max())[0]
        action = random_state.choice(actions)
    return action


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        done = False
        epsilon_i = epsilon[i]

        while not done:
            a = get_epsilon_greedy_action(env, q, epsilon_i, random_state)
            next_state, r, done = env.step(a)

            delta = r - q[a]
            q = next_state.dot(theta)
            # greedily choose next action
            next_action = q.argmax()
            delta = delta + gamma * q[next_action]
            theta = theta + (eta[i] * delta * features[a])
            features = next_state

    return theta
