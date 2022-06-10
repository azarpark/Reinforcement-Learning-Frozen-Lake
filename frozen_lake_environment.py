################ Environment ################

import numpy as np
import contextlib


# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        current_state = state

        return NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1. / n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            # print(action)
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
         lake =  [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        # self.max_steps = max_steps
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip

        n_states = self.lake.size + 1
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        self.absorbing_state = n_states - 1

        Environment.__init__(self, n_states=n_states, n_actions=n_actions, max_steps=max_steps, pi=pi, seed=seed)

        self.lake_states_mapping = np.arange(self.lake.size).reshape((self.lake.shape[0], self.lake.shape[1]))

        self.probability_transitions = self.get_probability_transitions()

    def get_probability_transitions(self):
        lake = self.lake

        p = []
        for next_state in range(0, lake.size + 1):
            next_state_prob = []
            for state in range(0, lake.size + 1):
                next_state_prob.append(self.get_next_state_prob(next_state, state))
            p.append(next_state_prob)
        p = np.array(p)

        return p

    def get_next_state_prob(self, next_state, state):
        lake_states = self.lake_states_mapping
        s_i, s_j = np.where(lake_states == state)  # coordinates of current state
        lake = self.lake
        # edge cases
        if state == self.absorbing_state and next_state == self.absorbing_state:
            return [1.0] * 4
        elif state == self.absorbing_state and next_state != self.absorbing_state:
            return [0.0] * 4

        elif (self.lake_flat[state] == '#' or self.lake_flat[state] == '$') and next_state == self.absorbing_state:
            return [1.0] * 4
        elif (self.lake_flat[state] == '#' or self.lake_flat[state] == '$') and next_state != self.absorbing_state:
            return [0.0] * 4

        dirs = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # actions mapped as coordinates
        valid_states = []
        for d in dirs:
            if 0 <= d[0] + s_i < lake.shape[0] and 0 <= d[1] + s_j < lake.shape[1]:
                valid_states.append(lake_states[d[0] + int(s_i)][d[1] + int(s_j)])
            else:
                valid_states.append(state)

        valid_states = np.array(valid_states)
        probs_to_ns = []
        prob_per_action = 0.1 / 4.0

        prob = 0.0
        for i in range(4):
            if valid_states[i] == next_state:
                prob = 1.0 - (prob_per_action * len(valid_states[valid_states != next_state]))
            else:
                prob = len(valid_states[valid_states == next_state]) * prob_per_action
            probs_to_ns.append(prob)

        return np.array(probs_to_ns)

    def step(self, action):
        state, reward, done = Environment.step(self, action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):

        return self.probability_transitions[next_state][state][action]

    def r(self, next_state, state, action):

        return 1.0 if state == self.lake.size - 1 and next_state == self.absorbing_state else 0.0

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))


def play(env):
    actions = ['w', 'a', 's', 'd']

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')

        state, r, done = env.step(actions.index(c))

        env.render()
        print('Reward: {0}.'.format(r))
