import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)
    counter = 0
    while True and counter <= max_iterations:
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            expected_r = 0.0
            for a in range(0, env.n_actions):
                if a != policy[s]:
                    deterministic_prob = 0.0
                else:
                    deterministic_prob = 1.0

                for ns in range(env.n_states):
                    expected_r += deterministic_prob * env.p(ns, s, a) * (env.r(ns, s, a) + (gamma * value[ns]))
            new_v = expected_r
            value[s] = new_v

            # print(value)

            delta = max(delta, np.abs(v - value[s]))
            # print(delta)

        if delta < theta:
            break
        counter += 1

    return value


def policy_iteration(env, gamma, theta, max_iterations, policy=None, assignment_q=False):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    value = np.zeros(env.n_states, dtype=np.float)
    counter = 0


    while True and counter <= max_iterations:
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        # print(value)
        prev_policy = policy
        policy = policy_improvement(env, value, gamma)

        if np.all(policy == prev_policy):

            if assignment_q:
                print(f"number of iterations to find optimal policy in policy iteration: {counter}")
            break
        counter += 1
    return policy, value


def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        best_action = np.argmax(get_best_action(env, s, gamma, value))
        policy[s] = best_action

    return policy


def get_best_action(env, state, gamma, value):
    actions = np.zeros(4)
    for a in range(env.n_actions):
        expected_return = 0.0
        for next_state in range(env.n_states):
            expected_return += env.p(next_state, state, a) * (env.r(next_state, state, a) + gamma * value[next_state])
        actions[a] = expected_return
    return actions


def value_iteration(env, gamma, theta, max_iterations, value=None, assignment_q=False):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    counter = 0
    while True and counter <= max_iterations:
        delta = 0

        for s in range(env.n_states):
            v = value[s]
            new_v = np.max(get_best_action(env, s, gamma, value))
            value[s] = new_v
            delta = max(delta, abs(v - new_v))

        if delta < theta:
            if assignment_q:
                print(f"number of iterations to find optimal policy in value iteration: {counter}")
            break
        counter += 1

    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        policy[s] = np.argmax(get_best_action(env, s, gamma, value))

    return policy, value
