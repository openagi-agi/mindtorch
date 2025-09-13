import numpy as np
import random
from collections import defaultdict

class MarkovChain:
    def __init__(self, n=1):
        self.n = n
        self.model = defaultdict(list)

    def fit(self, tokens):
        for i in range(len(tokens) - self.n):
            prefix = tuple(tokens[i:i+self.n])
            next_token = tokens[i+self.n]
            self.model[prefix].append(next_token)

    def generate(self, max_length=20, seed=None):
        if seed is None:
            seed = random.choice(list(self.model.keys()))
        output = list(seed)
        for _ in range(max_length):
            next_tokens = self.model.get(tuple(output[-self.n:]), None)
            if not next_tokens:
                break
            output.append(random.choice(next_tokens))
        return output

class HMM:
    def __init__(self, n_states, n_obs):
        self.n_states = n_states
        self.n_obs = n_obs
        self.start_probs = np.ones(n_states) / n_states
        self.trans_probs = np.ones((n_states, n_states)) / n_states
        self.emit_probs = np.ones((n_states, n_obs)) / n_obs

    def fit(self, obs_seq, state_seq):
        for i in range(len(obs_seq) - 1):
            self.trans_probs[state_seq[i], state_seq[i+1]] += 1
            self.emit_probs[state_seq[i], obs_seq[i]] += 1
        self.trans_probs /= self.trans_probs.sum(axis=1, keepdims=True)
        self.emit_probs /= self.emit_probs.sum(axis=1, keepdims=True)

    def viterbi(self, obs_seq):
        T = len(obs_seq)
        dp = np.zeros((T, self.n_states))
        path = np.zeros((T, self.n_states), dtype=int)

        dp[0] = np.log(self.start_probs) + np.log(self.emit_probs[:, obs_seq[0]])

        for t in range(1, T):
            for j in range(self.n_states):
                prob = dp[t-1] + np.log(self.trans_probs[:, j]) + np.log(self.emit_probs[j, obs_seq[t]])
                path[t, j] = np.argmax(prob)
                dp[t, j] = np.max(prob)

        best_path = [np.argmax(dp[-1])]
        for t in range(T-1, 0, -1):
            best_path.append(path[t, best_path[-1]])
        return best_path[::-1]
