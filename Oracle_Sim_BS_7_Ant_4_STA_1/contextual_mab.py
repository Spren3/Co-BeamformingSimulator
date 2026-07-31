import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeRegressor

'''
Collection of different contextual multi-armed bandits

Nice explanation of MAB: https://stats.stackexchange.com/questions/291906/can-reinforcement-learning-be-stateless

Author: Zubow (TU Berlin)
'''

class FakeSimBandit:
    def __init__(self, deg_resolution, n_features):
        self.action_to_null_tbl = np.arange(0, 360, deg_resolution) / 360
        self.n_actions = self.action_to_null_tbl.shape[0]
        self.n_features = n_features
        self.min_angular_separation = 5.0 / 360

    def get_reward_from_multiple_actions(self, action_ids, obs):
        # if null angles too close they are treated as single null
        assert len(action_ids) == 2
        if np.abs(self.action_to_null_tbl[action_ids[0]] - self.action_to_null_tbl[action_ids[0]]) < self.min_angular_separation:
            del action_ids[1]

        # just sum of rewards
        return sum([self.get_reward(action, obs) for action in action_ids])

    def get_reward(self, action_id, obs):
        # convert action ids to angles
        null_ang = self.action_to_null_tbl[action_id]

        # direction towards own STA
        beam_angle = obs[0]
        beam_dist = obs[1]

        if np.abs(null_ang - beam_angle) < self.min_angular_separation:
            # infeasible nulling: nulling direction too close to beamform direction
            return -1

        if np.abs(null_ang - beam_angle) < 2*self.min_angular_separation:
            # infeasible nulling: nulling direction too close to beamform direction
            return -0.5

        sta1_angle_reward = 10.0 / np.power((1.01 + np.abs(null_ang - obs[2])), 10)
        sta1_reward = obs[3] * sta1_angle_reward # scaled with distance

        sta2_angle_reward = 10.0 / np.power((1.01 + np.abs(null_ang - obs[4])), 10)
        sta2_reward = obs[5] * sta2_angle_reward # scaled with distance

        # sum reward
        total_reward = sta1_reward + sta2_reward

        return total_reward

    def get_optimal_reward(self, obs):
        # best action
        r = []
        for a in range(self.n_actions):
            r.append(self.get_reward(a, obs))

        return max(r)

class LinUCB:
    def __init__(self, n_actions, n_features, alpha=1.0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha

        # Initialize parameters
        self.A = np.array(
            [np.identity(n_features) for _ in range(n_actions)]
        )  # action covariance matrix
        self.b = np.array(
            [np.zeros(n_features) for _ in range(n_actions)]
        )  # action reward vector
        self.theta = np.array(
            [np.zeros(n_features) for _ in range(n_actions)]
        )  # action parameter vector

    def predict(self, context):
        context = np.array(context)  # Convert list to ndarray
        context = context.reshape(
            -1, 1
        )  # reshape the context to a single-column matrix
        p = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            theta = np.dot(
                np.linalg.inv(self.A[a]), self.b[a]
            )  # theta_a = A_a^-1 * b_a
            theta = theta.reshape(-1, 1)  # Explicitly reshape theta
            p[a] = np.dot(theta.T, context) + self.alpha * np.sqrt(
                np.dot(context.T, np.dot(np.linalg.inv(self.A[a]), context))
            )  # p_t(a|x_t) = theta_a^T * x_t + alpha * sqrt(x_t^T * A_a^-1 * x_t)
        return p

    def update(self, action, context, reward):
        self.A[action] += np.outer(context, context)  # A_a = A_a + x_t * x_t^T
        self.b[action] += reward * context  # b_a = b_a + r_t * x_tx

class NeuralNetwork(nn.Module):
    def __init__(self, n_features):
        super(NeuralNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_features, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layer(x)

class NeuralBandit:
    def __init__(self, n_actions, n_features, learning_rate=0.01):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate

        # Initialize the neural network model for each action
        self.models = [NeuralNetwork(n_features) for _ in range(n_actions)]
        self.optimizers = [
            optim.Adam(model.parameters(), lr=self.learning_rate)
            for model in self.models
        ]
        self.criterion = nn.MSELoss()

    def predict(self, context):
        context_tensor = torch.tensor(context, dtype=torch.float32)  # Convert to tensor
        with torch.no_grad():
            return torch.cat(
                [model(context_tensor).reshape(1) for model in self.models]
            )

    def update(self, action, context, reward):
        self.optimizers[action].zero_grad()
        context_tensor = torch.tensor(context, dtype=torch.float32)  # Convert to tensor
        reward_tensor = torch.tensor(reward, dtype=torch.float32)  # Convert to tensor
        pred_reward = self.models[action](context_tensor)
        loss = self.criterion(pred_reward, reward_tensor)
        loss.backward()
        self.optimizers[action].step()


class DecisionTreeBandit:
    def __init__(self, n_actions, n_features, max_depth=4):
        self.n_actions = n_actions
        self.n_features = n_features
        self.max_depth = max_depth

        # Initialize the decision tree model for each action
        self.models = [
            DecisionTreeRegressor(max_depth=self.max_depth) for _ in range(n_actions)
        ]
        self.data = [[] for _ in range(n_actions)]

    def predict(self, context):
        return np.array(
            [self._predict_for_action(a, context) for a in range(self.n_actions)]
        )

    def _predict_for_action(self, action, context):
        if not self.data[action]:
            return 0.0
        X, y = zip(*self.data[action])
        self.models[action].fit(np.array(X), np.array(y))
        context_np = np.array(context).reshape(
            1, -1
        )  # convert list to NumPy array and reshape
        return self.models[action].predict(context_np)[0]

    def update(self, action, context, reward):
        self.data[action].append((context, reward))

class PureRandomBandit:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def predict(self, context):
        # return np.random.random(self.n_actions)
        return ([0.2,0.5])

    def update(self, action, context, reward):
        # do nothing
        pass

class ContinuousRandomBandit:
    def __init__(self, num_bs, num_antennas):
        self.agent_type = 0
        self.num_bs = num_bs
        self.num_antennas = num_antennas

    def predict(self, context):
        # create N-1 nulls
        return np.random.random(min(self.num_bs - 1, self.num_antennas - 1))

    def update(self, action, context, reward):
        # do nothing
        pass

class OracleHeuristicBandit:
    '''
    Some kind of oracle heuristic ...
    '''
    def __init__(self, agent_id, num_bs, num_antennas):
        self.agent_type = 1
        self.agent_id = agent_id
        self.num_bs = num_bs
        self.num_antennas = num_antennas
        self.max_nulls = min(num_antennas - 1, num_bs - 1)
        self.min_angular_separation = 5.0 / 360 # 5 degree

    def predict(self, context):
        # own beam direction
        beam_dir = context[self.agent_id,0]
        # sort the STAs from OBSS according to their distance; closest first
        ap_context = context[self.agent_id,2:]
        # 1-col = angle; 2-col=pathloss
        sta_ctx = np.reshape(ap_context, (self.num_bs - 1, 2))
        # sort desc
        sorted_sta_ctx = sta_ctx[sta_ctx[:, 1].argsort()[::-1]]
        # STAs to null
        nulls = []
        for nu in range(sorted_sta_ctx.shape[0]):
            # check if bf angle and nulling are not too close
            if np.abs(sorted_sta_ctx[nu, 0] - beam_dir) > self.min_angular_separation:
                nulls.append(sorted_sta_ctx[nu, 0])
            else:
                nulls.append(np.nan)
            if len(nulls) == self.num_antennas - 1:
                break

        return nulls

    def update(self, action, context, reward):
        # do nothing
        pass

class Buffer:
    def __init__(self, n_actions, n_features, buffer_capacity = 100000):
        self.n_actions = n_actions
        self.n_features = n_features
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.features_buffer = np.zeros((self.buffer_capacity, n_features))
        self.action_buffer = np.zeros((self.buffer_capacity, n_features))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, context_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        assert context_tuple[0].shape[0] == self.n_features
        assert context_tuple[1][0].shape[0] == self.n_actions
        assert isinstance(context_tuple[2], float)

        self.features_buffer[index] = context_tuple[0]
        self.action_buffer[index] = context_tuple[1][0] # AZU!!!
        self.reward_buffer[index] = context_tuple[2]

        self.buffer_counter += 1

class NeuralNetwork2(nn.Module):
    def __init__(self, n_features, n_actions):
        super(NeuralNetwork2, self).__init__()
        num_neurons = 128
        self.layer = nn.Sequential(
            nn.Linear(n_features, num_neurons), nn.ReLU(), nn.Linear(num_neurons, n_actions), nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)

class NeuralBandit2:
    def __init__(self, n_actions, n_features, learning_rate=0.01):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate

        # Initialize the neural network model for each action
        self.model = NeuralNetwork2(n_features, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def _prepare_context(self, context):
        context_array = np.asarray(context, dtype=np.float32).reshape(1, -1)
        return torch.from_numpy(context_array)

    def predict(self, context):
        context_tensor = self._prepare_context(context)
        with torch.no_grad():
            pred = self.model(context_tensor)
        return pred.squeeze(0).tolist()

    def update(self, action, context, reward):
        self.optimizer.zero_grad()
        context_tensor = self._prepare_context(context)
        reward_tensor = torch.full((1, self.n_actions), reward, dtype=torch.float32)
        pred_reward = self.model(context_tensor)
        loss = self.criterion(pred_reward, reward_tensor)
        loss.backward()
        self.optimizer.step()