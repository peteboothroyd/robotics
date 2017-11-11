import math
import datetime
import pickle
import itertools

import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error

class Agent(object):
  def __init__(self, environment):
    self.NUM_POINTS = 21
    self.THETA_MIN, self.THETA_MAX = -90, 90
    self.THETA_SUPPORT = np.deg2rad(np.linspace(self.THETA_MIN, self.THETA_MAX, num=self.NUM_POINTS))
    self.THETA_DELTA = self.THETA_SUPPORT[1] - self.THETA_SUPPORT[0]
    
    self.A_SUPPORT = list(itertools.product([-self.THETA_DELTA, 0, self.THETA_DELTA], repeat=3))
    self.A_SUPPORT_POINTS = len(self.A_SUPPORT)

    self.GAMMA = 0.9

    self.support_states = np.array(list(itertools.product(self.THETA_SUPPORT, self.THETA_SUPPORT, self.THETA_SUPPORT)))
    self.support_values = np.zeros((self.NUM_POINTS * self.NUM_POINTS * self.NUM_POINTS, 1))
    self.environment = environment

  def learn(self):
    try:
      with open ('support_values', 'rb') as fp:
        print("loaded support_values file")
        self.support_values = pickle.load(fp)
        self.learn_value_function()
        self.gp_theta_1, self.gp_theta_2, self.gp_theta_3, self.gp_reward = self.learn_dynamics_reward(200)
        return
    except FileNotFoundError:
      print("support_values file not found")
      self.initialise_values()

    self.gp_theta_1, self.gp_theta_2, self.gp_theta_3, self.gp_reward = self.learn_dynamics_reward(200)

    converged_val = False

    while not converged_val:
      self.learn_value_function()
      k_v_inv_v = self.gp_val.alpha_
      k_v_chol = self.gp_val.L_.dot(self.gp_val.L_.T)
      k_v_inv = np.linalg.inv(k_v_chol)

      v_squared, l1, l2, l3 = np.exp(self.gp_val.kernel_.theta)

      # #TODO: Change back
      # v_squared, l1, l2, l3 = 0.0831182790874, 1.51456636819, 1.66341480186, 1.98263851055
      print("v_squared: %s, l1: %s, l2: %s, l3: %s" % (v_squared, l1, l2, l3))
      lengths = np.array([l1, l2, l3]).reshape((3,1))
      lengths_squared = np.square(lengths)
      print("lengths squared: ", lengths_squared.shape, lengths_squared)

      R = np.zeros((self.NUM_POINTS ** 3, 1))
      W = np.zeros((self.NUM_POINTS ** 3, self.NUM_POINTS ** 3))

      for state_index in range(len(self.support_states)):
        theta_1, theta_2, theta_3 = self.support_states[state_index]
        state_actions = np.array(list(itertools.product([theta_1], [theta_2], [theta_3], [-self.THETA_DELTA, 0, self.THETA_DELTA], [-self.THETA_DELTA, 0, self.THETA_DELTA], [-self.THETA_DELTA, 0, self.THETA_DELTA])))

        mu_theta_1, std_dev_theta_1 = self.gp_theta_1.predict(state_actions, return_std=True)
        mu_theta_2, std_dev_theta_2 = self.gp_theta_2.predict(state_actions, return_std=True)
        mu_theta_3, std_dev_theta_3 = self.gp_theta_3.predict(state_actions, return_std=True)

        means = np.array([mu_theta_1, mu_theta_2, mu_theta_3])
        var = np.square(np.array([std_dev_theta_1, std_dev_theta_2, std_dev_theta_3]))
        length_squared_plus_var = np.add(var, lengths_squared)
        state_diffs = np.subtract(self.support_states[:,:, np.newaxis], means)
        state_diffs_squared = np.square(state_diffs)
        state_diffs_squared_divided_length_plus_var = np.divide(state_diffs_squared, length_squared_plus_var)
        summed = -0.5 * np.sum(state_diffs_squared_divided_length_plus_var, axis=1)
        exponentiated = np.exp(summed)
        product = np.prod(length_squared_plus_var, axis=0)
        square_root = np.sqrt(product)

        w = np.prod(lengths) * v_squared * np.divide(exponentiated, square_root)
        r = self.gp_reward.predict(state_actions)

        v = self.GAMMA * w.T.dot(k_v_inv_v)
        val_i = r + v.T

        max_val_index = np.argmax(val_i)
        R[state_index][0] = r[max_val_index]
        W[state_index] = w[:,max_val_index]

      intermediate1 = np.eye(self.NUM_POINTS**3) - self.GAMMA * W.dot(k_v_inv)
      intermediate2 = np.linalg.inv(intermediate1)
      new_v = intermediate2.dot(R)

      change_in_val = math.sqrt(mean_squared_error(self.support_values, new_v))
      print("rms change in support point values: %s" % (change_in_val))

      # TODO: Check what an appropriate value for this criteria is
      if change_in_val < 0.001:
        converged_val = True

      self.support_values = new_v
    
      with open('support_values%s' % (datetime.datetime.now()), 'wb') as fp:
        pickle.dump(self.support_values, fp)

  def initialise_values(self):
    for i in range(self.NUM_POINTS * self.NUM_POINTS * self.NUM_POINTS):
      state = self.support_states[i]
      _, reward = env.next_state_reward(state, [0,0,0])
      self.support_values[i][0] = reward
    print("Initialised values.")
  
  def learn_value_function(self):
    print("Learning value function.")
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 100)) * \
             RBF(length_scale=[0.1, 0.1, 0.1], length_scale_bounds=(0.05, 10.0))

    gp_val = GaussianProcessRegressor(kernel=kernel, alpha=0.01) #, n_restarts_optimizer=9
    gp_val = gp_val.fit(self.support_states, self.support_values)
    # predicted_vals = gp_val.predict(self.support_states)
    # rms = math.sqrt(mean_squared_error(predicted_vals, self.support_values))
    # print("val function rms: %s" % rms)
    print("Learnt value function.")
    self.gp_val = gp_val
  
  def learn_dynamics_reward(self, num_dynamics_examples):
    theta_1s = np.deg2rad(np.random.uniform(low=self.THETA_MIN, high=self.THETA_MAX, size=num_dynamics_examples))
    theta_2s = np.deg2rad(np.random.uniform(low=self.THETA_MIN, high=self.THETA_MAX, size=num_dynamics_examples))
    theta_3s = np.deg2rad(np.random.uniform(low=self.THETA_MIN, high=self.THETA_MAX, size=num_dynamics_examples))
    action_indices = np.random.randint(low=0, high=27, size=num_dynamics_examples)
    action_1 = [self.A_SUPPORT[i][0] for i in action_indices]
    action_2 = [self.A_SUPPORT[i][1] for i in action_indices]
    action_3 = [self.A_SUPPORT[i][2] for i in action_indices]

    start_states = list(zip(theta_1s, theta_2s, theta_3s, action_1, action_2, action_3))
    next_theta_1s, next_theta_2s, next_theta_3s, rewards = [], [], [], []

    for state in start_states:
      theta_1, theta_2, theta_3, action = state[0], state[1], state[2], (state[3], state[4], state[5])
      (next_theta_1, next_theta_2, next_theta_3), reward = env.next_state_reward((theta_1, theta_2, theta_3), action)
      next_theta_1s.append(next_theta_1)
      next_theta_2s.append(next_theta_2)
      next_theta_3s.append(next_theta_3)
      rewards.append(reward)
    
    kernel1 = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25], length_scale_bounds=(1e-3, 20))
    gp_theta_1 = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer=9, alpha=1e-5)
    gp_theta_1.fit(start_states, next_theta_1s)

    kernel2 = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25], length_scale_bounds=(1e-3, 20))
    gp_theta_2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=9, alpha=1e-5)
    gp_theta_2.fit(start_states, next_theta_2s)

    kernel3 = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25], length_scale_bounds=(1e-3, 20))
    gp_theta_3 = GaussianProcessRegressor(kernel=kernel3, n_restarts_optimizer=9, alpha=1e-5)
    gp_theta_3.fit(start_states, next_theta_3s)

    kernel4 = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25], length_scale_bounds=(1e-3, 20))
    gp_reward = GaussianProcessRegressor(kernel=kernel4, n_restarts_optimizer=9, alpha=1e-5)
    gp_reward.fit(start_states, rewards)

    # estimated_next_xs = gp_x.predict(start_states)
    # estimated_next_x_dots = gp_x_dot.predict(start_states)
    # x_rms = math.sqrt(mean_squared_error(next_xs, estimated_next_xs))
    # x_dot_rms = math.sqrt(mean_squared_error(next_x_dots, estimated_next_x_dots))
    
    # print("Learning dynamics x_rms: %s, x_dot_rms: %s" % (x_rms, x_dot_rms))
    print("Learnt dynamics and rewards")

    return gp_theta_1, gp_theta_2, gp_theta_3, gp_reward

  def test_learn_dynamics_reward(self):
    num_training_examples = [5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 200, 250]
    num_validation_examples = 1000

    theta_1s = np.deg2rad(np.random.uniform(low=self.THETA_MIN, high=self.THETA_MAX, size=num_validation_examples))
    theta_2s = np.deg2rad(np.random.uniform(low=self.THETA_MIN, high=self.THETA_MAX, size=num_validation_examples))
    theta_3s = np.deg2rad(np.random.uniform(low=self.THETA_MIN, high=self.THETA_MAX, size=num_validation_examples))
    action_indices = np.random.randint(low=0, high=27, size=num_validation_examples)
    action_1 = [self.A_SUPPORT[i][0] for i in action_indices]
    action_2 = [self.A_SUPPORT[i][1] for i in action_indices]
    action_3 = [self.A_SUPPORT[i][2] for i in action_indices]

    start_states = list(zip(theta_1s, theta_2s, theta_3s, action_1, action_2, action_3))
    next_theta_1s, next_theta_2s, next_theta_3s, rewards = [], [], [], []
    theta_1_rmss, theta_2_rmss, theta_3_rmss, reward_rmss = [], [], [], []

    for state in start_states:
      theta_1, theta_2, theta_3, action = state[0], state[1], state[2], (state[3], state[4], state[5])
      (next_theta_1, next_theta_2, next_theta_3), reward = env.next_state_reward((theta_1, theta_2, theta_3), action)
      next_theta_1s.append(next_theta_1)
      next_theta_2s.append(next_theta_2)
      next_theta_3s.append(next_theta_3)
      rewards.append(reward)

    for num_examples in num_training_examples:
      gp_theta_1, gp_theta_2, gp_theta_3, gp_reward = self.learn_dynamics_reward(num_examples)
      mu_theta_1s = gp_theta_1.predict(start_states)
      mu_theta_2s = gp_theta_2.predict(start_states)
      mu_theta_3s = gp_theta_3.predict(start_states)
      ris = gp_reward.predict(start_states)
      
      theta_1_rms = math.sqrt(mean_squared_error(next_theta_1s, mu_theta_1s))
      theta_1_rmss.append(theta_1_rms)
      theta_2_rms = math.sqrt(mean_squared_error(next_theta_2s, mu_theta_2s))
      theta_2_rmss.append(theta_2_rms)
      theta_3_rms = math.sqrt(mean_squared_error(next_theta_3s, mu_theta_3s))
      theta_3_rmss.append(theta_3_rms)
      reward_rms = math.sqrt(mean_squared_error(ris, rewards))
      reward_rmss.append(reward_rms)

      print("Number of training samples: %s, theta_1_rms: %s, theta_2_rms: %s, theta_3_rms: %s, reward_rms: %s " % (num_examples, theta_1_rms, theta_2_rms, theta_3_rms, reward_rms))
    
    theta_1_handle, = plt.plot(num_training_examples, theta_1_rmss, label="THETA 1 RMS")
    theta_2_handle, = plt.plot(num_training_examples, theta_2_rmss, label="THETA 2 RMS")
    theta_3_handle, = plt.plot(num_training_examples, theta_3_rmss, label="THETA 3 RMS")
    reward_handle, = plt.plot(num_training_examples, reward_rmss, label="REWARD RMS")
    plt.legend(handles=[theta_1_handle, theta_2_handle, theta_3_handle, reward_handle])
    plt.show()

  def act(self, start_thetas):
    plt.close()
    theta_1, theta_2, theta_3 = start_thetas
    initial_val = self.gp_val.predict(np.array((start_thetas)).reshape(1, -1))
    path_theta_1, path_theta_2, path_theta_3, val = [theta_1], [theta_2], [theta_3], [initial_val]
    predicted_theta_1s, predicted_theta_2s, predicted_theta_3s = [theta_1], [theta_2], [theta_3]

    for i in range(15):
      max_v, argmax_v = None, None

      k_v_inv_v = self.gp_val.alpha_
      k_v_chol = self.gp_val.L_.dot(self.gp_val.L_.T)
      k_v_inv = np.linalg.inv(k_v_chol)
      v_squared, l1, l2 = np.exp(self.gp_val.kernel_.theta)
      l1, l2 = 1/l1, 1/l2

      for k in range(self.A_SUPPORT_POINTS):
        action = self.A_SUPPORT[k]
        state_action = np.array((current_x, current_x_dot, action)).reshape(1, -1)
        mu_x, std_dev_x = self.gp_x.predict(state_action, return_std=True)
        mu_x_dot, std_dev_x_dot = self.gp_x_dot.predict(state_action, return_std=True)

        ri = math.exp(-0.5 * (pow(0.6-mu_x[0], 2)/(pow(std_dev_x[0], 2) + pow(0.05, 2)) + pow(mu_x_dot[0], 2)/(pow(std_dev_x_dot[0], 2) + pow(0.05, 2)))) / (pow((pow(std_dev_x[0], 2) + pow(0.05, 2))*(pow(std_dev_x_dot[0], 2) + pow(0.05, 2)), 0.5) * 2 * math.pi * 63.66)
        wi = np.zeros((1, self.X_SUPPORT_POINTS * self.X_DOT_SUPPORT_POINTS))
        
        i = 0
        for statej in self.support_states:
          wi[0][i] = v_squared * math.exp(-0.5 * (pow(l1 * (statej[0] - mu_x[0]), 2) / (1 + pow(l1 * std_dev_x[0], 2)) + pow(l2 * (statej[1] - mu_x_dot[0]), 2) / (1 + pow(l2 * std_dev_x_dot[0], 2)))) / pow((pow(l1 * std_dev_x[0], 2) + 1) * (pow(l2 * std_dev_x_dot[0], 2) + 1), 0.5)
          i += 1

        vi = self.GAMMA * np.dot(wi, k_v_inv_v)
        val_i = ri + vi[0][0]

        if max_v is None or val_i > max_v:
          max_v = val_i
          argmax_v = action
          predicted_x, predicted_x_dot = mu_x, mu_x_dot
          
      current_x, current_x_dot = self.environment.next_state((current_x, current_x_dot), argmax_v, visualise=True)

      path_x.append(current_x)
      path_x_dot.append(current_x_dot)
      val.append(max_v)
      predicted_xs.append(predicted_x)
      predicted_x_dots.append(predicted_x_dot)

      if abs(current_x - target_x) < 0.05 and abs(current_x_dot - target_x_dot) < 0.05:
        break
    
    # plt.plot(path_x, path_x_dot, )
    plt.plot(predicted_xs, predicted_x_dots)
    plt.show()

class Environment(object):
  def __init__(self):
    self.angle_limits = np.deg2rad([-90, 90])
    self.L1, self.L2, self.L3 = 0.21, 0.21, 0.21

  def thetas_to_cartesian(thetas):
      theta1, theta2, theta3 = thetas[0], thetas[1], thetas[2]
      x = self.L2 * np.sin(theta2 - theta1) - self.L1 * np.sin(theta1) + self.L3 * np.sin(theta3 + theta2 - theta1)
      y = self.L2 * np.cos(theta2 - theta1) + self.L1 * np.cos(theta1) + self.L3 * np.cos(theta3 + theta2 - theta1)
      return x, y

  def next_state_reward(self, current_state, action):
    thetas = np.add(current_state, action)
    bounded_thetas = np.clip(thetas, self.angle_limits[0], self.angle_limits[1])

    current_x, current_y = thetas_to_cartesian(current_state)
    next_x, next_y = thetas_to_cartesian(bounded_thetas)

    # Only give reward if moving along horizontal line
    # TODO: Test this tolerance, how tight does the band need to be?
    if abs(next_y - current_y) > 0.1:
      return current_state, next_x
    else:
      return bounded_thetas, next_x

if __name__ == "__main__":
  env = Environment()
  agent = Agent(env)
  # agent.test_learn_dynamics_reward()
  agent.learn()
  # agent.act((-0.5, 0))

 # def visualise_value_function(self, maximising_actions=None, R=None, V=None, show_fig=False):
  #   fig = plt.figure()
  #   ax = fig.gca(projection='3d')

  #   X, X_DOT = np.meshgrid(self.X_SUPPORT_FINE, self.X_DOT_SUPPORT_FINE)
  #   predicted_vals = self.gp_val.predict(self.SUPPORT_STATES_FINE).reshape((250, 250))
  #   vals = self.support_values.reshape((self.X_SUPPORT_POINTS, self.X_DOT_SUPPORT_POINTS))
  #   surf = ax.plot_surface(X, X_DOT, predicted_vals, cmap=cm.rainbow, antialiased=True, linewidth=0.001)

  #   # Customize the z axis.
  #   ax.set_zlim(np.amin(vals), np.amax(predicted_vals))
  #   ax.zaxis.set_major_locator(LinearLocator(10))
  #   ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  #   # Add a color bar which maps values to colors.
  #   fig.colorbar(surf, shrink=0.5, aspect=5)
  #   plt.savefig("gp%s.png" % datetime.datetime.now())
    
  #   if maximising_actions is not None:
  #     plt.clf()
  #     X, X_DOT = np.meshgrid(self.X_SUPPORT, self.X_DOT_SUPPORT)
  #     maximising_actions = maximising_actions.reshape((self.X_SUPPORT_POINTS, self.X_DOT_SUPPORT_POINTS))
  #     contour = plt.contourf(X, X_DOT, maximising_actions)
  #     plt.colorbar(contour, shrink=0.5)
  #     plt.savefig("actions%s.png" % datetime.datetime.now())
    
  #   if R is not None:
  #     fig = plt.figure()
  #     ax = fig.gca(projection='3d')
  #     X, X_DOT = np.meshgrid(self.X_SUPPORT, self.X_DOT_SUPPORT)
  #     R = R.reshape((self.X_SUPPORT_POINTS, self.X_DOT_SUPPORT_POINTS))
  #     surf = ax.plot_surface(X, X_DOT, R, cmap=cm.plasma, antialiased=True, linewidth=0.001)
  #     ax.set_zlim(np.amin(R), np.amax(R))
  #     ax.zaxis.set_major_locator(LinearLocator(10))
  #     ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
  #     fig.colorbar(surf, shrink=0.5, aspect=5)
  #     plt.savefig("reward%s.png" % datetime.datetime.now())
    
  #   if V is not None:
  #     fig = plt.figure()
  #     ax = fig.gca(projection='3d')
  #     X, X_DOT = np.meshgrid(self.X_SUPPORT, self.X_DOT_SUPPORT)
  #     V = V.reshape((self.X_SUPPORT_POINTS, self.X_DOT_SUPPORT_POINTS))
  #     surf = ax.plot_surface(X, X_DOT, V, cmap=cm.coolwarm, antialiased=False, linewidth=0)
  #     ax.set_zlim(np.amin(V), np.amax(V))
  #     ax.zaxis.set_major_locator(LinearLocator(10))
  #     ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
  #     fig.colorbar(surf, shrink=0.5, aspect=5)
  #     plt.savefig("value%s.png" % datetime.datetime.now())

  #   if show_fig:
  #     plt.show()