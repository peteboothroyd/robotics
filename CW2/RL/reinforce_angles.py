import math
import datetime
import pickle
import itertools

import progressbar

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import multivariate_normal

class Agent(object):
  def __init__(self, environment):
    self.NUM_SUPPORT_POINTS = 21
    self.THETA_MIN, self.THETA_MAX = -90, 90
    self.MAX_THETA_STEP = np.deg2rad(9)
    self.THETA_SUPPORT = np.deg2rad(np.linspace(self.THETA_MIN, self.THETA_MAX, num=self.NUM_SUPPORT_POINTS))
    self.NUM_ACTIONS = 5
    
    self.A_SUPPORT = list(itertools.product(np.linspace(-self.MAX_THETA_STEP, self.MAX_THETA_STEP, self.NUM_ACTIONS), repeat=3))
    self.A_SUPPORT_POINTS = len(self.A_SUPPORT)

    self.GAMMA = 0.9

    self.support_states = np.array(list(itertools.product(self.THETA_SUPPORT, self.THETA_SUPPORT, self.THETA_SUPPORT)))
    self.support_values = np.zeros((self.NUM_SUPPORT_POINTS * self.NUM_SUPPORT_POINTS * self.NUM_SUPPORT_POINTS, 1))
    self.environment = environment

  def learn(self):
    try:
      with open ('gp_val', 'rb') as fp:
        print("loaded gp_val file")
        self.gp_val = pickle.load(fp)
        self.support_values = self.gp_val.y_train_
        self.gp_theta_1, self.gp_theta_2, self.gp_theta_3, self.gp_reward = self.learn_dynamics_reward(200)
        return
    except FileNotFoundError:
      print("gp_val file not found")
      self.initialise_values()
      self.learn_value_function()

    self.gp_theta_1, self.gp_theta_2, self.gp_theta_3, self.gp_reward = self.learn_dynamics_reward(200)

    converged_val = False

    while not converged_val:
      k_v_inv_v = self.gp_val.alpha_
      k_v_chol = self.gp_val.L_.dot(self.gp_val.L_.T)
      k_v_inv = np.linalg.inv(k_v_chol)

      v_squared, l1, l2, l3 = np.exp(self.gp_val.kernel_.theta)

      print("v_squared: %s, l1: %s, l2: %s, l3: %s" % (v_squared, l1, l2, l3))
      lengths = np.array([l1, l2, l3]).reshape((3,1))
      lengths_squared = np.square(lengths)

      R = np.zeros((self.NUM_SUPPORT_POINTS ** 3, 1))
      W = np.zeros((self.NUM_SUPPORT_POINTS ** 3, self.NUM_SUPPORT_POINTS ** 3))

      bar = progressbar.ProgressBar(maxval=len(self.support_states), \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
      bar.start()

      for state_index in range(len(self.support_states)):
        bar.update(state_index)

        theta_1, theta_2, theta_3 = self.support_states[state_index]
        state_actions = np.array(list(itertools.product([theta_1], [theta_2], [theta_3], \
                                      np.linspace(-self.MAX_THETA_STEP, self.MAX_THETA_STEP, self.NUM_ACTIONS), \
                                      np.linspace(-self.MAX_THETA_STEP, self.MAX_THETA_STEP, self.NUM_ACTIONS), \
                                      np.linspace(-self.MAX_THETA_STEP, self.MAX_THETA_STEP, self.NUM_ACTIONS))))

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
      
      bar.finish()

      intermediate1 = np.eye(self.NUM_SUPPORT_POINTS**3) - self.GAMMA * W.dot(k_v_inv)
      intermediate2 = np.linalg.inv(intermediate1)
      new_v = intermediate2.dot(R)

      change_in_val = math.sqrt(mean_squared_error(self.support_values, new_v))
      print("rms change in support point values: %s" % (change_in_val))

      self.support_values = new_v
      self.learn_value_function()

      theta_trajectories, _ = self.act(np.array(np.deg2rad([55,0,0])))
      self.environment.animate_links(theta_trajectories)

  def initialise_values(self):
    for i in range(self.NUM_SUPPORT_POINTS * self.NUM_SUPPORT_POINTS * self.NUM_SUPPORT_POINTS):
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
    with open('gp_val%s' % (datetime.datetime.now()), 'wb') as fp:
        pickle.dump(gp_val, fp)
    self.gp_val = gp_val
  
  def learn_dynamics_reward(self, num_dynamics_examples):
    print("Learning dynamics and rewards")
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
    
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25], length_scale_bounds=(1e-3, 20))
    
    gp_theta_1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=1e-5)
    gp_theta_1.fit(start_states, next_theta_1s)

    gp_theta_2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=1e-5)
    gp_theta_2.fit(start_states, next_theta_2s)

    gp_theta_3 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=1e-5)
    gp_theta_3.fit(start_states, next_theta_3s)

    gp_reward = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=1e-5)
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
    current_theta_1, current_theta_2, current_theta_3 = start_thetas
    # initial_val = self.gp_val.predict(np.array((start_thetas)).reshape(1, -1))
    theta_1s, theta_2s, theta_3s, values = [current_theta_1], [current_theta_2], [current_theta_3], [] #initial_val
    predicted_theta_1s, predicted_theta_2s, predicted_theta_3s = [current_theta_1], [current_theta_2], [current_theta_3]

    v_squared, l1, l2, l3 = np.exp(self.gp_val.kernel_.theta)
    print("v_squared: %s, l1: %s, l2: %s, l3: %s" % (v_squared, l1, l2, l3))
    lengths = np.array([l1, l2, l3]).reshape((3,1))
    lengths_squared = np.square(lengths)

    k_v_inv_v = self.gp_val.alpha_
    k_v_chol = self.gp_val.L_.dot(self.gp_val.L_.T)
    k_v_inv = np.linalg.inv(k_v_chol)

    for i in range(50):
      state_actions = np.array(list(itertools.product([current_theta_1], [current_theta_2], [current_theta_3], \
                                    np.linspace(-self.MAX_THETA_STEP, self.MAX_THETA_STEP, self.NUM_ACTIONS), \
                                    np.linspace(-self.MAX_THETA_STEP, self.MAX_THETA_STEP, self.NUM_ACTIONS), \
                                    np.linspace(-self.MAX_THETA_STEP, self.MAX_THETA_STEP, self.NUM_ACTIONS))))

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
          
      (current_theta_1, current_theta_2, current_theta_3), _ = self.environment.next_state_reward((current_theta_1, current_theta_2, current_theta_3), self.A_SUPPORT[max_val_index])

      theta_1s.append(current_theta_1)
      theta_2s.append(current_theta_2)
      theta_3s.append(current_theta_3)
      values.append(np.amax(val_i[0]))
      predicted_theta_1s.append(mu_theta_1[max_val_index])
      predicted_theta_2s.append(mu_theta_2[max_val_index])
      predicted_theta_3s.append(mu_theta_3[max_val_index])

    plt.plot(values)
    plt.savefig("values%s.png" % datetime.datetime.now())
    return np.array([theta_1s, theta_2s, theta_3s]), np.array([predicted_theta_1s, predicted_theta_2s, predicted_theta_3s])

class Environment(object):
  def __init__(self):
    self.angle_limits = np.deg2rad([-90, 90])
    self.L1, self.L2, self.L3 = 0.21, 0.21, 0.21
    # self.y_bands = np.linspace(-(self.L1 + self.L2 + self.L3),(self.L1 + self.L2 + self.L3), num=11)

  def thetas_to_cartesian(self, thetas):
      theta1, theta2, theta3 = thetas[0], thetas[1], thetas[2]
      x = self.L2 * np.sin(theta2 - theta1) - self.L1 * np.sin(theta1) + self.L3 * np.sin(theta3 + theta2 - theta1)
      y = self.L2 * np.cos(theta2 - theta1) + self.L1 * np.cos(theta1) + self.L3 * np.cos(theta3 + theta2 - theta1)
      return x, y

  def next_state_reward(self, current_state, action):
    thetas = np.add(current_state, action)
    bounded_thetas = np.clip(thetas, self.angle_limits[0], self.angle_limits[1])

    current_x, current_y = self.thetas_to_cartesian(current_state)
    next_x, next_y = self.thetas_to_cartesian(bounded_thetas)

    # right_idx = np.searchsorted(self.y_bands, current_y)
    # left_idx = right_idx - 1

    # Only give reward if moving along horizontal line
    # TODO: Test this tolerance, how tight does the band need to be?
    # if next_y > self.y_bands[right_idx] or next_y < self.y_bands[left_idx]:
    #   return current_state, current_x 
    # else:
    #   return bounded_thetas, next_x 
    return bounded_thetas, multivariate_normal.pdf([next_x, next_y], mean=[0.63, 0.3], cov=[0.63, 0.025])
  
  def animate_links(self, theta_trajectories):
    dt = 0.05
    t = np.arange(0.0, 20, dt)

    x1 = - self.L1 * np.sin(theta_trajectories[0])
    y1 = self.L1 * np.cos(theta_trajectories[0])

    x2 = self.L2 * np.sin(theta_trajectories[1] - theta_trajectories[0]) + x1
    y2 = self.L2 * np.cos(theta_trajectories[1] - theta_trajectories[0]) + y1

    x3 = self.L3 * np.sin(theta_trajectories[2] + theta_trajectories[1] - theta_trajectories[0]) + x2
    y3 = self.L3 * np.cos(theta_trajectories[2] + theta_trajectories[1] - theta_trajectories[0]) + y2

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.25, 1.5), ylim=(-0.5, 1.75))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
      line.set_data([], [])
      time_text.set_text('')
      return line, time_text

    def animate(i):
      thisx = [0, x1[i], x2[i], x3[i]]
      thisy = [0, y1[i], y2[i], y3[i]]

      line.set_data(thisx, thisy)
      time_text.set_text(time_template % (i*dt))
      return line, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(x1)), interval=250, blit=False, init_func=init)
    ani.save("trajectories%s.gif"%datetime.datetime.now(), writer='imagemagick', fps=30)
    # plt.scatter(x1,y1, c="r")
    # plt.scatter(x2,y2, c="g")
    # plt.scatter(x3,y3, c="b")
    # plt.show()

if __name__ == "__main__":
  env = Environment()
  agent = Agent(env)
  # agent.test_learn_dynamics_reward()
  agent.learn()
  theta_trajectories, predicted_theta_trajectories = agent.act(np.array(np.deg2rad([55,0,0])))
  env.animate_links(theta_trajectories)
  # env.animate_links(predicted_theta_trajectories)