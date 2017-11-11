import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import pickle

def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

# class Question1(object):
#     def run(self):
#         THETA_ONE_MIN, THETA_ONE_MAX = deg_to_rad(-90.0), deg_to_rad(90.0)
#         THETA_TWO_MIN, THETA_TWO_MAX = deg_to_rad(-120.0), deg_to_rad(160.0)
#         THETA_THREE_MIN, THETA_THREE_MAX = deg_to_rad(-160.0), deg_to_rad(160.0)
#         NUM_POINTS = 150j
        
#         points = np.mgrid[THETA_ONE_MIN:THETA_ONE_MAX:NUM_POINTS, THETA_TWO_MIN:THETA_TWO_MAX:NUM_POINTS, THETA_THREE_MIN:THETA_THREE_MAX:NUM_POINTS].reshape(3,-1).T
        
#         xPoints, yPoints = [], []
        
#         for point in points:
#             x, y = thetas_to_cartesian(point)
#             xPoints.append(x)
#             yPoints.append(y)
        
#         plt.scatter(xPoints, yPoints, marker='.')
#         plt.show()

# class Agent(object):
    # NUM_POINTS = 21
    # THETA_1_MIN, THETA_1_MAX = deg_to_rad(-90.0), deg_to_rad(90.0)
    # THETA_1_SUPPORT = np.linspace(THETA_1_MIN, THETA_1_MAX, num=NUM_POINTS)
    # THETA_1_DELTA = THETA_1_SUPPORT[1] - THETA_1_SUPPORT[0]

    # THETA_2_MIN, THETA_2_MAX = deg_to_rad(-120.0), deg_to_rad(160.0)
    # THETA_2_SUPPORT = np.linspace(THETA_2_MIN, THETA_2_MAX, num=NUM_POINTS)
    # THETA_2_DELTA = THETA_2_SUPPORT[1] - THETA_2_SUPPORT[0]
    
    # THETA_3_MIN, THETA_3_MAX = deg_to_rad(-160.0), deg_to_rad(160.0)
    # THETA_3_SUPPORT = np.linspace(THETA_3_MIN, THETA_3_MAX, num=NUM_POINTS)
    # THETA_3_DELTA = THETA_3_SUPPORT[1] - THETA_3_SUPPORT[0]

#     TOLERANCE = 0.10

#     A_SUPPORT_POINTS = 27
#     A_SUPPORT = cartesian(([-THETA_1_DELTA, 0, THETA_1_DELTA], [-THETA_2_DELTA, 0, THETA_2_DELTA], [-THETA_3_DELTA, 0, THETA_3_DELTA]))

#     def __init__(self, environment):
#         self.state_values = np.zeros((self.NUM_POINTS, self.NUM_POINTS, self.NUM_POINTS))
#         self.environment = environment

#     def learn(self):
#         try:
#             with open ('value_function', 'rb') as fp:
#                 print("loaded value_function file")
#                 self.state_values = pickle.load(fp)
#                 print(len(self.state_values), len(self.state_values[0]), len(self.state_values[0][0]))
#                 return
#         except FileNotFoundError:
#             print("value_function file not found")

#         not_converged = True
#         i = 0

#         while not_converged:
#             i += 1
#             print("Iterations %s" % i, np.max(self.state_values))
#             any_value_changed = False

#             for theta_1 in range(self.NUM_POINTS):
#                 for theta_2 in range(self.NUM_POINTS):
#                     for theta_3 in range(self.NUM_POINTS):
#                         current_theta_1, current_theta_2, current_theta_3 = self.THETA_1_SUPPORT[theta_1], self.THETA_2_SUPPORT[theta_2], self.THETA_3_SUPPORT[theta_3]
#                         max_v_estimate = None

#                         for a in range(self.A_SUPPORT_POINTS):
#                             current_action = self.A_SUPPORT[a]

#                             current_state = np.array([current_theta_1, current_theta_2, current_theta_3])
#                             next_state = self.environment.next_state(current_state, current_action)
#                             next_theta_1, next_theta_2, next_theta_3 = next_state[0], next_state[1], next_state[2]
#                             next_theta_1_index, next_theta_2_index, next_theta_3_index = self.binary_search(self.THETA_1_SUPPORT, next_theta_1), self.binary_search(self.THETA_2_SUPPORT, next_theta_2), self.binary_search(self.THETA_3_SUPPORT, next_theta_3)
                            
#                             instantaneous_reward = self.environment.reward((current_theta_1, current_theta_2, current_theta_3),(next_theta_1, next_theta_2, next_theta_3))
#                             action_val_estimate = instantaneous_reward + self.state_values[next_theta_1_index][next_theta_2_index][next_theta_3_index]
                            
#                             if max_v_estimate is None or action_val_estimate > max_v_estimate:
#                                 max_v_estimate = action_val_estimate

#                         if self.state_values[theta_1][theta_2][theta_3] != max_v_estimate:
#                             any_value_changed = True

#                         self.state_values[theta_1][theta_2][theta_3] = max_v_estimate
            
#             if not any_value_changed:
#                 not_converged = False

#             if i % 25 == 0:
#                 with open('value_function', 'wb') as fp:
#                     pickle.dump(self.state_values, fp)
    
#     def act(self, start_state):
#         start_x, start_y = start_state
#         print("start_x %s start_y %s" % (start_x, start_y))
#         target_x, target_y = -1.0, 0.0

#         min_dist, theta_1_opt, theta_2_opt, theta_3_opt = None, None, None, None

#         for theta_1 in range(self.NUM_POINTS):
#             for theta_2 in range(self.NUM_POINTS):
#                 for theta_3 in range(self.NUM_POINTS):
#                     x, y = thetas_to_cartesian((self.THETA_1_SUPPORT[theta_1], self.THETA_2_SUPPORT[theta_2], self.THETA_3_SUPPORT[theta_3]))
#                     if abs(x - start_x) < self.TOLERANCE and abs(y - start_y) < self.TOLERANCE:
#                         if min_dist is None or self.state_values[theta_1][theta_2][theta_3] < min_dist:
#                             min_dist = self.state_values[theta_1][theta_2][theta_3]
#                             theta_1_opt, theta_2_opt, theta_3_opt = theta_1, theta_2, theta_3

#         current_theta_1, current_theta_2, current_theta_3 = self.THETA_1_SUPPORT[theta_1_opt], self.THETA_2_SUPPORT[theta_2_opt], self.THETA_3_SUPPORT[theta_3_opt]
#         print("1:", current_theta_1, "2:", current_theta_2, "3:", current_theta_3)
#         path_x, path_y = [start_x], [start_y]

#         reached_target = False

#         while not reached_target:
#             max_v, argmax_v = None, None

#             for a in range(self.A_SUPPORT_POINTS):
#                 current_action = self.A_SUPPORT[a]
                
#                 current_state = np.array([current_theta_1, current_theta_2, current_theta_3])
#                 next_state = self.environment.next_state(current_state, current_action)
#                 next_theta_1, next_theta_2, next_theta_3 = next_state[0], next_state[1], next_state[2]
#                 next_theta_1_index, next_theta_2_index, next_theta_3_index = self.binary_search(self.THETA_1_SUPPORT, next_theta_1), self.binary_search(self.THETA_2_SUPPORT, next_theta_2), self.binary_search(self.THETA_3_SUPPORT, next_theta_3)
                
#                 action_val_estimate = self.state_values[next_theta_1_index][next_theta_2_index][next_theta_3_index]
                
#                 if max_v == None or action_val_estimate > max_v:
#                     max_v, argmax_v = action_val_estimate, a
            
#             best_next_theta_1, best_next_theta_2, best_next_theta_3 = self.environment.next_state((current_theta_1, current_theta_2, current_theta_3), self.A_SUPPORT[argmax_v])
#             best_next_theta_1_index, best_next_theta_2_index, best_next_theta_3_index = self.binary_search(self.THETA_1_SUPPORT, best_next_theta_1), self.binary_search(self.THETA_2_SUPPORT, best_next_theta_2), self.binary_search(self.THETA_3_SUPPORT, best_next_theta_3)
#             current_theta_1, current_theta_2, current_theta_3 = self.THETA_1_SUPPORT[best_next_theta_1_index], self.THETA_2_SUPPORT[best_next_theta_2_index], self.THETA_3_SUPPORT[best_next_theta_3_index]

#             x, y = thetas_to_cartesian((current_theta_1, current_theta_2, current_theta_3))
#             path_x.append(x)
#             path_y.append(y)

#             if abs(x - target_x) < self.TOLERANCE and abs(y - target_y) < self.TOLERANCE:
#                 reached_target = True
        
#         plt.plot(path_x, path_y, '-o')
#         plt.show()

#     def binary_search(self, sorted_list, target):
#         min = 0
#         max = len(sorted_list) - 1
#         if target < sorted_list[0]:
#             return 0
#         elif target > sorted_list[-1]:
#             return max
#         while True:
#             if (max - min) == 1:
#                 if abs(sorted_list[max] - target) < abs(target - sorted_list[min]):
#                     return max
#                 else:
#                     return min

#             mid = (min + max) // 2

#             if sorted_list[mid] < target:
#                 min = mid
#             elif sorted_list[mid] > target:
#                 max = mid
#             else:
#                 return mid


# class Environment(object):
#     X_TARGET, Y_TARGET = -1.0, 0.0
#     TOLERANCE = 0.05

#     def next_state(self, current_state, action):
#         thetas = np.add(current_state, action)
#         return thetas

#     def reward(self, previous_state, next_state):
#         x_new, y_new = thetas_to_cartesian(next_state)

#         # In obstacle
#         if 0.75 <= x_new <= 1.05 and 0.0 <= y_new <= 0.5:
#             return -10000000 #-Infinity

#         # Reached target
#         if abs(x_new - self.X_TARGET) < self.TOLERANCE  and abs(y_new - self.Y_TARGET) < self.TOLERANCE:
#             return 0
#         else:
#             x_old, y_old = thetas_to_cartesian(previous_state)
#             distance_moved = math.sqrt((x_new - x_old) ** 2 + (y_new - y_old) ** 2)
            
#             # We haven't reached the target so want to punish not moving
#             if distance_moved == 0:
#                 return -10000000
#             else:
#                 return - distance_moved

class Simulation(object):
    L1, L2, L3 = 0.8, 0.9, 0.6

    # def deg_to_rad(self, degrees):
    #     return degrees * np.pi / 180.0

    def thetas_to_cartesian(self, thetas):
        theta1, theta2, theta3 = thetas[0], thetas[1], thetas[2]
        x = self.L2 * np.sin(theta2 - theta1) - self.L1 * np.sin(theta1) + self.L3 * np.sin(theta3 + theta2 - theta1)
        y = self.L2 * np.cos(theta2 - theta1) + self.L1 * np.cos(theta1) + self.L3 * np.cos(theta3 + theta2 - theta1)
        return x, y

    def calc_theta123(self, x3, y3, thetas):
        theta1, theta2_guess, theta3_guess = thetas[0], thetas[1], thetas[2]
        x1, y1 = - self.L1 * np.sin(theta1), self.L1 * np.cos(theta1)
        x3rel1, y3rel1 = x3 - x1, y3 - y1
        theta3 = np.arccos((x3rel1**2 + y3rel1**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3))
        if np.isnan(theta3):
            print("AAAAHHH")
        
        if theta3_guess < 0:
            theta3 *= -1
        
        converged = False
        theta2 = theta2_guess

        while not converged:
            f_theta2 = self.L2 * np.sin(theta2 - theta1) + self.L3 * np.sin(theta3 + theta2 - theta1) - x3rel1
            f_dot_theta2 = self.L2 * np.cos(theta2 - theta1) + self.L3 * np.cos(theta3 + theta2 - theta1)
            theta_2_new = theta2 - f_theta2 / f_dot_theta2
            error = abs(theta_2_new - theta2)

            if error < 0.001:
                converged = True
            else:
                theta2 = theta_2_new
        
        return theta1, theta2, theta3

    def calc_theta12(self, x_targ, theta1_guess):
        theta2 = np.arccos((x_targ**2 + 0.5**2 - 2) / 2)
        converged = False
        theta1 = theta1_guess

        while not converged:
            f_theta1 = - np.sin(theta1) + np.sin(theta2 - theta1) - x_targ
            f_dot_theta1 = - np.cos(theta1) - np.cos(theta2 - theta1)
            theta_1_new = theta1 - f_theta1 / f_dot_theta1
            error = abs(theta_1_new - theta1)

            if error < 0.001:
                converged = True
            else:
                theta1 = theta_1_new
        
        return theta1, theta2
    
    def get_j_matrix(self, thetas):
        theta1, theta2, theta3 = thetas[0], thetas[1], thetas[2]
        j = np.zeros((2,3))
        j[0][0] = - self.L2 * np.cos(theta2 - theta1) - self.L1 * np.cos(theta1) - self.L3 * np.cos(theta3 + theta2 - theta1)
        j[0][1] = self.L2 * np.cos(theta2 - theta1) + self.L3 * np.cos(theta3 + theta2 - theta1)
        j[0][2] = self.L3 * np.cos(theta3 + theta2 - theta1)
        j[1][0] = self.L2 * np.sin(theta2 - theta1) - self.L1 * np.sin(theta1) + self.L3 * np.sin(theta3 + theta2 - theta1)
        j[1][1] = - self.L2 * np.sin(theta2 - theta1) - self.L3 * np.sin(theta3 + theta2 - theta1)
        j[1][2] = - self.L3 * np.sin(theta3 + theta2 - theta1)
        return j
    
    def simulate(self):
        # Calculate the starting thetas
        thetas = np.radians([-60, 120, -130])
        x_guess, y_guess = self.thetas_to_cartesian(thetas)
        x, y = 1.3, 0
        theta1, theta2, theta3 = self.calc_theta123(x, y, thetas)
        actualx, actualy = self.thetas_to_cartesian((theta1, theta2, theta3))
        print("Starting from (%s, %s)" % (actualx, actualy))
        
        thetas = np.array([theta1, theta2, theta3])
        n_steps = 101
        
        x_trajectory1 = np.linspace(1.3, 1.05, n_steps)
        y_trajectory1 = np.linspace(0.0, 0.5, n_steps)

        x_trajectory2 = np.linspace(1.05, 0.5875, n_steps)
        y_trajectory2 = np.linspace(0.5, 0.95, n_steps)

        x_trajectory = np.concatenate((x_trajectory1, x_trajectory2))
        y_trajectory = np.concatenate((y_trajectory1, y_trajectory2))

        actual_x_trajectory, actual_y_trajectory = [], []
        theta_trajectories = np.zeros((3, n_steps * 4 + 20))
        theta_trajectories[0][0], theta_trajectories[1][0], theta_trajectories[2][0] = theta1, theta2, theta3
        
        for i in range(1, n_steps * 2):
            jQ = self.get_j_matrix(thetas)
            pinv_jQ = np.linalg.pinv(jQ)
            delta_x = np.array([x_trajectory[i] - actualx, y_trajectory[i] - actualy])
            dthetas = np.dot(pinv_jQ, delta_x)
            thetas = np.add(thetas, dthetas.T)
            theta_trajectories[0][i], theta_trajectories[1][i], theta_trajectories[2][i] = thetas[0], thetas[1], thetas[2]
            actualx, actualy = self.thetas_to_cartesian(thetas)
            actual_x_trajectory.append(actualx)
            actual_y_trajectory.append(actualy)
        
        theta_guesses = np.radians([-27, 85, -160])
        x_guess, y_guess = self.thetas_to_cartesian(theta_guesses)
        print("x_guess %s yguess %s" %(x_guess, y_guess))
        x, y = 0.5875, 0.9375
        theta1, theta2, theta3 = self.calc_theta123(x, y, theta_guesses)

        dthetas = np.array([(theta1 - thetas[0])/20, (theta2 - thetas[1])/20, (theta3 - thetas[2])/20])
        for i in range(20):
            thetas = np.add(thetas, dthetas.T)
            theta_trajectories[0][i + 2 * n_steps], theta_trajectories[1][i + 2 * n_steps], theta_trajectories[2][i + 2 * n_steps] = thetas[0], thetas[1], thetas[2]

        thetas = np.array([theta1, theta2, theta3])
        theta_trajectories[0][2 * n_steps + 20], theta_trajectories[1][2 * n_steps + 20], theta_trajectories[2][2 * n_steps + 20] = thetas[0], thetas[1], thetas[2]
        
        delta = 0.01571
        dthetas = np.array([0.0, -delta, 0.0])
        for i in range(1, n_steps):
            thetas = np.add(thetas, dthetas.T)
            
            if abs(thetas[1]) < delta:
                thetas[1] = -delta

            theta_trajectories[0][i + 2 * n_steps + 20], theta_trajectories[1][i + 2 * n_steps + 20], theta_trajectories[2][i + 2 * n_steps + 20] = thetas[0], thetas[1], thetas[2]
            actualx, actualy = self.thetas_to_cartesian(thetas)
            actual_x_trajectory.append(actualx)
            actual_y_trajectory.append(actualy)
        
        x_trajectory4 = np.linspace(actual_x_trajectory[-1], -1.0, n_steps)
        y_trajectory4 = np.linspace(actual_y_trajectory[-1], 0.0, n_steps)

        for i in range(n_steps):
            jQ = self.get_j_matrix(thetas)
            pinv_jQ = np.linalg.pinv(jQ)
            delta_x = np.array([x_trajectory4[i] - actualx, y_trajectory4[i] - actualy])
            dthetas = np.dot(pinv_jQ, delta_x)
            thetas = np.add(thetas, dthetas.T)
            theta_trajectories[0][i + 3 * n_steps + 20], theta_trajectories[1][i + 3 * n_steps + 20], theta_trajectories[2][i + 3 * n_steps + 20] = thetas[0], thetas[1], thetas[2]
            actualx, actualy = self.thetas_to_cartesian(thetas)
            actual_x_trajectory.append(actualx)
            actual_y_trajectory.append(actualy)

        theta1_plot, = plt.plot(np.degrees(theta_trajectories[0]), label='Theta 1')
        theta2_plot, = plt.plot(np.degrees(theta_trajectories[1]), label='Theta 2')
        theta3_plot, = plt.plot(np.degrees(theta_trajectories[2]), label='Theta 3')
        plt.legend(handles=[theta1_plot, theta2_plot, theta3_plot])
        plt.show()

        return theta_trajectories, actual_x_trajectory, actual_y_trajectory

    def animate_links(self, theta_trajectories, xtraj, ytraj):
        dt = 0.05
        t = np.arange(0.0, 20, dt)

        x1 = - self.L1 * np.sin(theta_trajectories[0])
        y1 = self.L1 * np.cos(theta_trajectories[0])

        x2 = self.L2 * np.sin(theta_trajectories[1] - theta_trajectories[0]) + x1
        y2 = self.L2 * np.cos(theta_trajectories[1] - theta_trajectories[0]) + y1

        x3 = self.L3 * np.sin(theta_trajectories[2] + theta_trajectories[1] - theta_trajectories[0]) + x2
        y3 = self.L3 * np.cos(theta_trajectories[2] + theta_trajectories[1] - theta_trajectories[0]) + y2

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

        ani = animation.FuncAnimation(fig, animate, np.arange(1, len(x1)),
                                    interval=25, blit=True, init_func=init)
        rectangle = plt.Rectangle((0.75, 0.0), 0.3, 0.5, fc='r')
        plt.gca().add_patch(rectangle)
        plt.scatter(xtraj, ytraj, marker='.')
        plt.show()
    
    def calc_torq(self, theta1, theta2):
        G, M1, M2, M3, L1, L2 = 9.81, 2, 1.5, 1.0, 1.0, 1.0
        T1 = G * (M2 * (L2/2 * np.sin(theta2 - theta1) - L1 * np.sin(theta1)) + M3 * (L2 * np.sin(theta2 - theta1) - L1 * np.sin(theta1) - M1 * L1 / 2 * np.sin(theta1)))
        T2 = - G * L2 * np.sin(theta2 - theta1) * (M3 + 0.5 * M2)
        return T1, T2
    
    def calculate_torques(self):
        xs = np.linspace(0.5, 1.0, num=100)
        theta1s, theta2s = [0.4189], [2.419]
        t1s, t2s = [], []

        for i in range(1, 100):
            theta1, theta2 = self.calc_theta12(xs[i], theta1s[i-1])
            theta1s.append(theta1)
            theta2s.append(theta2)
            t1, t2 = self.calc_torq(theta1, theta2)
            t1s.append(t1)
            t2s.append(t2)

        t1_plot, = plt.plot(t1s, label='T1')
        t2_plot, = plt.plot(t2s, label='T2')
        plt.legend(handles=[t1_plot, t2_plot])
        plt.show()
        print(max(t1s), min(t2s))


if __name__ == "__main__":
    # q1 = Question1()
    # q1.run()
    # env = Environment()
    # agent = Agent(env)
    # agent.learn()
    # agent.act((1.3, 0))
    sim = Simulation()
    sim.calculate_torques()
    # theta_trajectories, xtraj, ytraj = sim.simulate()
    # sim.animate_links(theta_trajectories, xtraj, ytraj)
