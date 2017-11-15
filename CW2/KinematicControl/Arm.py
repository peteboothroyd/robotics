'''
Copyright (C) 2013 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import scipy.optimize

class Arm3Link:
    def __init__(self, q=None, L=None):
        """Set up the basic parameters of the arm.
        All lists are in order [shoulder, elbow, wrist].

        q : np.array
            the initial joint angles of the arm
        L : np.array
            the arm segment lengths
        """
        # initial joint angles
        self.q = [2.55, -0.1, -0.1] if q is None else q
        # some default arm positions
        self.q_last = self.q
        # arm segment lengths
        self.L = np.array([1, 1, 1]) if L is None else L

        self.q_bounds = [(0, np.pi), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2)]
        self.max_torques = [3, 3, 3]

    def get_xy(self, q=None):
        """Returns the corresponding hand xy coordinates for
        a given set of joint angle values [shoulder, elbow, wrist],
        and the above defined arm segment lengths, L

        q : np.array
            the list of current joint angles

        returns : list
            the [x,y] position of the arm
        """
        if q is None:
            q = self.q

        x = self.L[0]*np.cos(q[0]) + \
            self.L[1]*np.cos(q[0]+q[1]) + \
            self.L[2]*np.cos(np.sum(q))

        y = self.L[0]*np.sin(q[0]) + \
            self.L[1]*np.sin(q[0]+q[1]) + \
            self.L[2]*np.sin(np.sum(q))

        return [x, y]

    def inv_kin(self, xy):
        """This is just a quick write up to find the inverse kinematics
        for a 3-link arm, using the SciPy optimize package minimization
        function.

        Given an (x,y) position of the hand, return a set of joint angles (q)
        using constraint based minimization, constraint is to match hand (x,y),
        minimize the distance of each joint from it's last postition (q_last).

        xy : tuple
            the desired xy position of the arm

        returns : list
            the optimal [shoulder, elbow, wrist] angle configuration
        """

        def distance_to_default(q, *args):
            """Objective function to minimize
            Calculates the euclidean distance through joint space to the
            default arm configuration. The weight list allows the penalty of
            each joint being away from the resting position to be scaled
            differently, such that the arm tries to stay closer to resting
            state more for higher weighted joints than those with a lower
            weight.

            q : np.array
                the list of current joint angles

            returns : scalar
                euclidean distance to the default arm position
            """
            # weights found with trial and error,
            # get some wrist bend, but not much
            weight = [1, 1, 1]
            return np.sqrt(np.sum([(qi - q_lasti)**2 * wi
                           for qi, q_lasti, wi in zip(q, self.q_last, weight)]))

        def x_constraint(q, xy):
            """Returns the corresponding hand xy coordinates for
            a given set of joint angle values [shoulder, elbow, wrist],
            and the above defined arm segment lengths, L

            q : np.array
                the list of current joint angles
            xy : np.array
                current xy position

            returns : np.array
                the difference between current and desired x position
            """
            diff_x = (self.L[0]*np.cos(q[0]) + self.L[1]*np.cos(q[0]+q[1]) +
                 self.L[2]*np.cos(np.sum(q))) - xy[0]
            return diff_x

        def y_constraint(q, xy):
            """Returns the corresponding hand xy coordinates for
            a given set of joint angle values [shoulder, elbow, wrist],
            and the above defined arm segment lengths, L

            q : np.array
                the list of current joint angles
            xy : np.array
                current xy position
            returns : np.array
                the difference between current and desired y position
            """
            diff_y = (self.L[0]*np.sin(q[0]) + self.L[1]*np.sin(q[0]+q[1]) +
                 self.L[2]*np.sin(np.sum(q))) - xy[1]
            return diff_y

        new_q = scipy.optimize.fmin_slsqp(
            func=distance_to_default,
            x0=self.q,
            eqcons=[x_constraint,
                    y_constraint],
            bounds=self.q_bounds,
            args=(xy,),
            iprint=0)  # iprint=0 suppresses output
        
        self.q_last = self.q
        self.q = new_q
