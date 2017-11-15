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
import pyglet
from pyglet.window import key

import pickle

import Arm


def plot():
  """A function for plotting an arm, and having it calculate the
  inverse kinematics such that given the mouse (x, y) position it
  finds the appropriate joint angles to reach that point."""

  #left to right
  q=[2.9, -0.1, -0.1] 

  #right to left
#   q=[0.40, 0.1, 0.1] 

  # create an instance of the arm
  arm = Arm.Arm3Link(L=np.array([75, 75, 75]),q=q) #L=np.array([300, 200, 100])
  # make our window for drawin'
  window = pyglet.window.Window()
  thetas = []

  label = pyglet.text.Label(
    'Trajectory (x,y)', font_name='Times New Roman',
    font_size=36, x=window.width//2, y=window.height//2,
    anchor_x='center', anchor_y='center')

  def get_joint_positions():
    """This method finds the (x,y) coordinates of each joint"""

    x = np.array([
      0,
      arm.L[0]*np.cos(arm.q[0]),
      arm.L[0]*np.cos(arm.q[0]) + arm.L[1]*np.cos(arm.q[0]+arm.q[1]),
      arm.L[0]*np.cos(arm.q[0]) + arm.L[1]*np.cos(arm.q[0]+arm.q[1]) +
      arm.L[2]*np.cos(np.sum(arm.q))]) + window.width/2

    y = np.array([
      0,
      arm.L[0]*np.sin(arm.q[0]),
      arm.L[0]*np.sin(arm.q[0]) + arm.L[1]*np.sin(arm.q[0]+arm.q[1]),
      arm.L[0]*np.sin(arm.q[0]) + arm.L[1]*np.sin(arm.q[0]+arm.q[1]) +
      arm.L[2]*np.sin(np.sum(arm.q))])

    return np.array([x, y]).astype('int')

  window.jps = get_joint_positions()
  
  # Left to right
  trajectory_x = list(range(530, window.jps[0][3], -1))
  y = window.jps[1][3]

  # Right to left
#   trajectory_x = list(range(120, window.jps[0][3], 1))
#   y = window.jps[1][3]

  @window.event
  def on_draw():
    window.clear()

    label.draw()
    for i in range(3):
      pyglet.graphics.draw(
        2,
        pyglet.gl.GL_LINES,
        ('v2i', (window.jps[0][i], window.jps[1][i],
             window.jps[0][i+1], window.jps[1][i+1])))

  def update(dt):
    if len(trajectory_x)>0:
      x = trajectory_x.pop(-1)
      label.text = '(x,y) = (%.3f, %.3f)' % (x, y)
      arm.inv_kin([x - window.width/2, y])  # get new arm angles
      window.jps = get_joint_positions()  # get new joint (x,y) positions
      thetas.append((arm.q))
    if len(trajectory_x) == 1:
        with open('kinematic_control_thetas', 'wb') as fp:
            pickle.dump(thetas, fp)
        thetas_dumped = True
  
  pyglet.clock.schedule_interval(update, 0.01)

  pyglet.app.run()
  

plot()
