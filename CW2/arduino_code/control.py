import numpy as np
import serial
import struct
import pickle
import time
import scipy.ndimage.filters as filters

def read_angles():
  try:
    with open ('kinematic_control_thetas', 'rb') as fp:
      print("loaded kinematic_control_thetas file")
      thetas = pickle.load(fp)
      return thetas
  except FileNotFoundError:
      print("kinematic_control_thetas file not found")

  # theta_1s,theta_2s,theta_3s = [],[],[] 

  # with open('left_to_right_angles.csv', 'rb') as csvfile:
  #   lines = csv.reader(csvfile, delimiter=' ', quotechar='|')
    
  #   for row in lines:
  #     theta_1s.append(lines[1])
  #     theta_1s.append(lines[2])
  #     theta_1s.append(lines[3])
    
  #   return np.array([theta_1s, theta_2s, theta_3s])
  
def setup():
  ser = serial.Serial('/dev/cu.usbmodem1411', 9600, timeout=1)
  time.sleep(2)
  return ser

def main():
  angles = np.rad2deg(read_angles())
  ser = setup()
  angles = np.rad2deg(read_angles())
  # filtered_angles = filters.gaussian_filter1d(angles, 1, axis=1)

  for angle in angles:
    command = struct.pack('>BBB', int(angle[0]), int(angle[1] + 90), int(angle[2] + 90))
    print(command)
    ser.write(command)
    time.sleep(0.05)

def test_torques():
  ser = setup()
  command = struct.pack('>BBB', 90, 90, 160)
  ser.write(command)
  time.sleep(2)
  readings = []

  for i in range(160, 80, -1):
    command = struct.pack('>BBB', 90, 90, i)
    ser.write(command)
    vals = ser.readline().split()
    # print(len(vals))

    # if len(vals) >= 1:
    #   # print("A0: %s, A1: %s, A2: %s" % (int(vals[0]), vals[1].decode('ascii'), vals[2].decode('ascii')))# vals[0].decode('ascii')
    #   try:
    #     print("A0: %s" % (int(vals[0])))
    #   except ValueError:
    #     print("error ", vals[0].decode('ascii'))
    # decoded1 = vals[0].decode('ascii')
    # decoded2 = vals[1].decode('ascii')
    # decoded3 = vals[2].decode('ascii')
    # readings.append(data2)
    # print(decoded1, decoded2, decoded3)
    time.sleep(0.01)

if __name__ == "__main__":
  main()
  # test_torques()