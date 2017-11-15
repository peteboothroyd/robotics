#!/usr/bin/python
import numpy
from scipy.misc import toimage
import math
import Queue
from numpy.linalg import inv 
from scipy.misc import imsave
START = 0
END = 1
X_SIZE = 500
Y_SIZE = 250

SIZE_RATIO = 100

OFFSET_X = X_SIZE / 2
L_ONE = .8 * SIZE_RATIO
L_TWO = .8 * SIZE_RATIO
L_THREE = .8 * SIZE_RATIO
COLOUR = 2

X_INDEX = 0
Y_INDEX = 1
ANGLE_ONE_INDEX = 0
ANGLE_TWO_INDEX = 1
ANGLE_THREE_INDEX = 2

DIAGNAL_VALID = False

Q_ONE = [-90, 90]
Q_TWO = [-90, 90]
Q_THREE = [-90, 90]

# START_X = int(1.3 * SIZE_RATIO) + OFFSET_X
# START_Y = Y_SIZE
field = numpy.zeros(shape=(Y_SIZE,X_SIZE))

one_field = numpy.zeros(shape=(Y_SIZE,X_SIZE))
two_field = numpy.zeros(shape=(Y_SIZE,X_SIZE))
three_field = numpy.zeros(shape=(Y_SIZE,X_SIZE))

ONE = 1

ORIGIN_X = OFFSET_X
ORIGIN_Y = Y_SIZE

START_OFFSET_DEGREE = 90

def get_key(x, y):
		return str(x) + '_' + str(y)

#check and make sure location is in bounds
def in_bounds(x, y):
	#-1 for 0 based indexing
	x = int(x - ONE)
	y = int(y - ONE)

	return (x < X_SIZE and x >= 0 and y >= 0 and y < Y_SIZE)

#calculate and store data for a line
class LineData:

	def __init__(self, start_x, start_y, line_length, angle_one, angle_two, angle_three, line_type, line_one_points, line_two_points):
		self.start_x = start_x
		self.start_y = start_y
		self.angle_one = angle_one
		self.angle_two = angle_two
		self.angle_three = angle_three
		self.line_type = line_type
		self.line_length = line_length
		self.end_x = None
		self.end_y = None
		self.line_points = self.iterate_through_line(self.start_x, self.start_y, self.angle_one, self.angle_two, self.angle_three, self.line_length, self.line_type, line_one_points, line_two_points)

	def valid(self):
		return self.line_points != None

	def crossed(self, current_x, current_y, start_x, start_y, line_one_points, line_two_points):
		key = get_key(current_x, current_y)
		start_point = (current_x == start_x and current_y == start_y)

		return (not start_point and (key in line_one_points or key in line_two_points))
		

	def iterate_through_line(self, offset_x, offset_y, angle_one, angle_two, angle_three, line_length, line_type, line_one_points, line_two_points):
		points = {}

		try:

			for number in range(START, int(round(line_length))):
				final_angle = (-1 * angle_one) + angle_two + angle_three

				self.end_x = int(round(offset_x + math.sin(math.radians(final_angle)) * number))
				self.end_y = int(round(offset_y - math.cos(math.radians(final_angle)) * number))

				if in_bounds(self.end_x, self.end_y) and not self.crossed(self.end_x, self.end_y, offset_x, offset_y, line_one_points, line_two_points):

					key = get_key(self.end_x, self.end_y)
					points[key] = True

				else:
					raise Exception('Invalid point: ' + str(self.end_x) + ' ' + str(self.end_y))
		except Exception as e:
			# print('e: ' + str(e))
			# print('in bounds: ' + str(in_bounds(self.end_x, self.end_y)))
			# print('crossed: ' +  str(self.crossed(self.end_x, self.end_y, offset_x, offset_y, line_one_points, line_two_points)))
			points = None


		return points

#this will find the all sets of angles for each valid x,y location
def get_all_angles(start_x, start_y, degree_one_min, degree_one_max, degree_two_min, degree_two_max, degree_three_min, degree_three_max, l_one, l_two, l_three):
	angles = {}

	for degree_one in range(degree_one_min, degree_one_max + END):
		line_one_data = LineData(start_x, start_y, l_one, degree_one, 0, 0, 'LINE_ONE', {}, {})
		print('on degree: ' + str(degree_one))


		if line_one_data.valid():
			one_field[line_one_data.end_y - ONE][line_one_data.end_x - ONE] = COLOUR
	
			for degree_two in range(degree_two_min, degree_two_max + END):
				line_two_data = LineData(line_one_data.end_x, line_one_data.end_y, l_two, degree_one, degree_two, 0, 'LINE_TWO', line_one_data.line_points, {})

				if line_two_data.valid():
					two_field[line_two_data.end_y - ONE][line_two_data.end_x - ONE] = COLOUR

					for degree_three in range(degree_three_min, degree_three_max + END):
						# print('degree_one: ' + str(degree_one) + ' degree_two: ' + str(degree_two) + ' degree_three: ' + str(degree_three))
						line_three_data = LineData(line_two_data.end_x, line_two_data.end_y, l_three, degree_one, degree_two, degree_three, 'LINE_THREE', line_one_data.line_points, line_two_data.line_points)
						
						if line_three_data.valid():
							key = get_key(line_three_data.end_x, line_three_data.end_y)

							three_field[line_three_data.end_y - ONE][line_three_data.end_x - ONE] = COLOUR

							key = get_key(line_three_data.end_x, line_three_data.end_y)
							current_angles = []

							if key in angles:
								current_angles = angles[key]
							
							current_angles.append(','.join([str(degree_one), str(degree_two), str(degree_three)]))

							# print('current_angles: ' + str(current_angles))
							angles[key] = current_angles
	return angles

#this will hold the search data
class SearchData:

	def __init__(self, start_x, start_y, end_x, end_y, path):
		self.start_x = start_x
		self.start_y = start_y
		self.end_x = end_x
		self.end_y = end_y
		self.path = path
		self.next_locations = self.build_next_locations(path)

	def at_end(self, x,y):
		return x == self.end_x and y == self.end_y

	def build_next_locations(self, path):
		next_locations = {}

		count = 0
		for location in path:
			x = location[X_INDEX]
			y = location[Y_INDEX]

			count+=1

			if count < len(path):
				key = get_key(x, y)
				next_location = path[count]
				next_x = next_location[X_INDEX]
				next_y = next_location[Y_INDEX]
				next_locations[key] = [next_x, next_y]

		return next_locations

	def get_next_location(self, x,y):
		key = get_key(x, y)

		if key in self.next_locations:
			return self.next_locations[key]
		else:
			return None

#write the x,y location and all angles that can get to that location
def write_data(dict_data):
	out_file = open('output.txt','w') 
	for key, value in dict_data.iteritems():
		locations = key.split('_')
		x = locations[START]
		y = locations[END]

		out_file.write(x + ',' + y + ',' + ','.join(value) + '\n') 
	out_file.close() 

#load the data back from file
def load_data():

	dict_data = {}

	ys = numpy.zeros((2, 251))
	ys[0,:] = 500
	ys[1,:] = 0

	min_x, max_x = 500, 0

	file = open('output.txt','r') 
	for line in file: 
		line_data = line.split(',')
		x = line_data[START]
		y = line_data[END]

		if ys[0, int(y)] > int(x):
			ys[0, int(y)] = int(x)
		if ys[1, int(y)] < int(x):
			ys[1, int(y)] = int(x)

		if int(y) == 55:
			if int(x) > max_x:
				max_x = int(x)
			if int(x) < min_x:
				min_x = int(x)

		angle_one = None
		angle_two = None
		angle_three = None
		angles = []

		primary_counter = 0
		counter = 0
		for item in line_data:
			if primary_counter == X_INDEX:
				x = item
			elif primary_counter == Y_INDEX:
				y = item
			elif counter == ANGLE_ONE_INDEX:
				angle_one = int(item.strip())
				counter+=1
			elif counter == ANGLE_TWO_INDEX:
				angle_two = int(item.strip())
				counter+=1
			elif counter == ANGLE_THREE_INDEX:
				angle_three = int(item.strip())
				counter = 0
				angles.append([angle_one, angle_two, angle_three])
				# angles.append(','.join([str(angle_one), str(angle_two), str(angle_three)]))

			primary_counter+=1

		key = get_key(x, y)
		dict_data[key] = angles
	
	print(ys[1] - ys[0], min_x, max_x)

	return dict_data

#store angle data
class AngleData:

	def __init__(self, x, y, angle_one, angle_two, angle_three):
		self.x = x
		self.y = y
		self.angle_one = angle_one
		self.angle_two = angle_two
		self.angle_three = angle_three

#cost will be the difference between angles
def calculate_new_cost(prev_angle_one, prev_angle_two, prev_angle_three, angle_one, angle_two, angle_three):
	if prev_angle_one == None or prev_angle_two == None or prev_angle_three == None:
		return 0
	else:
		return (math.fabs(prev_angle_one - angle_one) + math.fabs(prev_angle_two - angle_two) + math.fabs(prev_angle_three - angle_three))

#uniform cost search helper
def add_angle_data(prev_angle_one, prev_angle_two, prev_angle_three, cost, q, angles, x, y, angle_path, visited_angles):
	key = get_key(x, y)

	if key not in loaded_angles:
		raise Exception('no infomation for location ' + str(key))

	angles = loaded_angles[key]
	for angle in angles:
		angle_one, angle_two, angle_three = angle


		key = str(angle_one) + '_' + str(angle_two) + '_' + str(angle_three)
		if key not in visited_angles:

			angle_data = AngleData(x, y, angle_one, angle_two, angle_three)
			new_angle_path = angle_path[:]
			new_angle_path.append([angle_one, angle_two, angle_three])

			new_cost = calculate_new_cost(prev_angle_one, prev_angle_two, prev_angle_three, angle_one, angle_two, angle_three)

			
			# location_data = [angle_data.x, angle_data.y, angle_data, angle_path]

			# print('first cost: ' + str(cost))
			# print('first location_data: ' + str(location_data))
			# print('***************')

			# if new_cost < 3.5:
			# 	current_cost = -1 * len(new_angle_path) + new_cost
			current_cost = cost + new_cost
			q.put((current_cost, angle_data.x, angle_data.y, angle_data, new_angle_path))
			visited_angles[key] = True

#uniform cost search - find the path with the shortest path
def find_angle_path(loaded_angles, path_data):
	q = Queue.PriorityQueue()
	initial_cost = 0

	best = 0
	visited_angles = {}

	add_angle_data(None, None, None, initial_cost, q, loaded_angles, path_data.start_x, path_data.start_y, [], visited_angles)

	while not q.empty():
		cost, x, y, angle_data, angle_path = q.get()

		if len(angle_path) > best:
			best = len(angle_path)
			# print('best: ' + str(best) + ' length: ' + str(q.qsize()))
			

		# print('cost: ' + str(cost))
		# print('location_data: ' + str(location_data))
		# print('***************')

		# x, y, angle_data, angle_path = location_data

		if path_data.at_end(x, y):
			print('found angle path solution! ' + str(len(angle_path)))
			return angle_path
		else:
			next_x, next_y = path_data.get_next_location(x, y)

			add_angle_data(angle_data.angle_one, angle_data.angle_two, angle_data.angle_three, cost, q, loaded_angles, next_x, next_y, angle_path, visited_angles)

	raise Exception('Could not find angle path')

#returns true if diagnal
def is_diagnal(x, y):
	diagnal_values = {}
	diagnal_values['-1_-1'] = True
	diagnal_values['-1_1'] = True
	diagnal_values['1_-1'] = True
	diagnal_values['1_1'] = True
	key = get_key(x, y)

	return key in diagnal_values

#breath first search between two points
def find_x_y_path(start_x, start_y, end_x, end_y, loaded_angles):
	start_key = get_key(start_x, start_y)
	end_key = get_key(end_x, end_y)

	if start_key not in loaded_angles:
		raise Exception('Could not find start location - ' + start_key + ' is not a valid location')

	if end_key not in loaded_angles:
		raise Exception('Could not find end location - ' + end_key + ' is not a valid location')

	visited = {}
	added = {}

	q = Queue.Queue()

	q.put([start_x, start_y, [[start_x, start_y]]])
	search_data = None

	solved = False
	while not solved and not q.empty():
		visited_x, visited_y, path = q.get()

		visited_key = get_key(visited_x, visited_y)


		if visited_key not in visited:
			visited[visited_key] = True

			#-1 for 0 based indexing
			if(visited_x == end_x and visited_y == end_y):
				solved = True
				print('solved x y path')

				search_data = SearchData(start_x, start_y, end_x, end_y, path)
			else:

				for y_offset in range(-1, 2):
					for x_offset in range(-1, 2):
						if DIAGNAL_VALID or not is_diagnal(x_offset, y_offset):
							current_y = visited_y + y_offset
							current_x = visited_x + x_offset
							current_key = get_key(current_x, current_y)

							if current_key not in added:
								added[current_key] = True

							if in_bounds(current_x, current_y) and current_key not in visited and current_key in loaded_angles:
								new_path = path[:]
								new_path.append([current_x, current_y])
								q.put([current_x, current_y, new_path])
		
	if not solved:
		raise Exception('Could not find a valid path from ' + str(start_x) + ' ' + str(start_y) + ' to ' + str(end_x) + ' ' + str(end_y))

	return search_data

#write array to image file
def write_image(points, image_array):
	for point in points:
		x, y = point.split('_')
		image_array[int(y) - ONE][int(x) - ONE] = COLOUR

#print out solution images
def create_solution_images(angle_data, start_x, start_y, l_one, l_two, l_three, folder):
	out_file = open('/Users/peterboothroyd/Documents/IIB/4M20/CW2/' + folder + '/' + folder + '_angles.csv','w') 

	t = 0
	for angles in angle_data:
		if True or t % 5 == 0:
			angle_one, angle_two, angle_three = angles
			
			line_one_data = LineData(start_x, start_y, l_one, angle_one, 0, 0, 'LINE_ONE', {}, {})
			line_two_data = LineData(line_one_data.end_x, line_one_data.end_y, l_two, angle_one, angle_two, 0, 'LINE_TWO', line_one_data.line_points, {})
			line_three_data = LineData(line_two_data.end_x, line_two_data.end_y, l_three, angle_one, angle_two, angle_three, 'LINE_THREE', line_one_data.line_points, line_two_data.line_points)
			
			if line_one_data.valid() and line_two_data.valid() and line_three_data.valid():
				out_file.write(str(t) + ',' + str(angle_one) + ',' + str(angle_two) + ',' + str(angle_three) + '\n') 
				current_line = numpy.ones(shape=(Y_SIZE,X_SIZE))
				write_image(line_one_data.line_points.keys(), current_line)
				write_image(line_two_data.line_points.keys(), current_line)
				write_image(line_three_data.line_points.keys(), current_line)
				imsave('/Users/peterboothroyd/Documents/IIB/4M20/CW2/' + folder + '/' + str(format(t, '03')) + '.png', current_line)

			t+=1		

	out_file.close() 

#helper to find start and end 
def find_start_and_end_locations(loaded_angles):
	start_end_locations = {}

	
	x, y = loaded_angle.split('_')
	x = int(x)

	if y in start_end_locations:
		x_min, x_max = start_end_locations[y]
		if x < x_min:
			x_min = x

		if x > x_max:
			x_max = x

		start_end_locations[y] = [x_min, x_max]

	else:
		start_end_locations[y] = [x,x]

	return start_end_locations

#helper to find start and end 
def find_max_min_y(x_input, loaded_angles):
	min_y = None
	max_y = None

	for loaded_angle in loaded_angles.keys():
		x, y = loaded_angle.split('_')

		if x_input == int(x):
			actual_y = 250 - int(y)

			if min_y == None or actual_y < min_y:
				min_y = actual_y 

			if max_y == None or actual_y > max_y:
				max_y = actual_y 

	return min_y, max_y

#smooth the angles
def interpolate_angles(angle_data):
	new_angle_data = []
	prev_angle_one = None
	prev_angle_two = None
	prev_angle_three = None

	for angles in angle_data:
		angle_one, angle_two, angle_three = angles

		if prev_angle_one == None:
			prev_angle_one = angle_one

		if prev_angle_two == None:
			prev_angle_two = angle_two

		if prev_angle_three == None:
			prev_angle_three = angle_three

		diff_one = math.fabs(prev_angle_one - angle_one)
		diff_two = math.fabs(prev_angle_two - angle_two)
		diff_three = math.fabs(prev_angle_three - angle_three)

		while(diff_one > 1 or diff_two > 1 or diff_three > 1):
			if diff_one > 1:
				if angle_one - prev_angle_one > 0:
					prev_angle_one+=1
				else:
					prev_angle_one-=1

			if diff_two > 1:
				if angle_two - prev_angle_two > 0:
					prev_angle_two+=1
				else:
					prev_angle_two-=1

			if diff_three > 1:
				if angle_three - prev_angle_three > 0:
					prev_angle_three+=1
				else:
					prev_angle_three-=1

			diff_one = math.fabs(prev_angle_one - angle_one)
			diff_two = math.fabs(prev_angle_two - angle_two)
			diff_three = math.fabs(prev_angle_three - angle_three)
			new_angle_data.append([prev_angle_one, prev_angle_two, prev_angle_three])

		prev_angle_one = angle_one
		prev_angle_two = angle_two
		prev_angle_three = angle_three

		new_angle_data.append([angle_one, angle_two, angle_three])

	return new_angle_data

#main
if __name__ == "__main__":
	print('running...')

	# #once this is run and saved to a file, there is no need solve again 
	# print('finding angles')
	# all_angles = get_all_angles(ORIGIN_X, ORIGIN_Y, Q_ONE[START], Q_ONE[END], Q_TWO[START], Q_TWO[END], Q_THREE[START], Q_THREE[END], L_ONE, L_TWO, L_THREE)

	# print('saving angles to file')
	# write_data(all_angles)

	print('loading data')
	loaded_angles = load_data()

	#left to right
	path_data = find_x_y_path(100, 65, 400, 65, loaded_angles)
	angle_data = find_angle_path(loaded_angles, path_data)
	angle_data = interpolate_angles(angle_data)
	create_solution_images(angle_data, ORIGIN_X, ORIGIN_Y, L_ONE, L_TWO, L_THREE, 'left_to_right')

	# #right to left
	# path_data = find_x_y_path(440, 125, 60, 125, loaded_angles)
	# angle_data = find_angle_path(loaded_angles, path_data)
	# angle_data = interpolate_angles(angle_data)
	# create_solution_images(angle_data, ORIGIN_X, ORIGIN_Y, L_ONE, L_TWO, L_THREE, 'right_to_left')


	# #top to bottom
	# path_data = find_x_y_path(249, 25, 249, 220, loaded_angles)
	# angle_data = find_angle_path(loaded_angles, path_data)
	# angle_data = interpolate_angles(angle_data)
	# create_solution_images(angle_data, ORIGIN_X, ORIGIN_Y, L_ONE, L_TWO, L_THREE, 'top_to_bottom')
 

	# imsave('/scratch/robotics/hw2/images/one_field.png', one_field)
	# imsave('/scratch/robotics/hw2/images/two_field.png', two_field)
	# imsave('/scratch/robotics/hw2/images/three_field.png', three_field)

	print('finished...')