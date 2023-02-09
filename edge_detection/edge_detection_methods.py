# Include modules, here
import numpy as np
import scipy
import matplotlib.pyplot as plt

# Debugging
from icecream import ic

# We may look at the rate of change through the definition of derivative:
# [f(x + h) - f(x)]/[(x + h) - x] = [f(x + h) - f(x)]/h
# In vector form, this looks like [-1, 0, 1] multiplied by [xf(x - h), f(x), f(x + h)]
# What is h?
# h is the change in x, which is 1 pixel

# We may look at an image as the output of a function
# Therefore, each pixel represents the output of f(x)
# If this is so, then we may look take the difference between subsequent pixels to find the rate of change
# The higher the rate of change, the greater the edge

# This is one vector in the Sobel Kernel
# This vector, when multiplied by the pixels in an image, represents the derivative of said pixels
# [f(x - h), 0, f(x + h)] corresponds to [-1, 0, 1]
# Of course, the direction matters
# A row vector is a derivative with respect to x
# A column vector is a derivative with respect to y

# It is important to note that the image indices start at (0,0) in the upper righthand corner and increase to the right and down
# We orient the filters to match this convention

# The Sobel Kernel
# The Sobel Kernel is the product of the derivative vector and the Gauss vector
DERIVATIVE_X = np.array([[-1, 0, 1]], dtype = np.float32)
DERIVATIVE_Y = np.array([[-1], [0], [1]], dtype = np.float32)

GAUSS_X = np.array([[1, 2, 1]], dtype = np.float32)
GAUSS_Y = np.array([[1], [2], [1]], dtype = np.float32)

# The Laplacian Kernel
# The Laplacian Kernel is the equivalent of the second derivative
# This will pinpoint the center of the edge
LAPLACIAN_XY = np.array([[0, -1, 0],
	                     [-1, 4, -1],
	                     [0, -1, 0]], dtype = np.float32)

# For blur detection
# The variance of a blurred image is less than that of a clear image
BLUR_THRESHOLD = 0.3

def filter_x(kernel, img):
	""" kernel: A 2D vector, where the size of the second dimension is 1
	    img: A 2D image that is already zero padded """

	kernel_size_x = kernel.shape[1] # Number of columns
	img_size_x = img.shape[1] # Number of columns
	img_size_y = img.shape[0] # Number of rows
	new_img_size_x = img_size_x - (kernel_size_x - 1) # Number of columns

	new_img = np.zeros((img_size_y, new_img_size_x))
	# Start in the upper lefthand corner of the img: (0, 0)
	# Move the kernel
	for row in range(img_size_y):
		for col in range(new_img_size_x):
			# Multiply the elements of the kernel by the elements of the image
			for idx in range(kernel_size_x):
				new_img[row][col] += (kernel[0][idx]*img[row][col + idx])

	return new_img

def filter_y(kernel, img):
	""" kernel: A 2D vector, where the size of the first dimension is 1
	    img: A 2D image that is already zero padded """
	
	kernel_size_y = kernel.shape[0] # Number of rows
	img_size_x = img.shape[1] # Number of columns
	img_size_y = img.shape[0] # Number of rows
	new_img_size_y = img_size_y - (kernel_size_y - 1) # Number of rows

	new_img = np.zeros((new_img_size_y, img_size_x))
	# Start in the upper lefthand corner of the img: (0, 0)
	# Move the kernel
	for row in range(new_img_size_y):
		for col in range(img_size_x):
			# Multiply the elements of the kernel by the elements of the image
			for idx in range(kernel_size_y):
				new_img[row][col] += (kernel[idx][0]*img[row + idx][col])

	return new_img

def filter_xy(kernel, img):
	""" kernel: A 2D array
	    img: A 2D image that is already zero padded """
	
	kernel_size_x = kernel.shape[1] # Number of columns
	kernel_size_y = kernel.shape[0] # Number of rows
	img_size_x = img.shape[1] # Number of columns
	img_size_y = img.shape[0] # Number of rows
	new_img_size_x = img_size_x - (kernel_size_x - 1) # Number of columns
	new_img_size_y = img_size_y - (kernel_size_y - 1) # Number of rows

	new_img = np.zeros((new_img_size_y, new_img_size_x))
	# Start in the upper lefthand corner of the img: (0, 0)
	# Move the kernel
	for row in range(new_img_size_y):
		for col in range(new_img_size_x):
			# Multiply the elements of the kernel by the elements of the image
			for row_idx in range(kernel_size_y):
				for col_idx in range(kernel_size_x):
					new_img[row][col] += (kernel[row_idx][col_idx]*img[row + row_idx][col + col_idx])

	return new_img

def add_noise_xy(noise_type, img):
	img_size_x = img.shape[1] # Number of columns
	img_size_y = img.shape[0] # Number of rows
	noisy_img = img

	if ("Gaussian" == noise_type):
		mean = 0
		stddev = 1
		noise = np.random.normal(mean, stddev, (img_size_y, img_size_x))
		noisy_img += noise

	return noisy_img
