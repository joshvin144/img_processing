# Include modules, here
import numpy as np
import scipy
import matplotlib.pyplot as plt

# For looking at the distribution of the pixels
from distributions import Distribution

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

#### Kernels ####

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

#### Image Quality ####

# For blur detection
# The variance of a blurred image is less than that of a clear image
BLUR_THRESHOLD = 0.3

#### Parent Class for Filters ####
class Filter(object):
	def __init__(self, kernel = None, function = None):
		self.kernel = kernel
		self.function = function

	def run(self, img):
		filtered_img = self.function(self.kernel, img)
		return filtered_img

	def __repr__(self):
		return "Parent Class for Filters"

#### Tester for Filter Functions ####

class Tester(object):
	def __init__(self):
		# Sequence of filters to apply to the image
		self.sequence = []

	def add_to_sequence(self, filter_):
		# Every filter needs a kernel and filter function
		# This must be set beforehand
		# Therefore, we pass filter objects to the tester that include a kernel and filter function
		self.sequence.append(filter_)

	def remove_last_function_from_sequence(self):
		self.sequence.pop()

	def reset_sequence(self):
		self.sequence = []

	def run(self, img):
		for filter_ in self.sequence:
			img = filter_.run(img)
		return img

	def __repr__(self):
		return "Tester for filter functions"

#### Filter Functions ####

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

#### Noise ####

def add_noise_xy(noise_type, img):
	img_size_x = img.shape[1] # Number of columns
	img_size_y = img.shape[0] # Number of rows
	noisy_img = img

	if ("Gaussian" == noise_type):
		mean = 0
		stddev = 0.3
		noise = np.random.normal(mean, stddev, (img_size_y, img_size_x))
		noisy_img += noise
	elif ("Poisson" == noise_type):
		num_events = 1
		noise = np.random.poisson(num_events, (img_size_y, img_size_x))
		noisy_img += noise
	elif ("Rayleigh" == noise_type):
		scale = 1
		noise = np.random.rayleigh(scale, (img_size_y, img_size_x))
		noisy_img = np.multiply(noisy_img, noise)
	return noisy_img

#### Contrast ####

# Contrast Stretching
# Min-Max stretching
def stretch_contrast_xy(scale, img):
	img_size_x = img.shape[1] # Number of columns
	img_size_y = img.shape[0] # Number of rows
	new_img = np.zeros((img_size_y, img_size_x))

	max_ = np.amax(img)
	min_ = np.amin(img)
	for row in range(img_size_y):
		for col in range(img_size_x):
			new_img[row][col] = scale*(img[row][col] - min_)/(max_ - min_)

	return new_img

# Histogram Equalization
# Histogram equalization is a technique used to improve contrast in an image
# Note that the histogram of an image is a destructive function
# You may go from an image to a histogram, but not a histogram to an image
# Therefore, the argument to perform_histogram_equalization_xy must be the image, itself
# The histogram of the image must be taken within the function
def equalize_histogram_xy(num_bins, img):
	img_size_x = img.shape[1] # Number of columns
	img_size_y = img.shape[0] # Number of rows
	equalized_img = np.zeros((img_size_y, img_size_x))

	# Distribution of pixels
	img_distribution = Distribution()
	img_distribution.mean = np.mean(img)
	img_distribution.stddev = np.sqrt(np.var(img))
	img_distribution.samples = img.flatten()
	img_distribution.sample_size = img.size
	img_distribution.plot()

	hist, bin_edges = np.histogram(img_distribution.samples, num_bins)
	# It is assumed that each bin represents a unique intensity value
	# Therefore, the number of intensity values is equal to the number of bins
	num_intensity_values = hist.shape[0]

	# There is a bin for each possible intensity
	# From the histogram, you may calculated the PMF at each intensity
	# Histogram equalization uses the CDF to increase the contrast of the image
	
	# Histogram equalization algorithm
	# Across all of the possible intensity values
	cdf = 0
	for n in range(num_intensity_values):
		# The PMF is a value between 0 and 1
		# The CDF is a value between 0 and 1
		pmf = hist[n]/img_distribution.sample_size
		cdf += pmf
		# Adjust the pixels that have the intensity value
		for row in range(img_size_y):
			for col in range(img_size_x):
				if((bin_edges[n] <= img[row][col]) and (bin_edges[n + 1] > img[row][col])):
					# The new intensity of the pixel is the product of the number of intensity values and the CDF
					# Essentially, this uses the CDF to weight the pixels
					equalized_img[row][col] = np.floor((num_intensity_values - 1)*cdf)
					# What if we weight by PMF?
	# equalized_img /= cdf # Normalize by the CDF
	# The CDF is 1 at it's maximum, so we do not need to divide by it again

	# Example from CV2
	# Calculate the CDF
	# cdf = hist.cumsum() # Cumulative sum
	# cdf_norm = cdf*float(hist.max())/cdf.max() # Normalize by the CDF
	# # Mask the array
	# cdf_m = np.ma.masked_equal(cdf,0) # Set the mask to False where-ever the CDF is equal to 0
	# cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min()) # Normalize the pixel intensities
	# cdf = np.ma.filled(cdf_m,0).astype('uint8') # Fill the missing values with 0
	# equalized_img = cdf[img]

	equalized_img_distribution = Distribution()
	equalized_img_distribution.mean = np.mean(equalized_img)
	equalized_img_distribution.stddev = np.sqrt(np.var(equalized_img))
	equalized_img_distribution.samples = equalized_img.flatten()
	equalized_img_distribution.sample_size = equalized_img.size
	equalized_img_distribution.plot()

	return equalized_img

#### Masking ####


#### Segmentation ####

