# Set the path, here
import sys
sys.path.append("../../statistics/analysis_of_distributions")

# Import all modules required to execute this section of code, here
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from edge_detection_methods import DERIVATIVE_X
from edge_detection_methods import DERIVATIVE_Y
from edge_detection_methods import GAUSS_X
from edge_detection_methods import GAUSS_Y
from edge_detection_methods import LAPLACIAN_XY
from edge_detection_methods import filter_x
from edge_detection_methods import filter_y
from edge_detection_methods import filter_xy
from edge_detection_methods import add_noise_xy
from edge_detection_methods import BLUR_THRESHOLD

# For looking at the distribution of the pixels
from distributions import Distribution

# Debug
from icecream import ic

# RGB Scheme
R = 0
G = 1
B = 2

# Path to image
path = "./480px-SheppLogan_Phantom.svg.png"

# Add command line arguments, here
def create_argument_parser():
	argument_parser = argparse.ArgumentParser()
	argument_parser.add_argument("-p", "--plot", action = "store_true")
	argument_parser.add_argument("-n", "--noise", action = "store_true")
	return argument_parser

# Define the execution, here
def main():
	argument_parser = create_argument_parser()
	args = argument_parser.parse_args()

    # Load an image
	img = mpimage.imread(path)
	
	# Restrict to 2 dimensions
	num_dims = 0
	for dim in img.shape:
		num_dims += 1

	if (2 < num_dims):
		img_xy = img[:,:,R]
	else:
		img_xy = img

	if (args.noise):
		# Add noise to the image to throw off the edge detection
		# img_xy = add_noise_xy("Gaussian", img_xy)
		# img_xy = add_noise_xy("Poisson", img_xy) # Common in X-Ray/CT imaging
		img_xy = add_noise_xy("Rayleigh", img_xy) # Common in ultrasound imaging

	# Distribution of pixels
	# Why look at the distribution of pixels
	# Firstly, it gives us an idea of image contrast
	img_distribution = Distribution()
	img_distribution.mean = np.mean(img_xy)
	img_distribution.stddev = np.sqrt(np.var(img_xy))
	img_distribution.samples = img_xy.flatten()
	img_distribution.sample_size = img_xy.size

	# Edge detection with the derivative and smoothing
	# This is equivalent to filtering with a Sobel Kernel
	# edge_x = filter_x(DERIVATIVE_X, img_xy)
	# gauss_y = filter_y(GAUSS_Y, edge_x)
	# edge_y = filter_x(DERIVATIVE_Y, gauss_y)
	# gauss_x = filter_y(GAUSS_X, edge_y)

	# Blurring before edge detection using the Laplacian
	smoothed_x = filter_x(GAUSS_X, img_xy)
	smoothed_y = filter_y(GAUSS_Y, smoothed_x)
	# Edge detection using the second derivative
	laplace_xy = filter_xy(LAPLACIAN_XY, smoothed_y)

	if (args.plot):
		img_distribution.plot()
		fig, axs = plt.subplots(1, 2)
		axs[0].imshow(img_xy)
		axs[1].imshow(laplace_xy)
		plt.show()

	return 0

# There should be no need to touch this
if (__name__ == "__main__"):
	_ = main()

