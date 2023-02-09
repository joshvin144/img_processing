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

	# Edge detection with the derivative and smoothing
	# This is equivalent to filtering with a Sobel Kernel
	edge_x = filter_x(DERIVATIVE_X, img_xy)
	gauss_y = filter_y(GAUSS_Y, edge_x)
	edge_y = filter_x(DERIVATIVE_Y, gauss_y)
	gauss_x = filter_y(GAUSS_X, edge_y)

	# Edge detection using the second derivative
	laplace_xy = filter_xy(LAPLACIAN_XY, img_xy)

	if (args.plot):
		plt.imshow(gauss_x)
		# plt.imshow(laplace_xy)
		plt.show()

	return 0

# There should be no need to touch this
if (__name__ == "__main__"):
	_ = main()