# Import all modules required to execute this section of code, here
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from edge_detection_methods import SOBEL_X
from edge_detection_methods import SOBEL_Y
from edge_detection_methods import GAUSS_X
from edge_detection_methods import GAUSS_Y
from edge_detection_methods import derivative_x
from edge_detection_methods import derivative_y

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

    # Edge detection with just the derivative
	# edge_x = derivative_x(SOBEL_X, img_xy)
	# edge_y = derivative_y(SOBEL_Y, edge_x)

	# Edge detection with the derivative and smoothing
	edge_x = derivative_x(SOBEL_X, img_xy)
	gauss_y = derivative_y(GAUSS_Y, edge_x)
	edge_y = derivative_y(SOBEL_Y, gauss_y)
	gauss_x = derivative_x(GAUSS_X, edge_y)

	np.savetxt("edges.txt", gauss_x)

	if (args.plot):
		plt.imshow(gauss_x)
		plt.show()

	return 0

# There should be no need to touch this
if (__name__ == "__main__"):
	_ = main()