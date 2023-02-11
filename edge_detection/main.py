# Set the path, here
import sys
sys.path.append("../../statistics/analysis_of_distributions")

# Import all modules required to execute this section of code, here
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage

from edge_detection_methods import add_noise_xy

from edge_detection_methods import Filter
from edge_detection_methods import Tester

from edge_detection_methods import DERIVATIVE_X
from edge_detection_methods import DERIVATIVE_Y
from edge_detection_methods import GAUSS_X
from edge_detection_methods import GAUSS_Y
from edge_detection_methods import LAPLACIAN_XY
from edge_detection_methods import filter_x
from edge_detection_methods import filter_y
from edge_detection_methods import filter_xy

# For looking at the distribution of the pixels
from distributions import Distribution

# Debug
from icecream import ic

# RGB Scheme
R = 0
G = 1
B = 2

# Path to image
# path = "./480px-SheppLogan_Phantom.svg.png"
path = "A-Anterior-view-of-the-heart-longitudinal-cross-section-showing-dilatation-of-both.png"

# Add command line arguments, here
def create_argument_parser():
	argument_parser = argparse.ArgumentParser()
	argument_parser.add_argument("-p", "--plot", action = "store_true")
	argument_parser.add_argument("-n", "--noise", action = "store_true")
	argument_parser.add_argument("-s", "--smooth", action = "store_true")
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

	# Gaussian Smoothing
	gauss_y_filter = Filter(GAUSS_Y, filter_y)
	gauss_x_filter = Filter(GAUSS_X, filter_x)

	if (args.smooth):
		gaussian_smoothing_tester = Tester()
		gaussian_smoothing_tester.add_to_sequence(gauss_x_filter)
		gaussian_smoothing_tester.add_to_sequence(gauss_y_filter)
		img_xy = gaussian_smoothing_tester.run(img_xy)

    # Sobel Filter
	derivative_x_filter = Filter(DERIVATIVE_X, filter_x)
	derivative_y_filter = Filter(DERIVATIVE_Y, filter_y)

	sobel_tester = Tester()
	sobel_tester.add_to_sequence(derivative_x_filter)
	sobel_tester.add_to_sequence(gauss_y_filter)
	sobel_tester.add_to_sequence(derivative_y_filter)
	sobel_tester.add_to_sequence(gauss_x_filter)
	sobel_filtered_img = sobel_tester.run(img_xy)

	# Laplacian Filter
	laplace_xy_filter = Filter(LAPLACIAN_XY, filter_xy)
	laplace_tester = Tester()
	laplace_tester.add_to_sequence(laplace_xy_filter)
	laplace_filtered_img = laplace_xy_filter.run(img_xy)

	sobel_filtered_img_blur = np.var(laplace_filtered_img)
	print("Sobel filtered image blur:\t{:.9f}".format(sobel_filtered_img_blur))

	if (args.plot):
		fig, axs = plt.subplots(1, 3)
		axs[0].imshow(img_xy)
		axs[1].imshow(sobel_filtered_img)
		axs[2].imshow(laplace_filtered_img)
		plt.show()

	return 0

# There should be no need to touch this
if (__name__ == "__main__"):
	_ = main()

