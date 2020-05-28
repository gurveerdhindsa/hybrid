"""hybrid.py"""
__author__ = "Gurveer Dhindsa"

from PIL import Image
from pylab import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import ceil
import cv2

"""
Convolutes image with a kernel

Args:
    image - the image being filtered
    kernel - a filter which is being applied to the image

Returns:
    result - an image with a kernel filter applied to it
"""
def cross_correlation_2d(image, kernel):
    if (len(image.shape) >= 3):
        # print("Color image detected")
        imageHeight, imageWidth, channels = image.shape

    else:
        # print("Grayscale image detected")
        imageHeight, imageWidth = image.shape 

    # Grab the kernel dimensions
    kernelHeight, kernelWidth = kernel.shape

    # Allocate memory for output image, so we must consider the padding
    # I chose to use 'same' convolution because I want the output image to be the same size of input image
    # In general, the equation to calculate padding is: p = (k-1)//2 (where p=padding, k=kernel)
    padHeight = (kernelHeight - 1) // 2
    padWidth = (kernelWidth - 1) // 2

    # The result image (post-convolution) will be the same dimensions of the input image
    result = np.zeros(image.shape, dtype=float)

    # Our image with padding will be the same dimensions of the input impage PLUS padding on all edges
    imagePadded = np.zeros((imageHeight + padHeight + padHeight, imageWidth + padWidth + padWidth, channels), dtype=float)

    # Grab the padded image dimensions
    imagePaddedHeight, imagePaddedWidth, imagePaddedChannel = imagePadded.shape

    # Copy image into center
    imagePadded[padHeight:imageHeight + padHeight, padWidth:imageWidth + padWidth] = image
    
    # Iterate across each channel
    for channel in range(channels):
        # Iterate the original image pixel by pixel...
        for x in range(imageHeight):
            for y in range(imageWidth):
                # Multiply the padded image with kernel filter then take sum of all pixels to create a single pixel
                # The single pixel will be added to the result image
                result[x,y,channel] = (np.matmul(imagePadded[x:kernelWidth + x, y:kernelHeight + y, channel], kernel)).sum()

    # Return the convoluted image with kernel applied
    return result

"""
Uses cross_correlation_2d to convolve image

Args:
    image - the image being filtered
    kernel - a filter which is being applied to the image

Returns:
    result - an image with a kernel filter applied to it
"""
def convolve_2d(image, kernel):
    # Accepts a kernel and calls cross_correlation_2d, then returns the image
    return cross_correlation_2d(image, kernel)

"""
Creates a gaussian filter

Args:
    image - the image being filtered
    sigma - an integer that determines the intensity of blurring

Returns:
    kernel - a filter that can be convoluted to image to smooth it
"""
def gaussian_blur_kernel_2d(image, sigma):
    image = np.asarray(image)
    filter_size = 2 * ceil(3 * sigma) + 1

    kernel = np.zeros((filter_size, filter_size), dtype=float)

    for x in range(-filter_size, filter_size):
        for y in range(-filter_size, filter_size):
            kernel[x, y] = (1 / (2 * np.pi * (sigma ** 2))) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return kernel

"""
Low pass filter removing fine details of image

Args:
    image - the image being filtered
    sigma - an integer that determines the intensity of blurring

Returns:
    lowPassedImage - a filter that can be convoluted to image to smooth it
"""
def low_pass(image, sigma):
    lowPassedImage = convolve_2d(image, gaussian_blur_kernel_2d(image, sigma))
    return lowPassedImage

"""
High pass filter retails fine details of image

Args:
    image - the image being filtered
    sigma - an integer that determines the intensity of blurring

Returns:
    highPassedImage - a filter that can be convoluted to image to smooth it
"""
def high_pass(image, sigma):
    highPassedImage = (image - low_pass(image, sigma))
    return highPassedImage

"""
Main function
"""
def main():
    # Declare constants
    sigma = 4
    mixinRatio = 0.5

    image1 = cv2.imread("./images/dog.png")
    image2 = cv2.imread("./images/cat.png")
    
    print('Creating hybrid image...')

    lowPass = low_pass(image1, 4) / 255
    highPass = high_pass(image2, 4) / 255

    hybridImage = (highPass * mixinRatio + lowPass * (1-mixinRatio))
    print('Hybrid image created!')

    # Display the hybrid image and save it
    plt.imshow(hybridImage)
    plt.axis("off")
    plt.title("Hybrid Image")
    plt.savefig('./results/hybrid.png')
    plt.show()

if __name__ == "__main__":
    main()