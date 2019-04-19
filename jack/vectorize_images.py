"""

This will vectorize a directory of images using resnet50 and save them in npy files

Make an empty output directory called OUTPUT_FOLDER to save the output

Provide the path to the top level directory of the source image (For our case it will be train/ dev/ ...)
(There should be subdirectories that contain the images)

"""

import sys
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.resnet50 import preprocess_input

# Directory to save output
OUTPUT_FOLDER = "out/"

def main():
	directory = sys.argv[1]
	for root, dirs, _ in os.walk(directory):
		for subdir in dirs:
			subdir_path = os.path.join(root, subdir)
			output_path = OUTPUT_FOLDER + str(subdir)
			os.mkdir(output_path)

			for _, _, files in os.walk(subdir_path):
				for image in files:
					output_file = output_path + "/" + str(image) + ".npz"
					link = os.path.join(subdir_path, image)
					img_data = process_image(link)
				
					# Save the np array
					np.savez_compressed(output_file, a=img_data)
					
					# Load the np array
					# loaded = np.load(output_file)['a']

def process_image(link: str):
	img = load_img(link, target_size=(530, 700))
	img_data = img_to_array(img)
	img_data = np.expand_dims(img_data, axis=0)
	img_data = preprocess_input(img_data)
	return img_data

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage: python3 vectorize_images.py <image directory>")
		sys.exit(1)

	main()
