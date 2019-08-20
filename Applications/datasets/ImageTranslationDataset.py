import glob
#import ntpath
import os
import random
import tensorflow as tf
from datasets.Dataset import Dataset
#import cv2

class ImageTranslationDataset(Dataset):
	FILE_BY_PART = {'train': 'train', 'test': 'test', 'val': 'val'}

	def __init__(self, data_directory, dataset_part, mean=None, std=None, num_classes=None, augment=True, png=True):
		if mean is None:
			mean = 100.50267074103701
		if std is None:
			std = 48.28870633760717
		if num_classes is None:
			num_classes = 751

		super().__init__(mean=mean, std=std, num_classes=num_classes, data_directory=data_directory, dataset_part=dataset_part, augment=augment, png=png)

	def get_input_data(self, is_training):
		image_paths = self.get_images_from_folder()

		if is_training:
			random.shuffle(image_paths)

		file_names = [os.path.basename(file) for file in image_paths]

		print('Read %d image paths for processing for dataset_part: %s' % (len(image_paths), self._dataset_part))
		return image_paths, file_names

	def get_number_of_samples(self):
		return len(self.get_images_from_folder())

	def prepare_sliced_data_for_batching(self, sliced_input_data, image_size):
		image_path_tensor, file_name_tensor = sliced_input_data
		image_tensor, target_tensor = self.load_data(image_path_tensor, image_size)

		return self.get_dict_for_batching(file_name_tensor=file_name_tensor, image_path_tensor=image_path_tensor,
										  image_tensor=image_tensor, target_tensor=target_tensor)

	def get_images_from_folder(self):
		data_file = self.get_data_file()
		return self.get_png_and_jpg(data_file)

	@staticmethod
	def get_png_and_jpg(data_file):
		all_images = glob.glob(os.path.join(data_file, '*.png'))
		all_images.extend(glob.glob(os.path.join(data_file, '*.jpg')))
		return all_images

	def get_data_file(self):
		data_file = self.FILE_BY_PART[self._dataset_part]
		return os.path.join(self._data_directory, data_file)

	def get_input_function_dictionaries(self, batched_input_data):
		return {'paths': batched_input_data['path'], 'images': batched_input_data['image'], 'file_names': batched_input_data['file_name']}, \
			   {'targets' : batched_input_data['target']}

	def load_data(self, image_path_tensor,image_size, which_direction="AtoB", scale_size=256):
		decode = tf.image.decode_jpeg
		with tf.name_scope("load_images"):
			content = tf.read_file(image_path_tensor)
			raw_input = decode(content)
			raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

		assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
		with tf.control_dependencies([assertion]):
			raw_input = tf.identity(raw_input)

		raw_input.set_shape([None, None, 3])

		width = tf.shape(raw_input)[1] # [height, width, channels]
		a_images = self.preprocess(raw_input[:,:width//2,:])
		b_images = self.preprocess(raw_input[:,width//2:,:])

		if which_direction == "AtoB":
			inputs, targets = [a_images, b_images]
		elif which_direction == "BtoA":
			inputs, targets = [b_images, a_images]
		else:
			raise Exception("invalid direction")


    # input and output images
		seed = 42
		def transform(r):

			if self._dataset_part == 'train':
				r = tf.image.random_flip_left_right(r, seed=seed)
			r.set_shape([256, 256, 3])
        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
		#	r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
		#	r = cv2.resize(r, (scale_size, scale_size))
		# 	offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - image_size + 1, seed=seed)), dtype=tf.int32)
		# 	if scale_size > image_size:
		# 		r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], image_size, image_size)
		# 	elif scale_size < image_size:
		# 		raise Exception("scale size cannot be less than crop size")
			return r

		with tf.name_scope("input_images"):
			input_images = transform(inputs)

		with tf.name_scope("target_images"):
			target_images = transform(targets)

		return input_images, target_images
	@staticmethod
	def preprocess(image):
		return image *2 -1
