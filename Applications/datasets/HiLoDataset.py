# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 20:01:08 2018

@author: hkhalid
"""
import glob
import ntpath
import os
import random
import tensorflow as tf
from datasets.Dataset import Dataset


class HiLoDataset(Dataset):
	FILE_BY_PART = {'train':{'HR': 'DIV2K_train_HR', 'LR': 'DIV2K_train_LR_unknown/X4'}, 'test':{'HR': 'DIV2K_valid_HR', 'LR': 'DIV2K_valid_LR_unknown/X4'}, 'val': 'val'}

	def __init__(self, data_directory, dataset_part, mean=None, std=None, num_classes=None, augment=True, png=True):
		if mean is None:
			mean = 100.50267074103701
		if std is None:
			std = 48.28870633760717
		if num_classes is None:
			num_classes = 751

		super().__init__(mean=mean, std=std, num_classes=num_classes, data_directory=data_directory, dataset_part=dataset_part, augment=augment, png=png)

	def get_input_data(self, is_training):
		image_paths_A = sorted(self.get_images_from_folder('LR'))
		image_paths_B = sorted(self.get_images_from_folder('HR'))

		if is_training:
			comb = list(zip(image_paths_A, image_paths_B))
			random.shuffle(comb)
			image_paths_A, image_paths_B = zip(*comb)

		file_names_A = [os.path.basename(file) for file in image_paths_A]
		file_names_B = [os.path.basename(file) for file in image_paths_B]

		print('Read %d image paths for processing for dataset_part: %s' % (len(image_paths_A), self._dataset_part))
		return image_paths_A, file_names_A, image_paths_B, file_names_B

	def get_number_of_samples(self):
		return len(self.get_images_from_folder('LR'))

	def prepare_sliced_data_for_batching(self, sliced_input_data, image_size):
		image_path_tensor_A, file_name_tensor_A, image_path_tensor_B, file_name_tensor_B = sliced_input_data
		image_tensor, target_tensor = self.load_data(image_path_tensor_A, image_path_tensor_B, crop_size=98)  #specify direction here

		return self.get_dict_for_batching(file_name_tensor=file_name_tensor_A, image_path_tensor=image_path_tensor_A,
										  image_tensor=image_tensor, target_tensor=target_tensor)

	def get_images_from_folder(self,domain):
		data_file = self.get_data_file(domain)
		return self.get_png_and_jpg(data_file)

	@staticmethod
	def get_png_and_jpg(data_file):
		all_images = glob.glob(os.path.join(data_file, '*.png'))
		all_images.extend(glob.glob(os.path.join(data_file, '*.jpg')))
		return all_images

	def get_data_file(self,domain):
		data_file = self.FILE_BY_PART[self._dataset_part][domain]
		return os.path.join(self._data_directory, data_file)

	def get_input_function_dictionaries(self, batched_input_data):
		return {'paths': batched_input_data['path'], 'images': batched_input_data['image'], 'file_names': batched_input_data['file_name']}, \
			   {'targets' : batched_input_data['target']}

	def load_data(self, image_path_tensor_A, image_path_tensor_B, crop_size=128):  # A to B makeup to nonmakeup
		decode = tf.image.decode_png
		with tf.name_scope("load_images"):
			content_A = tf.read_file(image_path_tensor_A)
			raw_input_A = decode(content_A, channels=3) ## change here to channel 3
			raw_input_A = tf.image.convert_image_dtype(raw_input_A, dtype=tf.float32)
			content_B = tf.read_file(image_path_tensor_B)
			raw_input_B = decode(content_B, channels=3) ## change here to channel 3
			raw_input_B = tf.image.convert_image_dtype(raw_input_B, dtype=tf.float32)


		raw_input_A.set_shape([None, None, 3]) ## changed here from 3
		raw_input_A = self.preprocess(raw_input_A)
		raw_input_B.set_shape([None, None, 3]) ## changed here from 3
		raw_input_B = self.preprocess(raw_input_B)

		#seed = random.randint(0, 2**31 - 1)

		def transform(LRimage, HRimage):
			#input and target corresponding random patches
			input_size = tf.shape(LRimage)
			offset_w = tf.cast(tf.floor(tf.random_uniform([], 0, tf.cast(input_size[1], tf.float32) - crop_size)), dtype=tf.int32)
			offset_h = tf.cast(tf.floor(tf.random_uniform([], 0, tf.cast(input_size[0], tf.float32) - crop_size)), dtype=tf.int32)

			input = tf.image.crop_to_bounding_box(LRimage, offset_h, offset_w, crop_size,
	                                                               crop_size)
			target = tf.image.crop_to_bounding_box(HRimage, offset_h*4, offset_w*4, crop_size*4,
																crop_size*4)
			return input, target

		if self._dataset_part == 'train':
			input_image, target_image = transform(raw_input_A, raw_input_B)
			input_image.set_shape([crop_size, crop_size, 3])
			target_image.set_shape([crop_size*4, crop_size*4, 3])
		else:
			input_image = raw_input_A
			target_image = raw_input_B

		return input_image, target_image

	@staticmethod
	def preprocess(image):
		return image *2 -1

	def random_flip(self, input, decision):
		f1 = tf.identity(input)
		f2 = tf.image.flip_left_right(input)
		output = tf.cond(tf.less(decision, 0.5), lambda: f2, lambda: f1)
		return output

	def random_rot(self, input, decision):
		f1 = tf.identity(input)
		f2 = tf.image.rot90(input)
		output = tf.cond(tf.less(decision, 0.5), lambda: f2, lambda: f1)
		return output
