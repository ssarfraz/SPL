# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:33:42 2018

@author: ssarfraz
"""

import argparse
import csv
import glob
import os
import shutil
import sys
import imageio
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import RunConfig
from tensorflow.python.estimator.estimator import _load_global_step_from_checkpoint_dir

from datasets.DatasetFactory import DatasetFactory

import pickle

slim = tf.contrib.slim

def start_prediction(data_directory, dataset_name, mean, model_dir, network_name, batch_size, batch_threads, num_classes, result_dir,img_size, model,mode):
	dataset_factory = DatasetFactory(dataset_name=dataset_name, data_directory=data_directory, mean=mean, augment=False, num_classes=num_classes)

	run_config = RunConfig(keep_checkpoint_max=10, save_checkpoints_steps=None)
	# Instantiate Estimator
	estimator = tf.estimator.Estimator(
		model_fn=get_model_function(model_dir, network_name, dataset_factory.get_dataset('train').num_classes()),
		model_dir=model_dir,
		config=run_config,
		params={})

	image_size = img_size
	run_prediction_and_evaluation(batch_size, batch_threads, dataset_factory, estimator, image_size, result_dir,mode)


def run_prediction_and_evaluation(batch_size, batch_threads, dataset_factory, estimator, image_size, result_dir,mode):
	output_directory = get_prediction_directory(estimator)

	if os.path.exists(output_directory):
		shutil.rmtree(output_directory)

	if not os.path.exists(output_directory):
		os.makedirs(output_directory)

	copy_checkpoint(estimator.model_dir, output_directory)

	print('Starting feature vector generation...')

	
	run_prediction_and_store_images(dataset_factory, batch_size, batch_threads, estimator, output_directory, mode, image_size,result_dir)

	print('Finished feature vector generation.')




def get_prediction_directory(estimator):
	return os.path.join(estimator.model_dir, "predictions")


def copy_checkpoint(model_dir, output_directory):
	print('Copying current checkpoint')
	shutil.copyfile(os.path.join(model_dir, 'checkpoint'), os.path.join(output_directory, 'checkpoint'))
	latest_checkpoint = tf.train.latest_checkpoint(model_dir)

	for file in glob.glob(latest_checkpoint + '*'):
		shutil.copy(file, output_directory)


def run_prediction_and_store_images(dataset_factory, batch_size, batch_threads, estimator, output_directory, dataset_part, image_size,result_dir):
	dataset = dataset_factory.get_dataset(dataset_part)


	print('\n\nRunning Prediction for %s' % dataset_part)
	input_function = get_input_function(dataset, batch_size, batch_threads, False, image_size)
	predicted = estimator.predict(input_fn=input_function)

	num_samples = dataset.get_number_of_samples()

	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	if not os.path.exists(os.path.join(result_dir,'input')):
		os.makedirs(os.path.join(result_dir,'input'))
	if not os.path.exists(os.path.join(result_dir,'prediction')):
		os.makedirs(os.path.join(result_dir,'prediction'))
	if not os.path.exists(os.path.join(result_dir,'target')):
		os.makedirs(os.path.join(result_dir,'target'))

	for sample, prediction in enumerate(predicted):
		if (sample + 1) % batch_size == 0:
			sys.stdout.write('\r>> Processed %d samples of %d' % (sample + 1, num_samples))
			sys.stdout.flush()

		out_im=(1 * prediction['output'])#.astype(np.uint8)
		in_im=(1 * prediction['input'])#.astype(np.uint8)
		tar_im=(1 * prediction['target'])#.astype(np.uint8)


		imageio.imwrite(os.path.join(result_dir,'input',prediction['file_names'].decode('UTF-8')), in_im)
		imageio.imwrite(os.path.join(result_dir,'prediction',prediction['file_names'].decode('UTF-8')), out_im)
		imageio.imwrite(os.path.join(result_dir,'target',prediction['file_names'].decode('UTF-8')), tar_im)

	print('\nFinished Prediction %s' % dataset_part)






def main(args):
	
	mean = None

	

	start_prediction(args.data_directory, args.dataset_name, mean, args.model_dir, args.network_name, args.batch_size, args.batch_threads, args.num_classes, args.result_dir,args.image_size, args.model,args.mode)

	print('Exiting ...')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', help='Specify the folder with the images to be trained and evaluated', dest='data_directory')
	parser.add_argument('--dataset-name', help='The name of the dataset')
	parser.add_argument('--batch-size', help='The batch size', type=int, default=16)
	parser.add_argument('--batch-threads', help='The number of threads to be used for batching', type=int, default=4)
	parser.add_argument('--model-dir', help='The model to be loaded')
	parser.add_argument('--model', default = 'img_translation', help='The model to be loaded')
	parser.add_argument('--mode', default = 'val', help='what split to evaluate')
	parser.add_argument('--image_size', default=256, help='Defines the size of the images')
	parser.add_argument('--network-name', help='Name of the network')
	parser.add_argument('--result_dir', default='./results/',help='Output directory of results')
	args = parser.parse_args()

	print('Running with command line arguments:')
	print(args)
	print('\n\n')

	if args.model == 'img_translation':
		from helper.model_helper import get_model_function, get_input_function
	elif args.model == 'makeup':
		from helper.model_helper_makeup import get_model_function, get_input_function
	elif args.model == 'hires':
		from helper.model_helper_HighRes import get_model_function, get_input_function
	else:
		raise ValueError('Model type \"{}\" not known'.format(model))

	main(args)
