# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:31:21 2018

@author: ssarfraz
"""

import tensorflow as tf
from helper import layers
from helper.checkpoint_helper import check_init_from_initial_checkpoint
from helper.checkpoint_helper import init_from_checkpoint
from tensorflow.python.ops import array_ops
from helper import SPL
ModeKeys = tf.estimator.ModeKeys
EPS = 1e-12

def get_input_function(dataset, batch_size, batch_threads, is_training, image_size):
	input_data = dataset.get_input_data(is_training)

	def input_fn():
		sliced_input_data = tf.train.slice_input_producer(input_data, num_epochs=1, shuffle=is_training, capacity=4096)
		sliced_data_dictionary = dataset.prepare_sliced_data_for_batching(sliced_input_data, image_size)

		batched_input_data = tf.train.batch(tensors=sliced_data_dictionary,
											batch_size=batch_size,
											num_threads=batch_threads,
											capacity=batch_threads * batch_size * 2,
											allow_smaller_final_batch=not is_training)
		# tf.summary.image(name='input_images', tensor=batched_input_data['image'])

		(features, targets) = dataset.get_input_function_dictionaries(batched_input_data)
		features.update(targets)
		return features, targets

	return input_fn


def get_model_function(output_directory, network_name, num_classes, initial_checkpoint=None, checkpoint_exclude_scopes=None, ignore_missing_variables=False, trainable_scopes=None,
					   not_trainable_scopes=None):
	def model_fn(features, labels, mode, params):
		spl_loss = SPL.SPL()
		if labels is None:  # when predicting, labels is None
			labels = {}

		images = features['images']

		targets_out = features['targets'] if 'targets' in features else None ## for sysu
		tensor_shape=targets_out.get_shape().as_list()

		with tf.variable_scope('generator'):
			out_channels = int(targets_out.get_shape()[-1])

			output_1 = build_generator_resnet_9blocks_tf(images,  targets_out, out_channels, batch=images.get_shape().as_list()[0], name='generator_main', ngf=64, rate=1)

		check_init_from_initial_checkpoint(output_directory, initial_checkpoint, checkpoint_exclude_scopes, ignore_missing_variables)

		predictions_dict = {}
		train_op = tf.no_op()
		eval_metric_ops = {}

		if mode == ModeKeys.EVAL or mode == ModeKeys.TRAIN:
			with tf.name_scope('input_summary'):
				tf.summary.image('Inputs', images)

			with tf.name_scope('targets_summary'):
				tf.summary.image('targets',targets_out)

			with tf.name_scope('generator_output1_summary'):
				tf.summary.image('Gen_output', output_1)

			with tf.name_scope('losses'):
				tf.summary.scalar(name='regularization', tensor=tf.losses.get_regularization_loss())

				with tf.name_scope("New_Gen_Loss"):


					overall_loss = spl_loss(targets_out, output_1)

					tf.summary.scalar('CPLoss+ GPLoss', overall_loss)

		if mode == ModeKeys.TRAIN:
			def learning_rate_decay_function(learning_rate, global_step):
				if not params['fixed_learning_rate']:
					return tf.train.exponential_decay(learning_rate=learning_rate,
													  global_step=global_step,
													  decay_steps=params['learning_rate_decay_steps'],
													  decay_rate=params['learning_rate_decay_rate'],
													  staircase=True,
													  name='learning-rate-decay')
				else:
					return learning_rate

			with tf.name_scope("generator_train"):
				gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
				gen_train = tf.contrib.layers.optimize_loss(loss=overall_loss,
													   global_step=tf.train.get_or_create_global_step(),
													   learning_rate = params['learning_rate_gen'],
													   optimizer= lambda learning_rate: tf.train.AdamOptimizer(learning_rate, params['beta1_gen']),
													   variables=gen_tvars,
													   learning_rate_decay_fn=learning_rate_decay_function)


			train_op = gen_train
		if mode == ModeKeys.PREDICT or mode == ModeKeys.EVAL:
			predictions_dict.update(features)
			predictions_dict.update(labels)
			predictions_dict['input']=(images+1)/2
			predictions_dict['target']=(targets_out+1)/2
			predictions_dict['output']=(output_1+1)/2
		tf.losses.add_loss(overall_loss) if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL else None
		total_loss = tf.losses.get_total_loss() if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL else None
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict, loss=total_loss, train_op=train_op,
										  eval_metric_ops=eval_metric_ops)



	return model_fn


def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT", rate=1): # rate=1 for no dilation
	"""build a single block of resnet.
	:param inputres: inputres
	:param dim: dim
	:param name: name
	:param padding: for tensorflow version use REFLECT; for pytorch version use
	 CONSTANT
	:return: a single block of resnet.
	"""
	with tf.variable_scope(name):
		out_res = tf.pad(inputres, [[0, 0], [1, 1], [
			1, 1], [0, 0]], padding)
		out_res = layers.general_conv2d(
			out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1", rate=rate)
		out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
		out_res = layers.general_conv2d(
			out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False, rate=rate)

		return tf.nn.relu(out_res + inputres)

def build_generator_resnet_9blocks_tf(inputgen,ref, out_channels, batch=12, name="generator", skip=False, ngf=32, reuse=False, rate=1):
	with tf.variable_scope(name, reuse=reuse):
		f = 7
		ks = 3
		padding = "REFLECT"

		pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [
			ks, ks], [0, 0]], padding)
		o_c1 = layers.general_conv2d(
			pad_input, ngf, f, f, 1, 1, 0.02, name="c1") # added "SAME" here for dilated
		o_c2 = layers.general_conv2d(
			o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")
		o_c3 = layers.general_conv2d(
			o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")

		o_r1 = build_resnet_block(o_c3, ngf * 4, "r1", padding)
		o_r2 = build_resnet_block(o_r1, ngf * 4, "r2", padding)
		o_r3 = build_resnet_block(o_r2, ngf * 4, "r3", padding)
		o_r4 = build_resnet_block(o_r3, ngf * 4, "r4", padding)
		o_r5 = build_resnet_block(o_r4, ngf * 4, "r5", padding)
		o_r6 = build_resnet_block(o_r5, ngf * 4, "r6", padding)
		o_r7 = build_resnet_block(o_r6, ngf * 4, "r7", padding)
		o_r8 = build_resnet_block(o_r7, ngf * 4, "r8", padding)
		o_r9 = build_resnet_block(o_r8, ngf * 4, "r9", padding)

		o_c4 = layers.general_deconv2d(
			o_r9, [batch, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02,
			"SAME", "c4")
		o_c5 = layers.general_deconv2d(
			o_c4, [batch, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02,
			"SAME", "c5")
		o_c6 = layers.general_conv2d(o_c5, out_channels, f, f, 1, 1,
									 0.02, "SAME", "c6",
									 do_norm=False, do_relu=False, rate=rate) 
		if skip is True:
			out_gen = tf.nn.tanh(ref + o_c6, "t1")
		else:
			out_gen = tf.nn.tanh(o_c6, "t1")
		return out_gen



