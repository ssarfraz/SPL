
import tensorflow as tf
import tensorflow.contrib.slim as slim
from helper.checkpoint_helper import check_init_from_initial_checkpoint
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
											allow_smaller_final_batch=False,
											dynamic_pad =True)

		(features, targets) = dataset.get_input_function_dictionaries(batched_input_data)
		features.update(targets)
		return features, targets

	return input_fn


def get_model_function(output_directory, network_name, num_classes, initial_checkpoint=None, checkpoint_exclude_scopes=None, ignore_missing_variables=False, trainable_scopes=None,
						 not_trainable_scopes=None):
	def model_fn(features, labels, mode, params):
		if labels is None:  # when predicting, labels is None
			labels = {}

		spl = SPL.SPL()


		images = features['images']
		print(images.get_shape().as_list())

		targets_out = features['targets'] if 'targets' in features else None
		print(targets_out.get_shape().as_list())

		with tf.variable_scope('generator'):
			out_channels = int(targets_out.get_shape()[-1])
			output_1 = generator(images, out_channels, reuse=False, is_training= mode == ModeKeys.TRAIN)

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
				tf.summary.image('Gen_output_1', output_1)
			with tf.name_scope('losses'):
				tf.summary.scalar(name='regularization', tensor=tf.losses.get_regularization_loss())

				with tf.name_scope("New_Gen_Loss"):
					overall_loss= spl(targets_out, output_1)
					tf.summary.scalar('SPL_Loss', overall_loss)

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
														 optimizer=lambda learning_rate: tf.train.AdamOptimizer(learning_rate, params['beta1_gen']),
														 variables=gen_tvars,
														 learning_rate_decay_fn=learning_rate_decay_function)


			train_op = gen_train
		if mode == ModeKeys.PREDICT or mode == ModeKeys.EVAL:
			predictions_dict.update(features)
			predictions_dict.update(labels)

			predictions_dict['input']= tf.image.convert_image_dtype((images+1)/2, tf.uint8)
			predictions_dict['target']=tf.image.convert_image_dtype((targets_out+1)/2, tf.uint8)
			predictions_dict['output']=tf.image.convert_image_dtype((output_1+1)/2, tf.uint8)

		tf.losses.add_loss(overall_loss) if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL else None
		total_loss = tf.losses.get_total_loss() if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL else None
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict, loss=total_loss, train_op=train_op,
											eval_metric_ops=eval_metric_ops)

	return model_fn

# Definition of the generator
def generator(gen_inputs, gen_output_channels, reuse=False, is_training=True, num_resblock=16):
		# The Bx residual blocks
		def residual_block(inputs, output_channel, stride, scope):
				with tf.variable_scope(scope):
						net = conv2(inputs, 3, output_channel, stride, use_bias=False, scope='conv_1')
						net = batchnorm(net, is_training)
						net = prelu_tf(net)
						net = conv2(net, 3, output_channel, stride, use_bias=False, scope='conv_2')
						net = batchnorm(net, is_training)
						net = net + inputs

				return net


		with tf.variable_scope('generator_unit', reuse=reuse):
				# The input layer
				with tf.variable_scope('input_stage'):
						net = conv2(gen_inputs, 9, 64, 1, scope='conv')
						net = prelu_tf(net)

				stage1_output = net

				# The residual block parts
				for i in range(1, num_resblock+1 , 1):
						name_scope = 'resblock_%d'%(i)
						net = residual_block(net, 64, 1, name_scope)

				with tf.variable_scope('resblock_output'):
						net = conv2(net, 3, 64, 1, use_bias=False, scope='conv')
						net = batchnorm(net, is_training)

				net = net + stage1_output

				with tf.variable_scope('subpixelconv_stage1'):
						net = conv2(net, 3, 256, 1, scope='conv')
						net = pixelShuffler(net, scale=2)
						net = prelu_tf(net)

				with tf.variable_scope('subpixelconv_stage2'):
						net = conv2(net, 3, 256, 1, scope='conv')
						net = pixelShuffler(net, scale=2)
						net = prelu_tf(net)

				with tf.variable_scope('output_stage'):
						net = conv2(net, 9, gen_output_channels, 1, scope='conv')
						net = tf.nn.tanh(net)
						#net = tf.sigmoid(net)

		return net

# Define the convolution building block
def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
		# kernel: An integer specifying the width and height of the 2D convolution window
		with tf.variable_scope(scope):
				if use_bias:
						return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
														activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
				else:
						return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
														activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
														biases_initializer=None)


# Define our tensorflow version PRelu
def prelu_tf(inputs, name='Prelu'):
		with tf.variable_scope(name):
				alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
		pos = tf.nn.relu(inputs)
		neg = alphas * (inputs - abs(inputs)) * 0.5

		return pos + neg


def batchnorm(inputs, is_training):
		return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
												scale=False, fused=True, is_training=is_training)


# The implementation of PixelShuffler
def pixelShuffler(inputs, scale=2):
		size = tf.shape(inputs)
		batch_size = size[0]
		h = size[1]
		w = size[2]
		c = inputs.get_shape().as_list()[-1]

		# Get the target channel size
		channel_target = c // (scale * scale)
		channel_factor = c // channel_target

		shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
		shape_2 = [batch_size, h * scale, w * scale, 1]

		# Reshape and transpose for periodic shuffling for each channel
		input_split = tf.split(inputs, channel_target, axis=3)
		output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

		return output

def phaseShift(inputs, scale, shape_1, shape_2):
		# Tackle the condition when the batch is None
		X = tf.reshape(inputs, shape_1)
		X = tf.transpose(X, [0, 1, 3, 2, 4])

		return tf.reshape(X, shape_2)

def image_gradients(image):
	"""Returns image gradients (dy, dx) for each color channel.
	Both output tensors have the same shape as the input: [batch_size, h, w,
	d]. The gradient values are organized so that [I(x+1, y) - I(x, y)] is in
	location (x, y). That means that dy will always have zeros in the last row,
	and dx will always have zeros in the last column.
	Arguments:
		image: Tensor with shape [batch_size, h, w, d].
	Returns:
		Pair of tensors (dy, dx) holding the vertical and horizontal image
		gradients (1-step finite difference).
	Raises:
		ValueError: If `image` is not a 4D tensor.
	"""
	if image.get_shape().ndims != 4:
		raise ValueError('image_gradients expects a 4D tensor '
										 '[batch_size, h, w, d], not %s.', image.get_shape())
	image_shape = array_ops.shape(image)
	batch_size, height, width, depth = array_ops.unstack(image_shape)
	dy = image[:, 1:, :, :] - image[:, :-1, :, :]
	dx = image[:, :, 1:, :] - image[:, :, :-1, :]

	# Return tensors with same size as original image by concatenating
	# zeros. Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
	shape = array_ops.stack([batch_size, 1, width, depth])
	dy = array_ops.concat([dy, array_ops.zeros(shape, image.dtype)], 1)
	dy = array_ops.reshape(dy, image_shape)

	shape = array_ops.stack([batch_size, height, 1, depth])
	dx = array_ops.concat([dx, array_ops.zeros(shape, image.dtype)], 2)
	dx = array_ops.reshape(dx, image_shape)

	return dy, dx



