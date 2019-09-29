from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from six.moves import xrange

import tensorflow as tf

from tensorflow.python.platform import gfile
import distance
import numpy as np

from model.cnn import CNN
from model.seq2seq_model import Seq2SeqModel
from util.data_gen import DataGen

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'attn_num_hidden', 128,
    'Number of hidden units in each layer in attention model.')

tf.app.flags.DEFINE_integer(
    'attn_num_layers', 2,
    'Number of layers in attention model.')

tf.app.flags.DEFINE_boolean(
    'use_gru', False,
    'Whether or not use GRU instead of LSTM.')

tf.app.flags.DEFINE_integer(
    'max_prediction_length', 11,
    'Maximum length of the predicted string.')

tf.app.flags.DEFINE_integer(
    'max_image_height', 500,
    'Maximum image height.')

tf.app.flags.DEFINE_integer(
    'max_image_width', 500,
    'Maximum image width.')
	
tf.app.flags.DEFINE_integer(
    'target_embedding_size', 10,
    'Embedding size for each target.')
	
tf.app.flags.DEFINE_integer(
    'gpu_id', 0,
    'ID of the GPU to use (-1: CPU).')

tf.app.flags.DEFINE_string(
    'output_file', '', 'Where to save the resulting file to.')

FLAGS = tf.app.flags.FLAGS

def prepare_image(img, width):
	"""Resize the image to a maximum height of `self.height` and maximum
	width of `self.width` while maintaining the aspect ratio. Pad the
	resized image to a fixed size of ``[self.height, self.width]``."""
	dims = tf.shape(img)
	
	img_data = tf.expand_dims(img, 0)
	
	height = tf.constant(DataGen.IMAGE_HEIGHT, dtype=tf.int32)
	height_float = tf.constant(DataGen.IMAGE_HEIGHT, dtype=tf.float64)

	max_width = tf.to_int32(tf.ceil(tf.truediv(dims[1], dims[0]) * height_float))
	max_height = tf.to_int32(tf.ceil(tf.truediv(width, max_width) * height_float))

	resized = tf.cond(
		tf.greater_equal(width, max_width),
		lambda: tf.cond(
			tf.less_equal(dims[0], height),
			lambda: tf.to_float(img_data),
			lambda: tf.image.resize_images(img_data, [height, max_width],
							method=tf.image.ResizeMethod.BICUBIC),
		),
		lambda: tf.image.resize_images(img_data, [max_height, width],
							method=tf.image.ResizeMethod.BICUBIC)
	)

	padded = tf.image.pad_to_bounding_box(resized, 0, 0, height, width)
	
	return padded

def build_graph(input_image,
				attn_num_hidden,
				attn_num_layers,
				use_gru,
				max_prediction_length,
				max_image_height,
				max_image_width,
				target_embedding_size):
				
	# We need resized width, not the actual width
	max_resized_width = 1. * max_image_width / max_image_height * DataGen.IMAGE_HEIGHT
	
	max_original_width = max_image_width
	max_width = int(math.ceil(max_resized_width))
	
	encoder_size = int(math.ceil(1. * max_width / 4))
	decoder_size = max_prediction_length + 2
	buckets = [(encoder_size, decoder_size)]

	img_data = prepare_image(input_image, max_width)
	
	num_images = tf.shape(img_data)[0]
	
	encoder_masks = []
	for i in xrange(encoder_size + 1):
		encoder_masks.append(
			tf.tile([[1.]], [num_images, 1])
		)

	decoder_inputs = []
	target_weights = []
	for i in xrange(decoder_size + 1):
		decoder_inputs.append(
			tf.tile([0], [num_images])
		)
		if i < decoder_size:
			target_weights.append(tf.tile([1.], [num_images]))
		else:
			target_weights.append(tf.tile([0.], [num_images]))
	
	cnn_model = CNN(img_data, False)
	
	conv_output = cnn_model.tf_output()
	perm_conv_output = tf.transpose(conv_output, perm=[1, 0, 2])
	attention_decoder_model = Seq2SeqModel(
		encoder_masks=encoder_masks,
		encoder_inputs_tensor=perm_conv_output,
		decoder_inputs=decoder_inputs,
		target_weights=target_weights,
		target_vocab_size=len(DataGen.CHARMAP),
		buckets=buckets,
		target_embedding_size=target_embedding_size,
		attn_num_layers=attn_num_layers,
		attn_num_hidden=attn_num_hidden,
		forward_only=True,
		use_gru=use_gru)

	table = tf.contrib.lookup.MutableHashTable(
		key_dtype=tf.int64,
		value_dtype=tf.string,
		default_value="",
		checkpoint=True)

	insert = table.insert(
		tf.constant(list(range(len(DataGen.CHARMAP))), dtype=tf.int64),
		tf.constant(DataGen.CHARMAP))
	
	with tf.control_dependencies([insert]):
		num_feed = []
		prb_feed = []

		for line in xrange(len(attention_decoder_model.output)):
			guess = tf.argmax(attention_decoder_model.output[line], axis=1)
			proba = tf.reduce_max(
					tf.nn.softmax(attention_decoder_model.output[line]), axis=1)
			num_feed.append(guess)
			prb_feed.append(proba)

		# Join the predictions into a single output string.
		trans_output = tf.transpose(num_feed)
		trans_output = tf.map_fn(
			lambda m: tf.foldr(
				lambda a, x: tf.cond(
					tf.equal(x, DataGen.EOS_ID),
					lambda: '',
					lambda: table.lookup(x) + a  # pylint: disable=undefined-variable
				),
				m,
				initializer=''
			),
			trans_output,
			dtype=tf.string
		)

		# Calculate the total probability of the output string.
		trans_outprb = tf.transpose(prb_feed)
		trans_outprb = tf.gather(trans_outprb, tf.range(tf.size(trans_output)))
		trans_outprb = tf.map_fn(
			lambda m: tf.foldr(
				lambda a, x: tf.multiply(tf.cast(x, tf.float64), a),
				m,
				initializer=tf.cast(1, tf.float64)
			),
			trans_outprb,
			dtype=tf.float64
		)

                
		prediction = tf.cond(
			tf.equal(tf.shape(trans_output)[0], 1),
			lambda: trans_output[0],
			lambda: trans_output,
		)
		probability = tf.cond(
			tf.equal(tf.shape(trans_outprb)[0], 1),
			lambda: trans_outprb[0],
			lambda: trans_outprb,
		)

		return prediction, probability

def main(_):
	#if not FLAGS.output_file:
	#	raise ValueError('You must supply the path to save to with --output_file')
	tf.logging.set_verbosity(tf.logging.INFO)
	with tf.Graph().as_default() as graph:
		
		input = tf.placeholder(name='input', dtype=tf.uint8, shape=[None, None, 3])
		
		device_id = '/cpu:0'
		if FLAGS.gpu_id >= 0:
			device_id = '/gpu:' + str(FLAGS.gpu_id)
		
		with tf.device(device_id):
		
			prediction, probability = build_graph(input,
				attn_num_hidden=FLAGS.attn_num_hidden,
				attn_num_layers=FLAGS.attn_num_layers,
				use_gru=FLAGS.use_gru,
				max_prediction_length=FLAGS.max_prediction_length,
				max_image_height=FLAGS.max_image_height,
				max_image_width=FLAGS.max_image_width,
				target_embedding_size=FLAGS.target_embedding_size)
		
			tf.identity(prediction, name='prediction')
			tf.sigmoid(probability, name='probability')
		
		graph_def = graph.as_graph_def()
		with gfile.GFile('exported-model/frozen_inference_graph.pb', 'wb') as f:
			f.write(graph_def.SerializeToString())

if __name__ == '__main__':
  tf.app.run()
