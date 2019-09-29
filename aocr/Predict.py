# TODO: update the readme with new parameters
# TODO: restoring a model without recreating it (use constants / op names in the code?)
# TODO: move all the training parameters inside the training parser
# TODO: switch to https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn instead of buckets

from __future__ import absolute_import

import sys
import argparse
import logging
import os,glob,time
import tensorflow as tf

from aocr.model.model import Model
from aocr.defaults import Config
from aocr.util import dataset
from aocr.util.data_gen import DataGen
from aocr.util.export import Exporter

tf.logging.set_verbosity(tf.logging.ERROR)


def process_args(args, defaults):

    parser = argparse.ArgumentParser()
    parser.prog = 'aocr'

    # Global arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--log-path', dest="log_path",
                             metavar=defaults.LOG_PATH,
                             type=str, default=defaults.LOG_PATH,
                             help=('log file path (default: %s)'
                                   % (defaults.LOG_PATH)))

    # Shared model arguments
    parser.add_argument('--visualize', default='visualize')
    
    parser.set_defaults(visualize=defaults.VISUALIZE)
    parser.set_defaults(load_model=defaults.LOAD_MODEL)
    parser.add_argument('--max-width', dest="max_width",
                              metavar=defaults.MAX_WIDTH,
                              type=int, default=defaults.MAX_WIDTH,
                              help=('max image width (default: %s)'
                                    % (defaults.MAX_WIDTH)))
    parser.add_argument('--max-height', dest="max_height",
                              metavar=defaults.MAX_HEIGHT,
                              type=int, default=defaults.MAX_HEIGHT,
                              help=('max image height (default: %s)'
                                    % (defaults.MAX_HEIGHT)))
    parser.add_argument('--max-prediction', dest="max_prediction",
                              metavar=defaults.MAX_PREDICTION,
                              type=int, default=defaults.MAX_PREDICTION,
                              help=('max length of predicted strings (default: %s)'
                                    % (defaults.MAX_PREDICTION)))
    parser.add_argument('--full-ascii', dest='full_ascii', action='store_true',
                              help=('use lowercase in addition to uppercase'))
    parser.set_defaults(full_ascii=defaults.FULL_ASCII)
    parser.add_argument('--color', dest="channels", action='store_const', const=3,
                              default=defaults.CHANNELS,
                              help=('do not convert source images to grayscale'))
    parser.add_argument('--no-distance', dest="use_distance", action="store_false",
                              default=defaults.USE_DISTANCE,
                              help=('require full match when calculating accuracy'))
    parser.add_argument('--gpu-id', dest="gpu_id", metavar=defaults.GPU_ID,
                              type=int, default=defaults.GPU_ID,
                              help='specify a GPU ID')
    parser.add_argument('--use-gru', dest='use_gru', action='store_true',
                              help='use GRU instead of LSTM')
    parser.add_argument('--attn-num-layers', dest="attn_num_layers",
                              type=int, default=defaults.ATTN_NUM_LAYERS,
                              metavar=defaults.ATTN_NUM_LAYERS,
                              help=('hidden layers in attention decoder cell (default: %s)'
                                    % (defaults.ATTN_NUM_LAYERS)))
    parser.add_argument('--attn-num-hidden', dest="attn_num_hidden",
                              type=int, default=defaults.ATTN_NUM_HIDDEN,
                              metavar=defaults.ATTN_NUM_HIDDEN,
                              help=('hidden units in attention decoder cell (default: %s)'
                                    % (defaults.ATTN_NUM_HIDDEN)))
    parser.add_argument('--initial-learning-rate', dest="initial_learning_rate",
                              type=float, default=defaults.INITIAL_LEARNING_RATE,
                              metavar=defaults.INITIAL_LEARNING_RATE,
                              help=('initial learning rate (default: %s)'
                                    % (defaults.INITIAL_LEARNING_RATE)))
    parser.add_argument('--model-dir', '--job-dir', dest="model_dir",
                              type=str, default=defaults.MODEL_DIR,
                              metavar=defaults.MODEL_DIR,
                              help=('directory for the model '
                                    '(default: %s)' % (defaults.MODEL_DIR)))
    parser.add_argument('--target-embedding-size', dest="target_embedding_size",
                              type=int, default=defaults.TARGET_EMBEDDING_SIZE,
                              metavar=defaults.TARGET_EMBEDDING_SIZE,
                              help=('embedding dimension for each target (default: %s)'
                                    % (defaults.TARGET_EMBEDDING_SIZE)))
    parser.add_argument('--output-dir', dest="output_dir",
                              type=str, default=defaults.OUTPUT_DIR,
                              metavar=defaults.OUTPUT_DIR,
                              help=('output directory (default: %s)'
                                    % (defaults.OUTPUT_DIR)))
    parser.add_argument('--max-gradient-norm', dest="max_gradient_norm",
                              type=int, default=defaults.MAX_GRADIENT_NORM,
                              metavar=defaults.MAX_GRADIENT_NORM,
                              help=('clip gradients to this norm (default: %s)'
                                    % (defaults.MAX_GRADIENT_NORM)))
    parser.add_argument('--no-gradient-clipping', dest='clip_gradients', action='store_false',
                              help=('do not perform gradient clipping'))
    parser.set_defaults(clip_gradients=defaults.CLIP_GRADIENTS)


    # Testing
    #parser.add_argument('--visualize', dest='visualize', action='store_true',
    #                         help=('visualize attentions'))
    
    parser.set_defaults(phase='predict', steps_per_checkpoint=0, batch_size=1)
    parameters = parser.parse_args(args)
    return parameters


def main(args=None):

    # if args is None:
    #     args = sys.argv[1:]

    parameters = process_args(args, Config)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    


    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        if parameters.full_ascii:
            DataGen.set_full_ascii_charmap()

        model = Model(
            phase=parameters.phase,
            visualize=parameters.visualize,
            output_dir=parameters.output_dir,
            batch_size=parameters.batch_size,
            initial_learning_rate=parameters.initial_learning_rate,
            steps_per_checkpoint=parameters.steps_per_checkpoint,
            model_dir=parameters.model_dir,
            target_embedding_size=parameters.target_embedding_size,
            attn_num_hidden=parameters.attn_num_hidden,
            attn_num_layers=parameters.attn_num_layers,
            clip_gradients=parameters.clip_gradients,
            max_gradient_norm=parameters.max_gradient_norm,
            session=sess,
            load_model=parameters.load_model,
            gpu_id=parameters.gpu_id,
            use_gru=parameters.use_gru,
            use_distance=parameters.use_distance,
            max_image_width=parameters.max_width,
            max_image_height=parameters.max_height,
            max_prediction_length=parameters.max_prediction,
            channels=parameters.channels,
        )
    
        
        counter = 0
        #递归获取指定目录下的所有图片以统计待检测数据总数
        total_number = 0
        for i in glob.iglob('./datasets/img' + os.sep + "**" + os.sep + "*.jpg", recursive=True):
            total_number += 1        

        for filename in glob.iglob('./datasets/img' + os.sep + "**" + os.sep + "*.jpg", recursive=True):
        #for line in sys.stdin:
        #    filename = line.rstrip()
            counter = counter + 1  # 计数
            print("Progress:" + str(counter) + "/" + str(total_number))

            try:
                with open(filename, 'rb') as img_file:
                    img_file_data = img_file.read()
            except IOError:
                logging.error('Result: error while opening file %s.', filename)
                continue
            start_time = time.time()
            text, probability = model.predict(img_file_data)
            end_time = time.time()
            runtime = end_time - start_time
            print('run time：%f' % (runtime * 1000) + 'ms')

            #logging.info('Result: OK. %s %s', '{:.2f}'.format(probability), text)
            img_name = filename.split(os.sep)[-1] # 获取图片名
            logging.info('Result: OK.testImg:{} , PredictResult: {} , Probability: {:.2f}'.format(img_name, text, probability))
        else:
            raise NotImplementedError

if __name__ == "__main__":
    main()
