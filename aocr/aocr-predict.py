from __future__ import absolute_import

from skimage import io, transform
import cv2 as cv
import logging,time,argparse,os,glob
import tensorflow as tf
import numpy as np

'''
    预测脚本
'''

def parser():
    parser = argparse.ArgumentParser('JuNeng')
    parser.add_argument('--input', required=True, help='The test data')
    parser.add_argument('--model', required=True, help='pb file')
    parser.add_argument('--image_format', required=True, help='Image format.Example: jpg or png')
    args = parser.parse_args()
    return args

def recognize(jpg_path, pb_file_path, img_format, args):

    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    # Distribution of gpu. config.gpu_options.per_process_gpu_memory_fraction = 0.8 Specify the GPU usage ratio
    config.gpu_options.allow_growth = True

    # Read the graph.
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session(config=config) as sess:
            input_x = sess.graph.get_tensor_by_name("input_image_as_bytes:0")
            print("sess.graph", sess.graph)
            print("input_x", input_x)
            out_softmax_text = sess.graph.get_tensor_by_name("prediction:0")
            print (out_softmax_text)
            probability = sess.graph.get_tensor_by_name("probability:0")
            print(probability)

            # Recursively retrieves all images in the specified directory
            # for filename in glob.iglob("./picture/**/*.jpg", recursive=True):
            for filename in glob.iglob(jpg_path + os.sep + "**" + os.sep + "*." + img_format, recursive=True):
                # Read an image.
                with open(filename, 'rb') as img_file:
                    img_file_data = img_file.read()
                
                start_time = time.time()
                # Run the model
                output_feed = [sess.graph.get_tensor_by_name('prediction:0'), sess.graph.get_tensor_by_name('probability:0')]
                img_out_softmax = sess.run(output_feed, feed_dict={input_x:img_file_data})
                 
                end_time = time.time()
                runtime = end_time - start_time
                print('run time：%f' % (runtime * 1000) + 'ms')
                # Parsing the output
                text = img_out_softmax[0]
                text = text.decode('iso-8859-1')
                probability = img_out_softmax[1]
                print('Result: OK.testImg:{} , PredictResult: {} , Probability: {:.2f}'.format(filename.split(os.sep)[-1], text, probability))



def main(_):
    args = parser()
    assert args.input, '`input` missing.'
    assert args.model, '`model` missing.'
    assert args.model, '`image_format` missing.'

    jpg_path = args.input
    pb_file_path = args.model
    img_format = args.image_format
    #recognize("datasets/img/M_AEU626527_8.jpg", "exported-model/frozen_inference_graph.pb")
    #recognize("datasets/img/M_AEU626527_8.jpg", "exported-model/frozen_graph.pb")  # 使用官方到处模型方式生成模型！
    recognize(jpg_path, pb_file_path, img_format, args)
if __name__ == '__main__':
    tf.app.run()
