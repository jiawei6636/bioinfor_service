# -*- coding: utf-8 -*-
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
from . import model


def DeepAVP(sequence_list):
    """
    DeepAVP
    :param sequence: such as ['SWLRDIWDWICEVLSD', 'ISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPW', ......]
    :return: {'class_id': [1, 0, ...], 'probabilities': [[0.99, 0.01], [0.001, 0.999], ......]}
    """
    # Preprocess to the sequence
    Alfabeto = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    max_length = 0
    # print(sequence_list)
    for item in sequence_list:
        # print(item)
        if len(item) > max_length:
            max_length = len(item)

    one_hot_data = []
    sequence_length = []
    for item in sequence_list:
        length_temp = len(item)
        data_temp = np.zeros(shape=[max_length, 20])
        for i in range(length_temp):
            for index in range(20):
                if item[i] == Alfabeto[index] or item[i] == Alfabeto[index].lower():
                    data_temp[i, index] = 1

        one_hot_data.append(data_temp)
        sequence_length.append(length_temp)

    # DeepAVP model prediction
    with tf.Graph().as_default():

        data_feed = tf.placeholder(dtype=tf.float32, shape=[None, max_length, 20])
        sequence_length_feed = tf.placeholder(dtype=tf.int32, shape=[None])
        keep_prob = tf.placeholder(tf.float32)

        logits = model.inference(data_feed, sequence_length_feed, keep_prob)
        predictions = model.prediction(logits)

        saver = tf.train.Saver()

        session_config = tf.ConfigProto()
        with tf.Session(config=session_config) as sess:
            ckpt = tf.train.get_checkpoint_state('./program/model_dir/')
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())

            pred = sess.run(predictions, feed_dict={data_feed: one_hot_data,
                                                    sequence_length_feed: sequence_length, keep_prob: 1.0})

        label = pred['label']
        probabilities = pred['probabilities']

    return label, probabilities


# # For test
# if __name__ == '__main__':
#     sequence_list = ['GPPISLERLDVGTNLGNAIAKLEAKELLESSDQI',
#                      'ISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPW',
#                      'ELSNIKENKCNGTDAKVKLIKQELDKYKNAVTELQ',
#                      'ELSNIKENKCNGTDAKVKLIKQELDKYKNAVTELQ',
#                      'ELSNIKENKCNGTDAKVKLIKQELDKYKNAVTELQ',
#                      'ELSNIKENKCNGTDAKVKLIKQELDKYKNAVTELQ']
#
#     file = open('./static/60_antiviral_peptide.txt', 'r')
#     content = filter(None, file.read().replace('\r', '').split('\n')) # Filter the 'None' line.
#     content = [i for i in content]
#
#     print(DeepAVP(content))
