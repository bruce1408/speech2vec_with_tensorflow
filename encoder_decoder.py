# # coding=utf-8
# import tensorflow as tf
# from tensorflow.contrib import rnn
# from tensorflow.python.layers.core import Dense
# from tensorflow.python import debug as tf_debug
#
# from collections import defaultdict
# import numpy as np
#
# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)
# tf.set_random_seed(1)
# np.random.seed(1)
#
# timesteps = 2
# num_hidden = 64
# batch_size = 3
# rnn_size = num_hidden  # how to set this num
# word_embedding = {'nice': 0, 'to': 1, 'meet': 2}
# X = np.array([
#     [[1, 2, 3], [2, 3, 4]],
#     [[2, 3, 4], [2, 5, 7]],
#     [[4, 6, 8], [0, 0, 0]]
# ], dtype=np.float32)
# label = np.array([
#     [[2, 3, 5], [2, 4, 6]],
#     [[3, 6, 9], [9, 0, 4]],
#     [[4, 5, 6], [0, 0, 0]]
# ], dtype=np.int32)
#
# X_input = tf.placeholder(tf.float32, [None, timesteps, 3])
# Y_input = tf.placeholder(tf.float32, [None, timesteps, 3])
# input_sequence = [2, 2, 1]
#
# #  encoder_layer(num_hidden, input_sequence):
# lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
# lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
#
# encoder_output, encoder_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, X_input,
#                                                                  sequence_length=input_sequence, dtype=tf.float32)
#
# # encoder_states 大小为 [2 x 3 x 64] encoder_output shape [3 x 2 x 64]
# encoder_states = tf.concat(encoder_states, 2)  # [2 x 3 x 128]
# encoder_state = encoder_states  # 最后的states是[3 x 128]
#
#
# # # define decoder
# # def decodeing_layer(decoder_input, rnn_size, target_sequence, max_target_sequence, encoder_state):
# #     decoder_embeddings = tf.Variable(tf.random_uniform([10, 3]))
# #     decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size*2, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
# #     output_layer = Dense(10, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
# #     with tf.variable_scope('decode'):
# #         training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=label, sequence_length=input_sequence)
# #         # 构造decoder
# #         training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, training_helper, encoder_state, output_layer)
# #         training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True,
# #                                                                           maximum_iterations=tf.constant(2))
# #     # training the decoder
# #     with tf.variable_scope('decode', reuse=True):
# #         start_token = tf.tile(tf.constant([0], dtype=tf.int32), [3], name='start_token')
# #         predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, start_token, 3)
# #         predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, predicting_helper, encoder_state,
# #                                                              output_layer)
# #         predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder, impute_finished=True,
# #                                                                             maximum_iterations=tf.constant(2))
# #     return training_decoder_output, predicting_decoder_output
#
# # define decoder
# decoder_embeddings = tf.Variable(tf.random_uniform([10, 3]))
# decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size*2, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
# output_layer = Dense(10, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
# with tf.variable_scope('decode'):
#     training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=Y_input, sequence_length=input_sequence)
#     # 构造decoder
#     training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, training_helper, encoder_state, output_layer)
#     training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True, maximum_iterations=tf.constant(2))
# # training the decoder
# with tf.variable_scope('decode', reuse=True):
#     start_token = tf.tile(tf.constant([0], dtype=tf.int32), [3], name='start_token')
#     predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, start_token, 3)
#     predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, predicting_helper, encoder_state,
#                                                          output_layer)
#     predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder, impute_finished=True,
#                                                                         maximum_iterations=tf.constant(2))
#
#
#
#
# # with tf.variable_scope("decode"):
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     init.run()
#     # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "bruce-2.local:6064")
#     # print(sess.run(encoder_states, feed_dict={X_input: X}))
#     batch_word = sess.run(predicting_decoder_output, feed_dict={X_input: X, Y_input: label})
#     print(batch_word.shape)


class A:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hello__(self):
        self.c = self.x + self.y
        print(self.c)
        return self.c

    def get_shape(self):
         print(self.x)
         return self.x


a = A(3, 4)
a.get_shape()
a.__hello__()





