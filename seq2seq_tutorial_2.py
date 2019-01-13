# coding=utf-8
import tensorflow as tf
import numpy as np
import helpers
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(1)
from tqdm import tqdm, trange



old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(1)

x = [[5, 7, 8], [6, 3], [3], [1]]
xt, xlen = helpers.batch(x)


PAD = 0
EOS = 1
vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

# encoder_decoder_input
encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
decoder_inputs = tf.placeholder(tf.int32, [None, None], name='decoder_inputs')

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
# encoder_inputs_embedded = tf.Print(encoder_inputs_embedded, [tf.shape(encoder_inputs_embedded)],
# message='encoder_inputs_embed')
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)
# decoder_inputs_embedded = tf.Print(decoder_inputs_embedded, [tf.shape(decoder_inputs_embedded)],
# message='decoder_inputs_embed')


# define encoder layer
encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded,
                                                         dtype=tf.float32,
                                                         time_major=False)
del encoder_outputs

# define decoder layer
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell,
    decoder_inputs_embedded,
    initial_state=encoder_final_state,
    dtype=tf.float32,
    time_major=False,
    scope='plain_decoder',)
# decoder_outputs shape = [None, None, 20]

decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)  # shape [batch, max_step, 10]
# decoder_logits = tf.Print(decoder_logits, [tf.shape(decoder_logits)], message='decoder_logits')
decoder_prediction = tf.argmax(decoder_logits, 2)  # shape[None, None] = [batch, max_step]

# optimizer
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32), logits=decoder_logits)
loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)
init = tf.global_variables_initializer()

# toy data and train the modle
# batch = [[6], [3, 4], [9, 8, 7]]
# batch, batch_length = helpers.batch(batch)
# print('batch_encoder:\n'+str(batch))
#
# din_, dlen_ = helpers.batch(np.ones(shape=(3, 1), dtype=np.int32))
# print('decoder inputs:\n'+str(din_))
#
# with tf.Session() as sess:
#     sess.run(init)
#     pred = sess.run(decoder_prediction, feed_dict={encoder_inputs: batch, decoder_inputs: din_})
#     print('decoder_pred:\n'+str(pred))


# normal training the data
batch_size = 10
batches = helpers.random_sequences(length_from=3, length_to=8, vocab_lower=2, vocab_upper=10,
                                   batch_size=batch_size)

print('head of the batch:')
for seq in next(batches)[:10]:
    print(seq)


def next_feed():
    batch = next(batches)
    # print('batch\n', batch)
    encoder_inputs_, _ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch([(sequence) + [EOS] for sequence in batch])
    decoder_inputs_, _ = helpers.batch([[EOS] + (sequence) for sequence in batch])

    # print('ecoder inputs\n', encoder_inputs_)
    # print('decoder inputs\n', decoder_inputs_)
    # print('decoder targets\n', decoder_targets_)

    return {encoder_inputs: encoder_inputs_,
            decoder_inputs: decoder_inputs_,
            decoder_targets: decoder_targets_}


loss_track = []
epochs = 30001
batches_in_epoch = 100
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    try:
        for batch in range(epochs):
            fd = next_feed()
            _, l, enfs= sess.run([train_op, loss, encoder_final_state], fd)
            loss_track.append(l)
            # print('enfs\n', enfs.__len__())
            if batch == 0 or batch % batches_in_epoch == 0:
                print('epoch {}'.format(batch))
                print('  minibatch loss: {}'.format(sess.run(loss, fd)))
                predict_ = sess.run(decoder_prediction, fd)
                for i, (inp, pred) in enumerate(zip(fd[encoder_inputs], predict_)):
                    print('  sample {}:'.format(i + 1))
                    print('    input     > {}'.format(inp))
                    print('    predicted > {}'.format(pred))
                    if i >= 2:
                        break
                print()
    except KeyboardInterrupt:
        print('training interrupted')
    print(sess.run(embeddings))
