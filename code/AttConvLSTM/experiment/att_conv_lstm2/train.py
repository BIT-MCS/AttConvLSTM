import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from experiment.att_conv_lstm2.conf import *
import utils.layer_def_new as ld
import utils.BasicConvLSTMCell as BasicConvLSTMCell
from utils.summary import Summary
import os
from environment.preprocessed_data.database import *
import time
from utils.layer_def_new import *
from utils.drawplot import *
import argparse

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('weight_init', .1,
                          """weight init for fully connected layers""")


def network(inputs, hidden=None, state=None, lstm=True, isTest=False, isEncoder=True):
    l = int(CONF['CNN_LAYERS'] / 4 - 1)
    if isEncoder:
        conv1 = ld.conv_layer(inputs, 3, 2, 8, "encode_1", is_bn=CONF['IS_BN'])
        if not isTest:
            conv1 = tf.layers.dropout(conv1, rate=CONF['KEEP_PROB'])

        for i in range(l):
            layer_name = "encode_one_%s" % (i)
            conv1 = ld.conv_layer(conv1, 3, 1, 8, layer_name, is_bn=CONF['IS_BN'])
            if not isTest:
                conv1 = tf.layers.dropout(conv1, rate=CONF['KEEP_PROB'])

        # conv2
        conv2 = ld.conv_layer(conv1, 3, 1, 8, "encode_2", is_bn=CONF['IS_BN'])
        if not isTest:
            conv2 = tf.layers.dropout(conv2, rate=CONF['KEEP_PROB'])

        for i in range(l):
            layer_name = "encode_two_%s" % (i)
            conv2 = ld.conv_layer(conv2, 3, 1, 8, layer_name, is_bn=CONF['IS_BN'])
            if not isTest:
                conv1 = tf.layers.dropout(conv1, rate=CONF['KEEP_PROB'])

        # conv3
        conv3 = ld.conv_layer(conv2, 3, 2, 8, "encode_3", is_bn=CONF['IS_BN'])
        if not isTest:
            conv3 = tf.layers.dropout(conv3, rate=CONF['KEEP_PROB'])

        for i in range(l):
            layer_name = "encode_three_%s" % (i)
            conv3 = ld.conv_layer(conv3, 3, 1, 8, layer_name, is_bn=CONF['IS_BN'])
            if not isTest:
                conv3 = tf.layers.dropout(conv3, rate=CONF['KEEP_PROB'])

        # conv4
        conv4 = ld.conv_layer(conv3, 1, 1, 4, "encode_4", is_bn=CONF['IS_BN'])
        if not isTest:
            conv4 = tf.layers.dropout(conv4, rate=CONF['KEEP_PROB'])

        for i in range(l):
            layer_name = "encode_four_%s" % (i)
            conv4 = ld.conv_layer(conv4, 1, 1, 4, layer_name, is_bn=CONF['IS_BN'])
            if not isTest:
                conv4 = tf.layers.dropout(conv4, rate=CONF['KEEP_PROB'])

        y_0 = conv4

        if lstm:
            # conv lstm cell
            with tf.variable_scope('conv_lstm', initializer=tf.random_uniform_initializer(-.01, 0.1)):
                cell = BasicConvLSTMCell.BasicConvLSTMCell([8, 8], [3, 3], 4)
                if state is None and not isTest:
                    state = cell.zero_state(CONF['BATCH'], tf.float32)
                    print(" train hidden is initialted!")
                if state is None and isTest:
                    state = cell.zero_state(1, tf.float32)
                    print(" test hidden is initialted!")

                hidden, state = cell(y_0, state)
        else:
            y_1 = ld.conv_layer(y_0, 3, 1, 8, "encode_3")
        return hidden, state


def network2(inputs, isTest=False):
    # conv5
    # if not isTest:
    #     inputs=tf.layers.dropout(inputs,rate=CONF['KEEP_PROB'])
    l = int(CONF['CNN_LAYERS'] / 4 - 1)
    conv5 = ld.transpose_conv_layer(inputs, 1, 1, 8, "decode_5", is_bn=CONF['IS_BN'])
    if not isTest:
        conv5 = tf.layers.dropout(conv5, rate=CONF['KEEP_PROB'])

    for i in range(l):
        layer_name = "decode_conv5_%s" % (i)
        conv5 = ld.transpose_conv_layer(conv5, 1, 1, 8, layer_name, is_bn=CONF['IS_BN'])
        if not isTest:
            conv5 = tf.layers.dropout(conv5, rate=CONF['KEEP_PROB'])

    # conv6
    conv6 = ld.transpose_conv_layer(conv5, 3, 2, 8, "decode_6", is_bn=CONF['IS_BN'])
    if not isTest:
        conv6 = tf.layers.dropout(conv6, rate=CONF['KEEP_PROB'])

    for i in range(l):
        layer_name = "decode_conv6_%s" % (i)
        conv6 = ld.transpose_conv_layer(conv6, 3, 1, 8, layer_name, is_bn=CONF['IS_BN'])
        if not isTest:
            conv6 = tf.layers.dropout(conv6, rate=CONF['KEEP_PROB'])

    # conv7
    conv7 = ld.transpose_conv_layer(conv6, 3, 1, 8, "decode_7", is_bn=CONF['IS_BN'])
    if not isTest:
        conv7 = tf.layers.dropout(conv7, rate=CONF['KEEP_PROB'])

    for i in range(l):
        layer_name = "decode_conv7_%s" % (i)
        conv7 = ld.transpose_conv_layer(conv7, 3, 1, 8, layer_name, is_bn=CONF['IS_BN'])
        if not isTest:
            conv7 = tf.layers.dropout(conv7, rate=CONF['KEEP_PROB'])

    # x_1
    x_1 = ld.transpose_conv_layer(conv7, 3, 2, 2, "decode_8", linear=True, is_bn=False)  # set activation to linear

    for i in range(l):
        layer_name = "decode_conv8_%s" % (i)
        x_1 = ld.transpose_conv_layer(x_1, 3, 1, 2, layer_name, is_bn=CONF['IS_BN'])
        if not isTest:
            x_1 = tf.layers.dropout(x_1, rate=CONF['KEEP_PROB'])

    return x_1


#
#


# make a template for reuse
network_template = tf.make_template('network', network)
network_template2 = tf.make_template('network2', network2)


# https://github.com/tensorflow/nmt


def get_attention_weight_list(in_dim, out_dim):
    w_l = []
    for i in range(CONF['SEQ_START']):
        name = 'attention_weights_{0}'.format(i)
        w = variable_with_weight_decay(name, shape=[in_dim, out_dim], stddev=FLAGS.weight_init, wd=FLAGS.weight_decay,
                                       isCNN=False)
        w_l.append(w)
    return w_l


def loung_score(w, h_t, h_s):
    h_t = tf.expand_dims(h_t, axis=2)
    h_t = tf.transpose(h_t, perm=[0, 2, 1])

    h_s = tf.expand_dims(h_s, axis=2)
    result = tf.einsum('ijk,ikj->ij', tf.einsum('ijk,kl->ijl', h_t, w), h_s)
    return result


def score(w, h_t, h_s):
    h_t = tf.expand_dims(h_t, axis=2)
    h_t = tf.transpose(h_t, perm=[0, 2, 1])

    h_s = tf.expand_dims(h_s, axis=2)
    result = tf.einsum('ijk,ikj->ij', h_t, h_s)
    return result


def get_alpha_list(w_l, source_hidden_list, hidden_target, isTest=False):
    # a_l=[]
    # for i,w in enumerate(w_l):
    #     a_l.append(tf.exp(loung_score(w=w,h_s=source_hidden_list[i],h_t=hidden_target)))
    # a_l=tf.transpose(tf.stack(a_l),[1,0,2])
    # a_l_sum=tf.reduce_sum(tf.stack(a_l),axis=1)
    # a_l_sum=(1.0+SMALL)/(a_l_sum+SMALL)
    # if not isTest:
    #     a_l=tf.reshape(a_l,[CONF['BATCH'],CONF['SEQ_START']])
    # else:
    #     a_l = tf.reshape(a_l, [1, CONF['SEQ_START']])
    # result=tf.einsum('ij,ik->ij',a_l,a_l_sum)
    a_l = []
    for i, w in enumerate(w_l):
        a_l.append(loung_score(w=w, h_s=source_hidden_list[i], h_t=hidden_target))
    a_l = tf.transpose(tf.stack(a_l), [1, 0, 2])
    a_l = tf.stack(a_l)

    if not isTest:
        a_l = tf.reshape(a_l, [CONF['BATCH'], CONF['SEQ_START']])
    else:
        a_l = tf.reshape(a_l, [1, CONF['SEQ_START']])
    result = tf.nn.softmax(a_l)
    return result


def hidden_flatten(h, batch):
    return tf.reshape(h, [batch, 8 * 8 * 4])


def hidden_de_flatten(h, batch):
    return tf.reshape(h, [batch, 8, 8, 4])


def get_context_vector(h_s_l, alpha_l):
    # alpha  32*5
    h_s_l = tf.transpose(tf.stack(h_s_l), [1, 0, 2])  # 32*5*256
    c = tf.einsum('ijk,ij->ik', h_s_l, alpha_l)
    return c


def get_attention_vector_weights(in_dim, out_dim):
    name = 'attention_vector_weights_0'
    w1 = variable_with_weight_decay(name, shape=[in_dim, out_dim], stddev=FLAGS.weight_init, wd=FLAGS.weight_decay,
                                    isCNN=False)
    name = 'attention_vector_weights_1'
    w2 = variable_with_weight_decay(name, shape=[in_dim, out_dim], stddev=FLAGS.weight_init, wd=FLAGS.weight_decay,
                                    isCNN=False)
    return w1, w2


def get_new_hidden_target(hidden_target, context_vector, w1, w2, acti=tf.nn.tanh, single=CONF['IS_SINGLE']):
    if acti == None:
        return tf.matmul(hidden_target, w1) + tf.matmul(context_vector, w2)
    elif single:

        return acti(tf.matmul(hidden_target + context_vector, w1))
    else:
        return acti(tf.matmul(hidden_target, w1) + tf.matmul(context_vector, w2))


def train():
    """Train ring_net for a number of steps."""

    with tf.Graph().as_default():
        print("CNN layers", CONF['CNN_LAYERS'])
        # make inputs
        x = tf.placeholder(tf.float32, [None, CONF['SEQ_LENGTH'], 32, 32, 2])
        # make test inputs
        x_test = tf.placeholder(tf.float32, [None, CONF['TEST_SEQ_LENGTH'], 32, 32, 2])
        # possible dropout inside
        keep_prob = tf.placeholder("float")
        # x_dropout = tf.nn.dropout(x, keep_prob)
        # x_dropout_test=tf.nn.dropout(x_test,keep_prob)
        x_dropout = x
        x_dropout_test = x_test
        # create network
        x_unwrap = []

        # conv network
        source_hidden_list = []
        state = None
        x_1 = None

        decoder_start = tf.zeros(dtype=tf.float32, shape=[CONF['BATCH'], 32, 32, 2])
        att_w_lst = get_attention_weight_list(8 * 8 * 4, 8 * 8 * 4)
        attention_vector_w1, attention_vector_w2 = get_attention_vector_weights(8 * 8 * 4, 8 * 8 * 4)

        for i in range(CONF['SEQ_LENGTH']):
            if i < CONF['SEQ_START']:
                hidden, state = network_template(inputs=x_dropout[:, i, :, :, :], state=state, isEncoder=True)
                source_hidden_list.append(hidden_flatten(h=hidden, batch=CONF['BATCH']))
            else:
                # if i==CONF['SEQ_START']:
                #
                # attention_states [batch,seq_start,8*8*8]
                # encoder_outputs=tf.stack(encoder_outputs)
                # attention_states=tf.transpose(encoder_outputs,[1,0,2,3,4])
                # attention_states=tf.reshape(attention_states,[CONF['BATCH'],CONF['SEQ_START'],8*8*8])
                # attention_mechanism=tf.contrib.seq2seq.LuongAttention(num_units=8*8*4,memory=attention_states,memory_sequence_length=np.zeros([CONF['BATCH']]))
                # decoder_cell=tf.nn.rnn_cell.BasicLSTMCell(8*8*4)
                # decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                #     decoder_cell, attention_mechanism,
                #     attention_layer_size=8*8*4)
                # # print decoder_cell
                # hidden = decoder_cell

                hidden_target, state = network_template(inputs=decoder_start, state=state, isEncoder=True)
                alpha_list = get_alpha_list(w_l=att_w_lst, source_hidden_list=source_hidden_list
                                            , hidden_target=hidden_flatten(hidden_target, batch=CONF['BATCH']))
                context_vector = get_context_vector(h_s_l=source_hidden_list, alpha_l=alpha_list)
                new_hidden = get_new_hidden_target(hidden_target=hidden_flatten(hidden_target, CONF['BATCH']),
                                                   context_vector=context_vector,
                                                   w1=attention_vector_w1, w2=attention_vector_w2)
                new_hidden = hidden_de_flatten(new_hidden, batch=CONF['BATCH'])
                x_1 = network_template2(inputs=new_hidden)
                x_unwrap.append(x_1)
                decoder_start = x_1

        # pack them all together
        x_unwrap = tf.stack(x_unwrap)
        x_unwrap = tf.transpose(x_unwrap, [1, 0, 2, 3, 4])

        # conv network test
        x_unwrap_g = []
        source_hidden_list_g = []
        state_g = None
        x_1_g = None

        decoder_start_g = tf.zeros(dtype=tf.float32, shape=[1, 32, 32, 2])

        for i in range(CONF['SEQ_LENGTH']):
            if i < CONF['SEQ_START']:
                hidden_g, state_g = network_template(inputs=x_dropout_test[:, i, :, :, :], state=state_g,
                                                     isEncoder=True, isTest=True)
                source_hidden_list_g.append(hidden_flatten(h=hidden_g, batch=1))
            else:

                hidden_target_g, state_g = network_template(inputs=decoder_start_g, state=state_g, isEncoder=True,
                                                            isTest=True)
                alpha_list_g = get_alpha_list(w_l=att_w_lst, source_hidden_list=source_hidden_list_g
                                              , hidden_target=hidden_flatten(hidden_target_g, batch=1), isTest=True)
                context_vector_g = get_context_vector(h_s_l=source_hidden_list_g, alpha_l=alpha_list_g)
                new_hidden_g = get_new_hidden_target(hidden_target=hidden_flatten(hidden_target_g, 1),
                                                     context_vector=context_vector_g,
                                                     w1=attention_vector_w1, w2=attention_vector_w2,
                                                     )
                new_hidden_g = hidden_de_flatten(new_hidden_g, 1)
                x_1_g = network_template2(inputs=new_hidden_g, isTest=True)
                x_unwrap_g.append(x_1_g)
                decoder_start_g = x_1_g

        # pack them all together
        x_unwrap_g = tf.stack(x_unwrap_g)
        x_unwrap_g = tf.transpose(x_unwrap_g, [1, 0, 2, 3, 4])
        # calc total loss (compare x_t to x_t+1)
        # loss = tf.nn.l2_loss(x[:,CONF['SEQ_START'] + 1:, :, :, :] - x_unwrap[:,CONF['SEQ_START']:, :, :, :])
        env = Database_Preprocess()
        rmse_loss = tf.reduce_mean(tf.squared_difference(x[:, CONF['SEQ_START']:, :, :, :] * env.scale,
                                                         x_unwrap[:, 0:, :, :, :] * env.scale)) / CONF['BATCH']

        mape_loss = tf.reduce_mean(
            (tf.abs(
                x[:, CONF['SEQ_START']:, :, :, :] * env.scale - x_unwrap[:, 0:, :, :, :] * env.scale) + 1) / (x[:, CONF[
                                                                                                                       'SEQ_START']:,
                                                                                                              :, :,
                                                                                                              :] * env.scale + 1)
        ) * CONF['GAMMA']
        loss = rmse_loss + mape_loss
        # rmse_loss = tf.reduce_mean(tf.squared_difference(x[:, CONF['SEQ_START']:, :, :, :] * env.scale,
        #                                                  x_unwrap[:, 0:, :, :, :] * env.scale))
        # mape_loss = tf.reduce_mean(tf.abs(
        #     x[:, CONF['SEQ_START']:, :, :, :] * env.scale - x_unwrap[:, 0:, :, :, :] * env.scale) + 1.* env.scale / (
        #                                x[:, CONF['SEQ_START']:, :, :, :] * env.scale + 1.* env.scale))
        #
        # loss = rmse_loss + mape_loss
        # loss = tf.nn.l2_loss(x[:, CONF['SEQ_START']:, :, :, :]*env.scale-x_unwrap[:, 0:, :, :, :]*env.scale)

        tf.summary.scalar('loss', loss)

        # training
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(CONF['LR'],
                                                   global_step=global_step,
                                                   decay_steps=CONF['DECAY_STEPS'],
                                                   decay_rate=CONF['DECAY_RATE'])
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        add_global = global_step.assign_add(1)

        # Build a saver
        saver = tf.train.Saver(tf.global_variables())

        # Summary op
        summary_op = tf.summary.merge_all()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.5)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        summary = Summary(sess=sess, saver=tf.train.Saver(), CONF=CONF)
        summary.load()
        # Summary op
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        summary_writer = tf.summary.FileWriter(CONF['PATH'], graph_def=graph_def)

        # step_rmse = Drawplot(is_draw=True, xlabel='step', ylabel='rmse')
        min_rmse = 100000
        min_mape = 100000
        for step in range(CONF['STEP_START'], CONF['MAX_STEP']):

            dat = env.get_batch(CONF['BATCH'], CONF['SEQ_LENGTH'])
            t = time.time()

            if step >= 0:
                sess.run([add_global])

            _, loss_r, r_l, m_l, a_l, lr = sess.run([train_op, loss, rmse_loss, mape_loss, alpha_list, learning_rate],
                                                    feed_dict={x: dat, keep_prob: CONF['KEEP_PROB']})
            elapsed = time.time() - t

            if step % 100 == 0 and step != 0:
                summary_str = sess.run(summary_op, feed_dict={x: dat, keep_prob: CONF['KEEP_PROB']})
                summary_writer.add_summary(summary_str, step)
                print("time per batch is " + str(elapsed))
                print('step is', step, 'loss is ', loss_r, 'r_l', r_l, 'm_l', m_l, "LR is ", lr)

            assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

            if step % 1000 == 0:

                # dat_gif = env.get_test(CONF['TEST_SEQ_LENGTH'],is_test_data=True)
                # dat_res,a_l_g = sess.run([x_unwrap_g,alpha_list_g], feed_dict={x_test: dat_gif[:,:CONF['TEST_SEQ_LENGTH'],:,:,:], keep_prob:CONF['KEEP_PROB']})
                # print "Precision ",env.cal_one_rmse(gt=dat_gif[:, CONF['SEQ_START']:, :, :, :],res=np.stack(dat_res)[:, 0:, :, :, :])
                # print "attention",a_l_g
                mape_list = []
                dat_res_list = []
                a_l_g_list = []
                while not env.is_test_over():
                    dat_gif = env.get_test_one_by_one()
                    dat_res, a_l_g = sess.run([x_unwrap_g, alpha_list_g],
                                              feed_dict={x_test: dat_gif[:, :CONF['TEST_SEQ_LENGTH'], :, :, :],
                                                         keep_prob: CONF['KEEP_PROB']})
                    mape_list.append(
                        env.cal_one_mape(gt=dat_gif[:, CONF['SEQ_START']:, :, :, :],
                                         res=np.stack(dat_res)[:, 0:, :, :, :]))
                    # mape_list.append(
                    #     env.get_mape(y_true=dat_gif[:, CONF['SEQ_START']:, :, :, :],
                    #                      y_pred=np.stack(dat_res)[:, 0:, :, :, :]))
                    dat_res_list.append(env.cal_one_rmse(gt=dat_gif[:, CONF['SEQ_START']:, :, :, :],
                                                         res=np.stack(dat_res)[:, 0:, :, :, :]))
                    a_l_g_list.append(a_l_g)
                mean_rmse = np.mean(dat_res_list, axis=0)
                mean_mape = np.mean(mape_list)
                if mean_rmse < min_rmse and CONF['SAVE_MODEL']:
                    min_rmse = mean_rmse

                    summary.save()
                print("mean rmse is ", mean_rmse, 'mean mape is ', mean_mape, "mean alpha is", np.mean(a_l_g_list,
                                                                                                 axis=0), "min_rmse", min_rmse)
                # if mean_rmse<20 or step>10000:
                # step_rmse.show_step(current_step=step, current_total_reward=mean_rmse, draw=True)


def test():
    """Train ring_net for a number of steps."""
    with tf.Graph().as_default():

        # make test inputs
        x_test = tf.placeholder(tf.float32, [None, CONF['TEST_SEQ_LENGTH'], 32, 32, 2])
        # possible dropout inside
        keep_prob = tf.placeholder("float")

        # x_dropout_test=tf.nn.dropout(x_test,keep_prob)
        x_dropout_test = x_test
        # conv network test
        x_unwrap_g = []
        source_hidden_list_g = []
        state_g = None
        x_1_g = None

        decoder_start_g = tf.zeros(dtype=tf.float32, shape=[1, 32, 32, 2])

        att_w_lst = get_attention_weight_list(8 * 8 * 4, 8 * 8 * 4)
        attention_vector_w1, attention_vector_w2 = get_attention_vector_weights(8 * 8 * 4, 8 * 8 * 4)

        for i in range(CONF['SEQ_LENGTH']):
            if i < CONF['SEQ_START']:
                hidden_g, state_g = network_template(inputs=x_dropout_test[:, i, :, :, :], state=state_g,
                                                     isEncoder=True, isTest=True)
                source_hidden_list_g.append(hidden_flatten(h=hidden_g, batch=1))
            else:

                hidden_target_g, state_g = network_template(inputs=decoder_start_g, state=state_g, isEncoder=True,
                                                            isTest=True)
                alpha_list_g = get_alpha_list(w_l=att_w_lst, source_hidden_list=source_hidden_list_g
                                              , hidden_target=hidden_flatten(hidden_target_g, batch=1), isTest=True)
                context_vector_g = get_context_vector(h_s_l=source_hidden_list_g, alpha_l=alpha_list_g)
                new_hidden_g = get_new_hidden_target(hidden_target=hidden_flatten(hidden_target_g, 1),
                                                     context_vector=context_vector_g,
                                                     w1=attention_vector_w1, w2=attention_vector_w2,
                                                     )
                new_hidden_g = hidden_de_flatten(new_hidden_g, 1)
                x_1_g = network_template2(inputs=new_hidden_g, isTest=True)
                x_unwrap_g.append(x_1_g)
                decoder_start_g = x_1_g

        # pack them all together
        x_unwrap_g = tf.stack(x_unwrap_g)
        x_unwrap_g = tf.transpose(x_unwrap_g, [1, 0, 2, 3, 4])

        # Build a saver
        saver = tf.train.Saver(tf.global_variables())

        # Summary op
        summary_op = tf.summary.merge_all()

        sess = tf.Session()

        summary = Summary(sess=sess, saver=tf.train.Saver(), CONF=CONF)
        summary.load()
        # Summary op
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        summary_writer = tf.summary.FileWriter(CONF['PATH'], graph_def=graph_def)

        env = Database_Preprocess()

        truth = []
        predict = []

        mape_list = []
        dat_res_list = []
        a_l_g_list = []
        st = time.time()
        while not env.is_test_over():
            dat_gif = env.get_test_one_by_one()
            dat_res, a_l_g = sess.run([x_unwrap_g, alpha_list_g],
                                      feed_dict={x_test: dat_gif[:, :CONF['TEST_SEQ_LENGTH'], :, :, :],
                                                 keep_prob: CONF['KEEP_PROB']})

            mape_list.append(
                env.cal_one_mape(gt=dat_gif[:, CONF['SEQ_START']:, :, :, :], res=np.stack(dat_res)[:, 0:, :, :, :]))
            truth.append(dat_gif[:, CONF['SEQ_START']:, :, :, :])
            predict.append(np.stack(dat_res)[:, 0:, :, :, :])
            dat_res_list.append(
                env.cal_one_rmse(gt=dat_gif[:, CONF['SEQ_START']:, :, :, :], res=np.stack(dat_res)[:, 0:, :, :, :]))
            a_l_g_list.append(a_l_g)
        mean_rmse = np.mean(dat_res_list, axis=0)
        mean_mape = np.mean(mape_list)

        rmse_interval = env.cal_confidence_interval(dat_res_list)
        mape_interval = env.cal_confidence_interval(mape_list)
        print("mean rmse is ", mean_rmse, 'mean mape is', mean_mape, "mean alpha is", np.mean(a_l_g_list,
                                                                                              axis=0),
              ' rmse_interval ',
              rmse_interval, ' mape_interval ', mape_interval)
        truth = np.concatenate(truth, axis=0) * env.scale
        predict = np.concatenate(predict, axis=0) * env.scale
        # np.save(file=CONF['TRUTH_RESULT_PATH'], arr=truth)
        # np.save(file=CONF['PREDICT_RESULT_PATH'], arr=predict)
        tm = time.time() - st
        print(tm)


if __name__ == '__main__':
    # python / usr / local / lib / python2
    # .7 / dist - packages / tensorflow / tensorboard / tensorboard.py - -logdir = / home / linc / Desktop / convlstm_flow / experiment / data_2_attention / checkpoint
    #
    from utils.manager import GPUManager

    #
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('PATH', type=str, help='The directory of saving model')
    parser.add_argument('SEQ_LENGTH', type=str, help='Total sequence length')
    parser.add_argument('DATA_WIDTH', type=str, help='The resolution of data')
    args = parser.parse_args()
    gm = GPUManager()
    with gm.auto_choice():
        if CONF['IS_TEST']:
            test()
        else:
            train()
