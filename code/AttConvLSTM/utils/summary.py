import os
import tensorflow as tf


class Summary(object):
    def __init__(self, saver, sess, CONF):
        self.save_model = CONF['SAVE_MODEL']
        self.load_model = CONF['LOAD_MODEL']
        self.sess = sess
        self.saver = saver
        self.path = CONF['PATH']
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.counter = 0
        self.delta = CONF['SAVE_DELTA']

    def set_current_ddpy(self, ddpg):
        self.ddpg = ddpg

    def save(self):

        self.saver.save(self.sess, self.path + '/model.ckpt')
        print("model has been saved")

    def load(self):
        if self.load_model and os.path.exists(self.path):
            self.saver.restore(self.sess, self.path + '/model.ckpt')
            print("model has been loaded")
        else:
            self.sess.run(tf.global_variables_initializer())
            print("model has been initialized")
