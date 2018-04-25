import tensorflow as tf
import numpy as np
import functools

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ------------------------------------------------------------------------------
# Generic Functions for making layers
# ------------------------------------------------------------------------------

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# ------------------------------------------------------------------------------
# Functions to prevent mass creation of variables
# ------------------------------------------------------------------------------
# see: https://danijar.com/structuring-your-tensorflow-models/

# A helpful function used to prevent collisions
def define_scope(function):
    attribute = '_cache_' + function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):

        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

# same as the function above but it doesnt wrap functions in variable scopes
def lazy_property(function):
    attribute = '_cache_' + function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class NARX:
    """""
    Todo:
        Add summaries to all nodes!!!!!

    Properties:
        x_0 - a tf.placeholder to be used when training the network. This
        enables batch trainging as 0th dimension is None

    Note:
        All variables in this graph have been set to tf.float32

    """

    def __init__( self, x_1, x_2, y, queue_size, hidden_layers, learning_rate=1,
                  out_layer=None):
        """hidden_layers should be an array of sizes for each layer

        Args:
            x_1 - TDL for motor input (when using predict_spara)
            x_2 - TDL for observation input (when using predict_spara)
            y - target values to be used when training MLP
            queue_size - int of number previous steps to recall in TDLs
            hidden_layers - array if ints corresponding to number of nodes in hidden layers

        Kwargs:
            learning_rate - learnging rate for adam grad optimizer
            out_layer - not implemented


        TODO:
            update:
                hidden layers -
                    an array of dictionaries to be used as kwargs parameters
                    for tf.layers.dense() initializers
                out_layer -
                    kwargs/dictionary for output layer
            """

        # Inputs
        self.x_1 = x_1
        self.x_2 = x_2
        self.queue_size = queue_size
        queue_dim = (x_1.shape[1] + x_2.shape[1]) * queue_size
        self.x_0 = tf.placeholder(dtype=tf.float32, shape=[None,queue_dim],
                                  name='x_0_training_input')
        # Targets
        self.y = y
        # Optimization
        self.lr = learning_rate
        # Layers
        self.hidden_layers = hidden_layers

        with tf.variable_scope('fifo_queue'):
            self.x_1_queue = tf.get_variable('x_1_queue',dtype=tf.float32,
                                             initializer=tf.zeros([1, x_1.shape[
                                                 1]*queue_size]))
            self.x_2_queue = tf.get_variable('x_2_queue',dtype=tf.float32,
                                             initializer=tf.zeros([1,x_2.shape[
                                                 1]*queue_size]))
            self.queue = tf.get_variable('narx_queue',
                                         initializer=tf.zeros([1,queue_dim]),
                                         dtype=tf.float32)
            self.enqueue

        with tf.variable_scope('MLP',reuse=tf.AUTO_REUSE):
            # temporary variable to stop multiple graph additions to tensorboard
            self.__ffn__ = 0
            self.predict_para  # Parallel Prediction
            self.__ffn__ += 1
            self.predict_spara  # Series Parallel Prediction
            del(self.__ffn__)

        self.cost
        self.optimizer
        self.queue_dim = queue_dim
        # self.error

    # Currently retruns all values between -1 and 1
    def feed_forward_net(self, x, name):
        # Hidden layers
        layers = [x]
        for i in range(len(self.hidden_layers)):
            layer_name = 'h_{}'.format(i)
            layers.append(tf.layers.dense(layers[i], name=layer_name,
                                          **self.hidden_layers[i]))
            variable_summaries(layers[-1])

        # Prediction
        y_pred = tf.layers.dense(layers[-1], self.y.shape[1],
                                 name='y_pred', activation=tf.nn.tanh,
                                 reuse=tf.AUTO_REUSE)
        variable_summaries(y_pred)
        return y_pred

    # SERIES-PARALLEL
    @lazy_property
    def predict_spara(self):
        # pedict using concatenated queues
        with tf.name_scope('series_parallel'):
            return self.feed_forward_net(self.x_0, 'series_parallel')

    @define_scope
    def cost(self):
        mse = tf.losses.mean_squared_error(self.y, self.predict_spara)
        tf.summary.scalar('mean_squared_error',mse)
        return mse

    @define_scope
    def optimizer(self):
        # TODO: monitor loss in tensorboard
        optimizer = tf.train.AdamOptimizer(self.lr)
        return optimizer.minimize(self.cost)

    # PARALLEL
    @define_scope
    def queue_x_1(self):
        with tf.name_scope('x_1_enqueue'):
            x_1_split = tf.split(self.x_1_queue, self.queue_size, 1)
            x_1_split.append(self.x_1)
            return self.x_1_queue.assign(tf.concat(x_1_split[1:], 1))

    @define_scope
    def queue_x_2(self):
        with tf.name_scope('x_2_enqueue'):
            x_2_split = tf.split(self.x_2_queue, self.queue_size, 1)
            x_2_split.append(self.x_2)
            return self.x_2_queue.assign(tf.concat(x_2_split[1:], 1))

    @define_scope
    def enqueue(self):
        # TODO: monitor loss in tensorboard
        with tf.name_scope('enqueue'):
            x_1_con = self.queue_x_1
            x_2_con = self.queue_x_2


            concat = tf.concat([x_1_con,x_2_con], 1)
            return self.queue.assign(concat)

    @lazy_property
    def predict_para( self ):
        # pedict using concatenated queues
        with tf.name_scope('parallel'):
            return self.feed_forward_net(self.queue, 'parallel')

    #Todo: add an op to clear the queues
    def reset(self):
        """Just run sess.run(narx.reset())"""
        x_1 = self.x_1_queue.assign(tf.zeros(self.x_1_queue.shape))
        x_2 = self.x_2_queue.assign(tf.zeros(self.x_2_queue.shape))
        queue = self.queue.assign(tf.zeros(self.queue.shape))
        return x_1, x_2, queue