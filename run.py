from narx import NARX
import tensorflow as tf
import numpy as np

def main ():
    x_1 = tf.placeholder(dtype=tf.float32,
                         shape=[1,6],
                         name='x_1')
    x_2 = tf.placeholder(dtype=tf.float32,
                         shape=[1,6],
                         name='x_2')
    y = tf.placeholder(dtype=tf.float32,
                       shape=[None,1],
                       name='y')

    queue_size = 3
    nn = NARX(x_1, x_2, y, 3, [2], 1)

    # inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # x1_gate = np.array([[-1], [-1], [-1], [1]])
    # split = tf.split(inputs,4)
    # merge = tf.concat(split[1:],0)

    _x1 = np.ones([3,6])
    _x2 = np.ones([3,6])

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./tmp/tboard', sess.graph)
        # # test_writer = tf.summary.FileWriter('./tmp/tboard/test', sess.graph)
        tf.global_variables_initializer().run()

        # for i in range(10):
        #     _, pred = sess.run([nn.optimizer,nn.predict_spara],
        #                        feed_dict={ nn.x_0:inputs, y:x1_gate})
        #     print(pred)
        for i in range(len(_x1)):
            _ = sess.run(nn.enqueue,feed_dict={x_1:_x1[[i],:],x_2:_x2[[i],:]})
            print(_)
            print(sess.run(nn.predict_para))
            print()

        writer.close()



if __name__ == '__main__':
    main()