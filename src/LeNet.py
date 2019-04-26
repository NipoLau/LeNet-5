import tensorflow as tf
import Mnist
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def LeNet(x):
    mu = 0.0
    sigma = 0.1

    # Layer 1 : Convolutional ; Input : 32 * 32 * 1 ; Output : 28 * 28 * 6
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5,5,1,6], mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1,1,1,1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)

    # Layer 2 : Max_Pooling ; Input : 28 * 28 * 6 ; Output : 14 * 14 * 6
    pool_1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # Layer 3 : Convolutional ; Input : 14 * 14 * 6 ; Output : 10 * 10 * 16
    conv2_w = tf.Variable(tf.truncated_normal(shape=[5,5,6,16], mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1,1,1,1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)

    # Layer 4 : Max_Pooling ; Input : 10 * 10 * 16 ; Output : 5 * 5 * 16
    pool_2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # Layer 5 : Flatten ; Input : 5 * 5 * 16 ; Output : 1 * 400
    fc1 = tf.layers.flatten(pool_2)

    # Layer 6 : Fully Connected ; Input : 1 * 400 ; Output : 1 * 120
    fc1_w = tf.Variable(tf.truncated_normal(shape=[400, 120], mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b
    fc1 = tf.nn.relu(fc1)

    # Layer 7 : Fully Connected ; Input : 1 * 120 ; Output : 1 * 84
    fc2_w = tf.Variable(tf.truncated_normal(shape=[120, 84], mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    fc2 = tf.nn.relu(fc2)

    # Layer 8 : Fully Connected ; Input : 1 * 84 ; Output : 1 * 10
    fc3_w = tf.Variable(tf.truncated_normal(shape=[84,10], mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b

    return logits

def Check_accuracy(sess, logits, x, X_val, y_val, batch_size):

    num_correct = 0
    for offset in range(0, X_val.shape[0], batch_size):
        X_batch = X_val[offset:offset+batch_size]
        y_batch = y_val[offset:offset+batch_size]

        logits_np = sess.run(logits, feed_dict={x: X_batch})

        num_correct += (logits_np.argmax(axis=1) == y_batch).sum()

    return float(num_correct) / X_val.shape[0]


def Train(X_train, y_train, X_val, y_val, batch_size, epochs, print_every):
    learning_rate = 1e-3

    with tf.device('/cpu:0'):
        # x : placeholder for a batch of input images ; y : placeholder for a batch of labels
        x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 1])
        y = tf.placeholder(dtype=tf.int32, shape=[None])

        # Forward pass
        logits = LeNet(x)

        # softmax loss function
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.one_hot(y, 10))
        loss = tf.reduce_mean(loss)

        # Adam Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_op):
        train_op = optimizer.minimize(loss)

    # Start Training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        t = 0
        for epoch in range(epochs):
            print('Epoch %d:'%epoch)
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[offset:offset+batch_size]
                y_batch = y_train[offset:offset+batch_size]
                loss_np, _ = sess.run([loss, train_op], feed_dict={x:X_batch, y:y_batch})
                if t % print_every == 0:
                    print('Iteration: %d | Loss : %f'%(t, loss_np))
                    accuracy = Check_accuracy(sess, logits, x, X_val, y_val, batch_size)
                    print('Accuracy : %f'%accuracy)
                    print()
                t += 1
        test_accuracy = Check_accuracy(sess, logits, x, X_test, y_test, batch_size)
        print('Test Accuracy : %f'%test_accuracy)

if __name__ == '__main__':
    X_train, y_train = Mnist.get_train_data()
    X_test, y_test = Mnist.get_test_data()
    X_train = np.pad(X_train, ((0,0), (2,2), (2,2), (0,0)), 'constant')
    X_test = np.pad(X_test, ((0,0), (2,2), (2,2), (0,0)), 'constant')
    # get validation data
    X_val, y_val = X_train[55000:60000], y_train[55000:60000]
    X_train, y_train = X_train[:55000], y_train[:55000]

    # Show Some Images
    mask = np.random.choice(np.arange(55000), 10, replace=False)
    for i, num in enumerate(mask):
        plt.subplot(2,5,i+1)
        plt.imshow(X_train[num].reshape(32,32))
        plt.title(y_train[num])
    plt.show()

    # set hyperparameter
    BATCH_SIZE = 150
    EPOCHS = 10
    PRINT_EVERY = 10

    Train(X_train, y_train, X_val, y_val, BATCH_SIZE, EPOCHS, PRINT_EVERY)