import tensorflow as tf
from keras.losses import categorical_crossentropy
import numpy as np

def iterative_fgsm(imgs, model, target_class, n_iterations, epsilon):
    for i in range(n_iterations):
        for j in len(imgs):
            grads = model.gradients(target_class, imgs[j])
            imgs[j] += grads * (epsilon / n_iterations)

TARGETED = True
N_ITERATIONS = 100
EPSILON = 0.01

class FGSM:
    def __init__(self, sess, model, batch_size = 1, 
                targeted = TARGETED, n_iterations= N_ITERATIONS,
                epsilon = EPSILON, boxmin = 0., boxmax = 1.):
        
        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.TARGETED = targeted
        self.N_ITERATIONS = n_iterations
        self.EPSILON = epsilon
        self.batch_size = batch_size

        shape = (batch_size,image_size,image_size,num_channels)

        # using C&W trick to assign variable in a more efficient way
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.n_iterations = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        self.epsilon = tf.Variable(np.zeros(batch_size), dtype=tf.float32)

        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels))
        self.assign_n_iterations = tf.placeholder(tf.float32, [batch_size])
        self.assign_epsilon = tf.placeholder(tf.float32, [batch_size])

        self.output = model.predict(self.timg)
        real = tf.reduce_sum((self.tlab)*self.output,1)
        other = tf.reduce_sum((1-self.tlab)*self.output,1)

        #if self.TARGETED:
        #    self.loss = tf.maximum(0.0, other-real)
        #else:
        #    self.loss = tf.maximum(0.0, real-other)
        self.loss = -categorical_crossentropy(self.tlab, self.output)

        # gradients returns a list of gradients which will always have just one element in our case
        self.grads = tf.gradients(self.loss, self.timg)[0]
        self.modifier = self.grads * (self.epsilon * image_size**2 / self.n_iterations)

        # use tanh to keep it in the desired interval (i.e. yielding a valid image)
        #self.boxmul = (boxmax - boxmin) / 2.
        #self.boxplus = (boxmin + boxmax) / 2.
        #self.newimg = tf.tanh(self.modifier + self.timg) * self.boxmul + self.boxplus
        self.newimg = tf.maximum(tf.minimum(self.modifier + self.timg, 1), 0)
        
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.n_iterations.assign(self.assign_n_iterations))
        self.setup.append(self.epsilon.assign(self.assign_epsilon))
        
    def attack(self, imgs, targets):
        r = []
        for i in range(0,len(imgs),self.batch_size):
            print("tick:", i)
            r.extend(self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size]))
        return np.array(r)

    def attack_batch(self, imgs, labs):
        batch_size = self.batch_size
        batch = imgs[:batch_size]
        batchlab = labs[:batch_size]

        assign_N_ITERATIONS = np.ones(batch_size) * self.N_ITERATIONS
        assign_EPSILON = np.ones(batch_size) * self.EPSILON

        for i in range(self.N_ITERATIONS):
            self.sess.run(self.setup, {
                self.assign_timg: batch,
                self.assign_tlab: batchlab,
                self.assign_n_iterations: assign_N_ITERATIONS,
                self.assign_epsilon: assign_EPSILON
            })
            batch = self.sess.run(self.newimg)

        return batch
