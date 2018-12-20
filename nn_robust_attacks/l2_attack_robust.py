## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np

from PIL import Image

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e2     # the initial constant c to pick as a first guess

transformations = [
	[1, 0, 0, 0, 1, 0, -0.0015, 0],
	[1, 0, 0, 0, 1, 0, -0.001, 0],
	[1, 0, 0, 0, 1, 0, 0, 0],
	[1, 0, 0, 0, 1, 0, 0.001, 0],
	[1, 0, 0, 0, 1, 0, 0.0015, 0],
]

class CarliniL2Robust:
    def __init__(self, sess, model, image_size, num_channels, num_labels, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST,
                 boxmin = 0, boxmax = 1):
        """
        The L_2 optimized attack. 

        This attack is the most efficient and should be used as the primary 
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """

        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size

        self.repeat = binary_search_steps >= 10

        self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = False

        shape = (batch_size,image_size,image_size,num_channels)
        
        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape,dtype=np.float32))

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        
        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        self.newimg = tf.tanh(modifier + self.timg) * self.boxmul + self.boxplus

        self.ll_img = tf.contrib.image.transform(self.newimg, transformations[0])
        self.l_img = tf.contrib.image.transform(self.newimg, transformations[1])
        self.c_img = tf.contrib.image.transform(self.newimg, transformations[2])
        self.r_img = tf.contrib.image.transform(self.newimg, transformations[3])
        self.rr_img = tf.contrib.image.transform(self.newimg, transformations[4])

        self.output_ll = model(self.ll_img)
        self.output_l = model(self.l_img)
        self.output_c = model(self.c_img)
        self.output_r = model(self.r_img)
        self.output_rr = model(self.rr_img)

        self.output = self.output_c

        # prediction BEFORE-SOFTMAX of the model
        self.model = model
        
        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-(tf.tanh(self.timg) * self.boxmul + self.boxplus)),[1,2,3])
        
        # compute the probability of the label class versus the maximum other
        real_ll = tf.reduce_sum((self.tlab)*self.output_ll,1)
        other_ll = tf.reduce_max((1-self.tlab)*self.output_ll - (self.tlab*10000),1)

        real_l = tf.reduce_sum((self.tlab)*self.output_l,1)
        other_l = tf.reduce_max((1-self.tlab)*self.output_l - (self.tlab*10000),1)

        real_c = tf.reduce_sum((self.tlab)*self.output_c,1)
        other_c = tf.reduce_max((1-self.tlab)*self.output_c - (self.tlab*10000),1)

        real_r = tf.reduce_sum((self.tlab)*self.output_r,1)
        other_r = tf.reduce_max((1-self.tlab)*self.output_r - (self.tlab*10000),1)

        real_rr = tf.reduce_sum((self.tlab)*self.output_rr,1)
        other_rr = tf.reduce_max((1-self.tlab)*self.output_rr - (self.tlab*10000),1)

        # TODO: this could also be max
        real = tf.reduce_mean(tf.concat([real_ll, real_l, real_c, real_r, real_rr], axis=0))
        other = tf.reduce_mean(tf.concat([other_ll, other_l, other_c, other_r, other_rr], axis=0))

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other-real+self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real-other+self.CONFIDENCE)

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const*loss1)
        self.loss = self.loss1+self.loss2
        
        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        
        self.init = tf.variables_initializer(var_list=[modifier]+new_vars)

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            print('tick',i)
            r.extend(self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size]))
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x,y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        # convert to tanh-space
        imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        upper_bound = np.ones(batch_size)*1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10]*batch_size
        o_bestscore = [-1]*batch_size
        o_bestattack = [np.zeros(imgs[0].shape)]*batch_size
        
        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0) # only difference
        
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(o_bestl2)
            pred = softmax(self.model.model.predict(o_bestattack[0].reshape(1, 64, 64, 3))[0])
            for i, p in sorted(enumerate(pred), key=lambda x: -x[1])[:3]:
                print(i, p)
            print("===")

            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]
    
            bestl2 = [1e10]*batch_size
            bestscore = [-1]*batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST})
            
            prev = np.inf
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack 
                _, l, l2s, scores, nimg = self.sess.run([self.train, self.loss, 
                                                         self.l2dist, self.output, 
                                                         self.newimg])

                if np.all(scores>=-.0001) and np.all(scores <= 1.0001):
                    if np.allclose(np.sum(scores,axis=1), 1.0, atol=1e-3):
                        if not self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK:
                            raise Exception("The output of model.predict should return the pre-softmax layer. It looks like you are returning the probability vector (post-softmax). If you are sure you want to do that, set attack.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = True")
                
                # print out the losses every 10%
                #if iteration%(self.MAX_ITERATIONS//10) == 0:
                if iteration % 100 == 0:
                    _, ll_img, l_img, c_img, r_img, rr_img, o_ll, o_l, o_c, o_r, o_rr = self.sess.run([self.train,
                        self.ll_img, self.l_img, self.c_img, self.r_img, self.rr_img,
                        self.output_ll, self.output_l, self.output_c, self.output_r, self.output_rr])

                    img = Image.fromarray(np.rint(ll_img[0] * 255).astype('uint8'), 'RGB')
                    img.save("lol/{}_{}ll_img.png".format(outer_step, iteration))
                    img = Image.fromarray(np.rint(l_img[0] * 255).astype('uint8'), 'RGB')
                    img.save("lol/{}_{}l_img.png".format(outer_step, iteration))
                    img = Image.fromarray(np.rint(c_img[0] * 255).astype('uint8'), 'RGB')
                    img.save("lol/{}_{}c_img.png".format(outer_step, iteration))
                    img = Image.fromarray(np.rint(r_img[0] * 255).astype('uint8'), 'RGB')
                    img.save("lol/{}_{}r_img.png".format(outer_step, iteration))
                    img = Image.fromarray(np.rint(rr_img[0] * 255).astype('uint8'), 'RGB')
                    img.save("lol/{}_{}rr_img.png".format(outer_step, iteration))
                    img = Image.fromarray(np.rint(nimg[0] * 255).astype('uint8'), 'RGB')
                    img.save("lol/{}_{}img.png".format(outer_step, iteration))

                    o_ll = softmax(o_ll[0])
                    o_l = softmax(o_l[0])
                    o_c = softmax(o_c[0])
                    o_r = softmax(o_r[0])
                    o_rr = softmax(o_rr[0])

                    print(np.argmax(o_ll), np.max(o_ll))
                    print(np.argmax(o_l), np.max(o_l))
                    print(np.argmax(o_c), np.max(o_c))
                    print(np.argmax(o_r), np.max(o_r))
                    print(np.argmax(o_rr), np.max(o_rr))
                    print("="*10)
                    
                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                    if l > prev*.9999:
                        break
                    prev = l

                # adjust the best result found so far
                for e,(l2,sc,ii) in enumerate(zip(l2s,scores,nimg)):
                    if l2 < bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 10

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack
