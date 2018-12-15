'''
MIT License

Copyright (c) 2018 University of Washington, University of Michigan, and University of California Berkeley

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import keras
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.contrib.graph_editor import connect

import sys
from PIL import Image

class Physical:
    def __init__(self, sess, model, mask_path, max_iterations=2000):
        with Image.open(mask_path) as mask:
            mask = np.asarray(mask, dtype="uint8")
            mask = mask[:,:,:3]
            mask = mask.copy()

            # TODO: Use method from train_model.py
            assert mask.shape == (model.image_size, model.image_size, 3), mask.shape
            mask = mask / 255

        self.sess = sess
        self.model = model
        self.mask = mask
        self.max_iterations = max_iterations
        self.printability_optimization = False

        op, placeholders, varops = setup_attack_graph(sess, model, model.image_size, model.image_size, model.num_channels, nb_classes=model.num_labels,
            regloss="l1", printability_optimization=self.printability_optimization, printability_tuples=False, clipping=True,
            noise_clip_min=-20.0, noise_clip_max=20.0, noisy_input_clip_min=0, noisy_input_clip_max=1,
            attack_lambda=0.001, optimization_rate=0.25, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=0.0001)

        self.op = op
        self.pholders = placeholders
        self.varops = varops

        self.regloss = True

    def attack(self, inputs, targets):
        assert len(inputs) == 1
        assert len(targets) == 1

        tf_model = self.model.model

        # TODO: Find out why:
        # This call needs to be included, otherwise the
        # Adam-optimizer fails
        src_class = np.argmax(tf_model.predict(inputs[0:1])[0])
        

        feed_dict = {self.pholders['image_in']: inputs,
                     self.pholders['attack_target']: targets,
                     self.pholders['noise_mask']: self.mask}

        clean_model_loss = model_loss(self.pholders['attack_target'], 
                                      self.varops['adv_pred'], mean=True)

        if self.printability_optimization:
            printability_tuples = "30values.txt"
            feed_dict[self.pholders['printable_colors']] = get_print_triplets(printability_tuples, self.model.image_size)

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0) # only difference

        best_images = np.zeros(inputs.shape)
        prediction = [0] * len(best_images)

        for iteration in range(self.max_iterations):
            if self.max_iterations > 0 and iteration >= self.max_iterations:
                break

            _,  train_loss, mod_loss, noisy_in, noisy_classes = self.sess.run( \
                (self.op, \
                self.varops['adv_loss'], \
                self.varops['loss'], \
                self.varops['noisy_inputs'], \
                self.varops['adv_pred']) \
                , feed_dict=feed_dict)

            for i, img in enumerate(noisy_in):
                target_label = np.argmax(targets[i])
                noisy_classification = noisy_classes[i][target_label]
                if prediction[i] < noisy_classification:
                    prediction[i] = noisy_classification
                    best_images[i] = noisy_in[i]

            if self.regloss != "none":
                reg_loss = self.sess.run(self.varops['reg_loss'], feed_dict=feed_dict)
            else:
                reg_loss = 0

            if iteration % 20 == 0:
                pred = softmax(tf_model.predict(noisy_in[0:1])[0])
                print(np.argmax(pred), np.max(pred))
                sys.stdout.flush()

        for pred in tf_model.predict(best_images):
            softmax_pred = softmax(pred)
            print(np.argmax(softmax_pred), np.max(softmax_pred))

        return best_images

def l1_norm(tensor):
    '''
    Provides a Tensorflow op that computes the L1 norm of the given tensor
    :param tensor: the tensor whose L1 norm is to be computed
    :return: a TF op that computes the L1 norm of the tensor
    '''
    return tf.reduce_sum(tf.abs(tensor))

def l2_norm(tensor):
    '''
    Provides a Tensorflow op that computess the L2 norm of the given tensor
    :param tensor: the tensor whose L2 norm is to be computed
    :return: a TF op that computes the L2 norm of the tensor
    ''' 
    return tf.sqrt(tf.reduce_sum(tf.pow(tensor, 2)))

def l2_loss(tensor1, tensor2):
    '''
    Provides a Tensorflow op that computess the L2 loss (the Euclidean distance)
    between the tensors provided.
    :param tensor1: the first tensor
    :param tensor2: the other tensor
    :return: a TF op that computes the L2 distance between the tensors
    '''
    return tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(tensor1,tensor2), 2)))

def model_loss(y, model, mean=True):
    """
    MIT License

    Copyright (c) 2017 Google Inc., OpenAI and Pennsylvania State University

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    """
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :param mean: boolean indicating whether should return mean of loss
                 or vector of losses for each input of the batch
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """

    op = model.op
    if "softmax" in str(op).lower():
        logits, = op.inputs
    else:
        logits = model

    out = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)

    if mean:
        out = tf.reduce_mean(out)
    return out

def setup_attack_graph(sess, keras_model, input_rows, input_cols, nb_channels, nb_classes=43,
    regloss="l2", printability_optimization=True, printability_tuples=False, clipping=True,
    noise_clip_min=-20.0, noise_clip_max=20.0, noisy_input_clip_min=-0.5, noisy_input_clip_max=0.5,
    attack_lambda=0.001, optimization_rate=0.25, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=0.0001):
    '''
    This function sets up an attack graph
    based on the Robust Physical Perturbations
    optimization algorithm and returns the Tensorflow operation
    to run that graph, along with the model, session, variables, operations,
    and placeholders defined along the way in dictionaries
    addressed by their names.
    :return: a tuple of (operation, model, session, placeholders, varops)
    where operation is the operation to run,
    provided initially, session is the TF session used to run the attack,
    and placeholder and varops are dictionaries holding the placeholders,
    variables, and intermediate TF operations defined
    '''

    # place all placeholders in this dict
    # so that they can be returned for use outside of this function
    placeholders = {}

    # note that these are set to the size of the input,
    # resizing happens later (before building model) if different
    placeholders['image_in'] = tf.placeholder(tf.float32, \
            shape=(None, input_rows, input_cols, nb_channels),
            name="noiseattack/image_in")

    placeholders['attack_target'] = tf.placeholder(tf.float32, \
        shape=(None, nb_classes),
        name="noiseattack/attack_targe")

    # resize later
    placeholders['noise_mask'] = tf.placeholder(tf.float32, \
                                                shape= \
                                                (input_rows, \
                                                input_cols, \
                                                nb_channels), \
                                                name="noiseattack/noise_mask")

    if printability_optimization:
        ####!!! Assumption: the printable tuples were all expanded to match
        ### the size of the image, so one tuple (x, y, z) gets replicated 32x32 times
        placeholders['printable_colors'] = tf.placeholder(tf.float32, \
                                                          shape=(None, \
                                                          input_rows, \
                                                          input_cols, \
                                                          nb_channels), \
                                                          name="noiseattack/printable_colors")

    # will hold the variables and operations defined from now on
    varops = {}

    varops['noise'] = tf.Variable(tf.random_normal( \
        [input_rows, input_cols, nb_channels]), \
        name='noiseattack/noise', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'adv_var'])

    # the following operations are these:
    # noise: a clipped value of the noise
    # noise_mul: the multiplication of the noise by the mask
    # noisy_inputs: the addition of the masked noise to the image
    if clipping:
        varops['noise'] = tf.clip_by_value(varops['noise'], \
            noise_clip_min, noise_clip_max, \
            name="noiseattack/noise_clipped")
        varops['noise_mul'] = tf.multiply(placeholders['noise_mask'], varops['noise'], \
            name="noiseattack/noise_mul")
        varops['noisy_inputs'] = tf.clip_by_value(tf.add(placeholders['image_in'], \
                                                  varops['noise_mul']), \
                                noisy_input_clip_min, noisy_input_clip_max, \
                                name="noiseattack/noisy_inputs")
    else:
        varops['noise_mul'] = tf.multiply(placeholders['noise_mask'], varops['noise'], \
                                          name="noiseattack/noise_mul")
        varops['noisy_inputs'] = tf.add(placeholders['image_in'], varops['noise_mul'], \
                                        name="noiseattack/noisy_inputs")
 
    # instantiate the model
    # adv_pred is the output of the model for an image (or images) with noise
    varops['adv_pred'] = keras_model.model(varops['noisy_inputs'])
    #model = YadavModel(train=False, custom_input=varops['noisy_inputs'])

     # Regularization term to control size of perturbation
    if regloss != "none":
        if regloss == 'l1':
            varops['reg_loss'] = attack_lambda * \
                l1_norm(tf.multiply(placeholders['noise_mask'], varops['noise']))
        elif regloss == 'l2':
            varops['reg_loss'] = attack_lambda * \
                l2_norm(tf.multiply(placeholders['noise_mask'], varops['noise']))
        else:
            raise Exception("Regloss may only be none or l1 or l2. Now%s"%regloss)

    # Compares adv predictions to given predictions
    varops['loss'] = model_loss(placeholders['attack_target'], varops['adv_pred'], mean=True)

    if printability_optimization:
        ####!!! Assumption: the printable tuples were all expanded to match
        ### the size of the image, so one tuple (x, y, z) gets replicated 32x32 times
        varops['printab_pixel_element_diff'] = tf.squared_difference(varops['noise_mul'], \
            placeholders['printable_colors'])
        varops['printab_pixel_diff'] = tf.sqrt(tf.reduce_sum( \
            varops['printab_pixel_element_diff'], 3))
        varops['printab_reduce_prod'] = tf.reduce_prod(varops['printab_pixel_diff'], 0)
        varops['printer_error'] = tf.reduce_sum(varops['printab_reduce_prod'])
        if regloss != "none":
            varops['adv_loss'] = varops['loss'] + varops['reg_loss'] + varops['printer_error']
        else:
            varops['adv_loss'] = varops['loss'] + varops['printer_error']
    else:
        if regloss != "none":
            varops['adv_loss'] = varops['loss'] + varops['reg_loss']
        else:
            varops['adv_loss'] = varops['loss']

    optimization_op = tf.train.AdamOptimizer(learning_rate=optimization_rate, \
        beta1=adam_beta1, \
        beta2=adam_beta2, \
        epsilon=adam_epsilon).minimize(varops['adv_loss'], \
        var_list=tf.get_collection('adv_var'))

    return optimization_op, placeholders, varops

def get_print_triplets(printability_tuples, img_size):
    '''
    Reads the printability triplets from the specified file
    and returns a numpy array of shape (num_triplets, FLAGS.img_cols, FLAGS.img_rows, nb_channels)
    where each triplet has been copied to create an array the size of the image
    :return: as described 
    '''     
    p = []  
        
    # load the triplets and create an array of the speified size
    with open(printability_tuples) as f:
        for l in f:
            p.append(l.split(",")) 
    p = list(map(lambda x: [[x for _ in range(img_size)] for __ in range(img_size)], p))
    p = np.float32(p)
    p -= 0.5
    return p