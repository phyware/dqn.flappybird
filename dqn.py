#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
import random
from collections import deque
import time
import os
import glob
import matplotlib.pyplot as plt
import math
import io
#game
import sys
sys.path.append('game/')
import wrapped_flappy_bird as game
sys.path.append('tflib/')
import kernel_grid

GAME = 'bird' # the name of the game being played for log files
N_ACTIONS = 2 # number of valid actions
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
FRAME_PER_ACTION = 1
GAMMA = 0.99 # decay rate of past observations
MODEL_SAVE_PATH = 'saved_model'
FRAME_DIR = 'frames_states'
SUMMARY_DIR = 'summary'
s = None # input, state
a = None # input, action taken
y = None # input, target q
s_pltbuf = None # input, plt buf by matplotlib
conv1_pltbuf = None # input, plt buf by matplotlib
pool1_pltbuf = None # input, plt buf by matplotlib
conv2_pltbuf = None # input, plt buf by matplotlib
conv3_pltbuf = None # input, plt buf by matplotlib
net = None
optimizer = None
mean_readout_max = None
summary_q = None
summary_layer = None
summary_layer_pltbuf = None
merged_summary = None

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def conv_net(s, weights, biases):
    net = {}

    # hidden layers
    with tf.name_scope('Conv1'):
        net['conv1'] = tf.nn.relu(conv2d(s, weights['wc1'], 4) + biases['bc1']) # out: 20 20 32
    with tf.name_scope('Pool1'):
        net['pool1'] = max_pool_2x2(net['conv1']) # out: 10 10 32

    with tf.name_scope('Conv2'):
        net['conv2'] = tf.nn.relu(conv2d(net['pool1'], weights['wc2'], 2) + biases['bc2']) # out: 5 5 64

    with tf.name_scope('Conv3'):
        net['conv3'] = tf.nn.relu(conv2d(net['conv2'], weights['wc3'], 1) + biases['bc3']) # out: 5 5 64

    with tf.name_scope('Conv3_flat'):
        net['conv3_flat'] = tf.reshape(net['conv3'], [-1, 1600])

    with tf.name_scope('Full_Connect'):
        net['fc1'] = tf.nn.relu(tf.matmul(net['conv3_flat'], weights['wf1']) + biases['bf1']) # out: 512

    # readout layer
    with tf.name_scope('Readout'):
        net['readout'] = tf.matmul(net['fc1'], weights['out']) + biases['out'] # out: N_ACTIONS
    
    return net

def layer_to_grid_summary(layer, grid_Y, grid_X, tag):
    layer_first = tf.slice(layer, (0, 0, 0, 0), (1, -1, -1, -1))
    layer_first_transposed = tf.transpose(layer_first, (1, 2, 0, 3))
    layer_first_transposed_grid = kernel_grid.put_kernels_on_grid(layer_first_transposed, grid_Y, grid_X, 20)
    summary = tf.image_summary(tag, layer_first_transposed_grid)
    return summary

def pltbuf_to_summary(buf, tag):
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf, channels=1)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    # Add image summary
    summary = tf.image_summary(tag, image)
    return summary

def create_graph():
    global s, a, y, s_pltbuf, conv1_pltbuf, pool1_pltbuf, conv2_pltbuf, conv3_pltbuf
    global net, optimizer, mean_readout_max, summary_q, summary_layer, summary_layer_pltbuf, merged_summary
    
    # Model
    s = tf.placeholder(tf.float32, [None, 80, 80, 4], name='State')
    
    weights = {
        # 8x8 conv, 1 input of 4 frames, 32 outputs
        'wc1': weight_variable([8, 8, 4, 32]),
        # 4x4 conv, 32 inputs, 64 outputs
        'wc2': weight_variable([4, 4, 32, 64]),
        # 3x3 conv, 64 inputs, 64 outputs
        'wc3': weight_variable([3, 3, 64, 64]),
        # fully connected, 5*5*64 inputs, 512 outputs
        'wf1': weight_variable([1600, 512]),
        # 512 inputs, n_classes
        'out': weight_variable([512, N_ACTIONS])
    }

    biases = {
        'bc1': bias_variable([32]),
        'bc2': bias_variable([64]),
        'bc3': bias_variable([64]),
        'bf1': bias_variable([512]),
        'out': bias_variable([N_ACTIONS])
    }
    
    with tf.name_scope('Model'):
        net = conv_net(s, weights, biases)
    
    # Loss
    a = tf.placeholder(tf.float32, [None, N_ACTIONS], 'ActionTaken')
    y = tf.placeholder(tf.float32, [None], 'Target')
    with tf.name_scope('Loss'):
        readout_action_taken = tf.reduce_sum(tf.mul(net['readout'], a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - readout_action_taken))
    
    # SGD
    with tf.name_scope('SGD'):
        optimizer = tf.train.AdamOptimizer(1e-6).minimize(cost)
    
    # Eval
    with tf.name_scope('Eval'):
        mean_readout_max = tf.reduce_mean(tf.reduce_max(net['readout'], reduction_indices=1))
        
    # Summary
    with tf.name_scope('Summary'):
        summary_q = tf.scalar_summary('Q', mean_readout_max)
        # plot layer by tensorflow
        # s in this order, so reverse: t+3, t+2, t+1, t
        summary_s = layer_to_grid_summary(tf.reverse(s, [False, False, False, True]), 1, 4, 'tf_S')
        summary_conv1 = layer_to_grid_summary(net['conv1'], 8, 4, 'tf_conv1')
        summary_pool1 = layer_to_grid_summary(net['pool1'], 8, 4, 'tf_pool1')
        summary_conv2 = layer_to_grid_summary(net['conv2'], 16, 4, 'tf_conv2')
        summary_conv3 = layer_to_grid_summary(net['conv3'], 16, 4, 'tf_conv3')
        summary_layer = tf.merge_summary([summary_s, summary_conv1, summary_pool1, summary_conv2, summary_conv3])
        # plot layer by pyplot
        s_pltbuf = tf.placeholder(tf.string, name='s_pltbuf')
        conv1_pltbuf = tf.placeholder(tf.string, name='conv1_plybuf')
        pool1_pltbuf = tf.placeholder(tf.string, name='pool1_plybuf')
        conv2_pltbuf = tf.placeholder(tf.string, name='conv2_plybuf')
        conv3_pltbuf = tf.placeholder(tf.string, name='conv3_plybuf')
        s_pltbuf_summary = pltbuf_to_summary(s_pltbuf, 'matplotlib_s')
        conv1_pltbuf_summary = pltbuf_to_summary(conv1_pltbuf, 'matplotlib_conv1')
        pool1_pltbuf_summary = pltbuf_to_summary(pool1_pltbuf, 'matplotlib_pool1')
        conv2_pltbuf_summary = pltbuf_to_summary(conv2_pltbuf, 'matplotlib_conv2')
        conv3_pltbuf_summary = pltbuf_to_summary(conv3_pltbuf, 'matplotlib_conv3')
        summary_layer_pltbuf = tf.merge_summary(
            [s_pltbuf_summary, conv1_pltbuf_summary, pool1_pltbuf_summary, conv2_pltbuf_summary, conv3_pltbuf_summary])
        
        merged_summary = tf.merge_all_summaries()

def process_frame(frame):
    out_frame = cv2.cvtColor(cv2.resize(frame, (80, 80)), cv2.COLOR_BGR2GRAY)
    # trying using greyscale image, so comment below
    # ret, out_frame = cv2.threshold(out_frame,1,255,cv2.THRESH_BINARY)
    out_frame = np.reshape(out_frame, (80, 80, 1))
    return out_frame

def play_game_or_train(sess, n_observe=1000, n_explore=0, initial_epsilon=0, final_epsilon=0, frame_dir=''):
    global s, a, y, net, optimizer
    
    # open up a game state to communicate with emulator
    game_state = game.GameState()
    
    saver = tf.train.Saver(max_to_keep=None)

    # store the previous observations in replay memory
    D = deque()

    frame_file = ''
    state_file = ''
    if frame_dir and frame_dir.strip():
        frame_file = os.path.join(frame_dir, 'frame.{0:06d}.png')
        state_file = os.path.join(frame_dir, 'state.{0:06d}.npy')
    
    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(N_ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = process_frame(x_t)
    s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=2)

    # start playing
    epsilon = initial_epsilon
    t = 0
    while 'flappy bird' != 'angry bird':
        # choose an action epsilon greedily
        a_t = np.zeros([N_ACTIONS])
        action_type = 'Noop'
        action_index = 0 # do nothing by default
        readout_t = net['readout'].eval(feed_dict={s : [s_t]})[0]
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                action_type = 'Random'
                action_index = random.randrange(N_ACTIONS)
            else:
                action_type = 'Model'
                action_index = np.argmax(readout_t)
        a_t[action_index] = 1
        print('-----Action type {0: <6}, action index {1}-----'.format(action_type, action_index))

        # scale down epsilon
        if epsilon > final_epsilon and t > n_observe and n_explore != 0:
            epsilon -= (initial_epsilon - final_epsilon) / n_explore

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t, frame_file.format(t))
        x_t1 = process_frame(x_t1_colored)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
        if state_file:
            np.save(state_file.format(t), s_t1)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > n_observe:
            # observe only, no need to train
            if n_explore <= 0:
                break
            
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH_SIZE)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = net['readout'].eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            optimizer.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, GAME + '-dqn'), global_step = t)

        # print info
        state = ''
        if t <= n_observe:
            state = 'observe'
        elif t > n_observe and t <= n_observe + n_explore:
            state = 'explore_and_train'
        else:
            state = 'train'

        print('TIMESTEP', t, '/ STATE', state, \
            '/ EPSILON', epsilon, '/ ACTION', action_index, '/ REWARD', r_t, \
            '/ Q_MAX %e' % np.max(readout_t))
        
def train():
    for f in glob.glob(os.path.join(MODEL_SAVE_PATH, '*')):
        os.remove(f)
    for f in glob.glob(os.path.join(SUMMARY_DIR, '*')):
        os.remove(f)

    create_graph()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(SUMMARY_DIR, graph=tf.get_default_graph())
        play_game_or_train(sess, n_observe=10000, n_explore=1000000, initial_epsilon=0.5, final_epsilon=0.0001)      
        summary_writer.flush()

def load_last_model(sess):
    saver = tf.train.Saver(max_to_keep=None)
    checkpoint = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        model_path = checkpoint.model_checkpoint_path
        saver.restore(sess, model_path)
        print('Successfully loaded:', model_path)
        return True
    return False
        
def play():
    for f in glob.glob(os.path.join(FRAME_DIR, '*')):
        os.remove(f)
        
    create_graph()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        load_last_model(sess)
        play_game_or_train(sess, n_observe=1000, n_explore=0, initial_epsilon=0.0001, final_epsilon=0.0001, frame_dir=FRAME_DIR)
        
def eval_models():
    global s, summary_q
    if not os.path.exists(FRAME_DIR):
        return
    
    create_graph()

    states = [np.load(f) for f in glob.glob(os.path.join(FRAME_DIR, 'state*.npy'))]
    print('found {} states'.format(len(states)))

    models =  [f for f in glob.glob(os.path.join(MODEL_SAVE_PATH, GAME + '-dqn*')) if '.meta' not in f]
    models = sorted(models, key=lambda m: int(os.path.basename(m).replace(GAME + '-dqn-', '')))
    print('found {} models'.format(len(models)))
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(SUMMARY_DIR)

        for i in range(len(models)):
            saver.restore(sess, models[i])
            q_avg = summary_q.eval(feed_dict={s : states})
            summary_writer.add_summary(q_avg, i)

def layer_to_pltbuf(layer):
    layer = np.array(layer)
    n_filters = layer.shape[3]
    plt.figure(figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(n_filters / n_columns) + 1
    for i in range(n_filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(layer[0,:,:,i], interpolation="nearest", cmap="gray")
    # plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return buf.getvalue()
            
def vis_model():
    global s, s_pltbuf, conv1_pltbuf, pool1_pltbuf, conv2_pltbuf, conv3_pltbuf, summary_layer, summary_layer_pltbuf
    state_path = os.path.join(FRAME_DIR, 'state.000100.npy')
    print('State: {}'.format(state_path))
    state_to_show = np.load(state_path)

    create_graph()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        load_last_model(sess)
        
        summary_writer = tf.train.SummaryWriter(SUMMARY_DIR)
        summary_layer_value = summary_layer.eval(feed_dict={s : [state_to_show]})
        s_value = [state_to_show]
        conv1_value, pool1_value, conv2_value, conv3_value, summary_layer_value =\
            sess.run([net['conv1'], net['pool1'], net['conv2'], net['conv3'], summary_layer], feed_dict={s : s_value})
        summary_writer.add_summary(summary_layer_value, 0)
        summary_layer_pltbuf_value = summary_layer_pltbuf.eval(feed_dict={
                s_pltbuf: layer_to_pltbuf(s_value),
                conv1_pltbuf: layer_to_pltbuf(conv1_value),
                pool1_pltbuf: layer_to_pltbuf(pool1_value),
                conv2_pltbuf: layer_to_pltbuf(conv2_value),
                conv3_pltbuf: layer_to_pltbuf(conv3_value)})
        summary_writer.add_summary(summary_layer_pltbuf_value, 0)
            
def make_sure_dir(target_dir):
    if os.path.exists(target_dir):
        print('Exists: {}.'.format(target_dir))
    else:
        print('Creating: {}'.format(target_dir))
        os.makedirs(target_dir)

def make_sure_need_dirs():
    make_sure_dir(FRAME_DIR)
    make_sure_dir(MODEL_SAVE_PATH)
    
def main():
    if len(sys.argv) < 2:
        print('Please specify a mode')
        return
    
    make_sure_need_dirs()
    
    mode = sys.argv[1].lower()
    if mode == 'train':
        train()
    elif mode =='play':
        play()
    elif mode =='eval':
        eval_models()
    elif mode == 'visualize':
        vis_model()
    else:
        print('mode not recognized')
    
if __name__ == '__main__':
    main()