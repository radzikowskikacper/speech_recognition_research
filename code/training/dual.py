#!/usr/bin/env python

import gym
import numpy as np
import tensorflow as tf
import os

from recognition.ctc import model as recognition_model
from speech_model import model as speech_model
#from generation import model as synthesis_model
from language_model import model as language_model

def load_models(session):
    models = dict()

    dev1 = '/gpu:1'
    dev0 = '/gpu:0'

    with tf.Session() as session:
        with tf.device(dev0):
            temp = recognition_model.load_existing_model(
                    '../data/umeerj/models/recognition/0_5000_500_2_40_0.0001_0.9_1_0.2_1_1/2017-09-22 10:22:52/', session)
            models['recognition'] = {'inputs' : temp[0], 'targets' : temp[1], 'seq_len' : temp[2],
                                         'input_dropout_keep' : temp[3], 'output_dropout_keep' : temp[4],
                                         'state_dropout_keep' : temp[5], 'affine_dropout_keep' : temp[6],
                                         'cost' : temp[7], 'ler' : temp[8], 'ler2' : temp[9], 'ler3' : temp[10],
                                         'dense_hypothesis' : temp[11], 'dense_targets' : temp[12], 'decoded' : temp[13],
                                         'optimizer' : temp[14]}

            temp = speech_model.load_existing_model(
                    '../data/umeerj/models/speech_model', session)
            models['speech_model'] = {'inputs' : temp[0], 'targets' : temp[1], 'seq_len' : temp[2],
                                         'input_dropout_keep' : temp[3], 'output_dropout_keep' : temp[4],
                                         'state_dropout_keep' : temp[5], 'affine_dropout_keep' : temp[6],
                                         'cost' : temp[7], 'ler' : temp[8], 'ler2' : temp[9], 'ler3' : temp[10],
                                         'dense_hypothesis' : temp[11], 'dense_targets' : temp[12], 'decoded' : temp[13],
                                         'optimizer' : temp[14]}

        #with tf.device(dev1):
            temp = synthesis_model.load_existing_model(
                    '../data/umeerj/models/synthesis', session)
            models['synthesis'] = {}

            temp = language_model.load_existing_model(
                    '../data/umeerj/models/language_model', session)
            models['language_model'] = {}

    return models

def sample_sentence():
    pass

def sample_recording():
    pass

def synth(sentence):
    pass

def recognize(recording, session, models):
    #TODO
    int_to_char = {}

    d, dh = session.run([models['recognize']['decoded'][0], models['recognize']['dense_hypothesis']],
                        {
                            models['recognize']['inputs']: test_input,
                            models['recognize']['seq_len'] : test_seq_len,
                            models['recognize']['input_dropout_keep'] : 1,
                            models['recognize']['output_dropout_keep']: 1,
                            models['recognize']['state_dropout_keep']: 1,
                            models['recognize']['affine_dropout_keep']: 1
                        }
                        )

    dh = [f[:np.where(f == -1)[0][0] if -1 in f else len(f)] for f in dh]
    decoded_results = [''.join([int_to_char[x] for x in hypothesis]) for hypothesis in np.asarray(dh)]
    decoded_results = [s.replace('<space>', ' ') for s in decoded_results]

    for h in decoded_results:
        print('{} -> {}\n'.format(data[j][1], h))
        j += 1

def estimate_sentence(sentence):
    pass

def estimate_recording(recording):
    pass

def main():
    models = load_models(None)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        sentence_plh = tf.placeholder(tf.int32, (1, None))

        recording = synth(sentence_plh)
        #TODO
        imm_reward = estimate_recording(recording)
        #TODO

        reconstructed_sentence = recognize(recording, sess, models)
        lt_reward = estimate_sentence(reconstructed_sentence)
        total_reward = lt_reward + imm_reward

        loss = -total_reward
        optimizer = tf.train.RMSPropOptimizer(0.01)

        #TODO
        training_step_1 = optimizer.minimize(loss)
        training_step_2 = optimizer.minimize(loss)

        while True:
            sentence = sample_sentence()

            _, _, rc_sentence = sess.run([training_step_1, training_step_2, reconstructed_sentence],
                     feed_dict={
                        sentence_plh : sentence
                     })

            print('{} ->\n{}'.format(sentence, rc_sentence))

main()