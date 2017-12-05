#!/usr/bin/env python

import gym
import numpy as np
import tensorflow as tf
import os

from recognition.ctc import model as recognition_model
from speech_model import model as speech_model
#from generation import model as synthesis_model
from language_model.karnna import model as language_model

def load_models(session):
    models = dict()

    dev1 = '/gpu:1'
    dev0 = '/gpu:0'

    with tf.Session() as session2:
            '''            
            temp = speech_model.load_existing_model(
                '../data/umeerj/models/speech_model', session)
            models['speech_model'] = {'inputs': temp[0], 'targets': temp[1], 'seq_len': temp[2],
                                      'input_dropout_keep': temp[3], 'output_dropout_keep': temp[4],
                                      'state_dropout_keep': temp[5], 'affine_dropout_keep': temp[6],
                                      'cost': temp[7], 'ler': temp[8], 'ler2': temp[9], 'ler3': temp[10],
                                      'dense_hypothesis': temp[11], 'dense_targets': temp[12], 'decoded': temp[13],
                                      'optimizer': temp[14]}

            # with tf.device(dev1):
            temp = synthesis_model.load_existing_model(
                '../data/umeerj/models/synthesis', session)
            models['synthesis'] = {}

            temp = language_model.load_existing_model(
                '../data/umeerj/models/language_model', session)
            '''

            temp = recognition_model.load_existing_model(
                '../data/umeerj/models/recognition/0_5000_500_2_40_0.0001_0.9_1_0.2_1_1/2017-09-22 10:22:52/', session)

            models['recognition'] = {'inputs': temp[0], 'targets': temp[1], 'seq_len': temp[2],
                                     'input_dropout_keep': temp[3], 'output_dropout_keep': temp[4],
                                     'state_dropout_keep': temp[5], 'affine_dropout_keep': temp[6],
                                     'cost': temp[7], 'ler': temp[8], 'ler2': temp[9], 'ler3': temp[10],
                                     'dense_hypothesis': temp[11], 'dense_targets': temp[12], 'decoded': temp[13],
                                     'optimizer': temp[14], 'logits' : temp[15], 'probabilities' : temp[16], 'mindex' : temp[17], 'score' : temp[18]}

            models['language_model'] = {'sampled_sentence' : tf.Variable([[10, 30, 40, 10, 20]], dtype = tf.int32, trainable=False)}

            temp = language_model.create_model(len(vocab), batch_size=1,
                                                                                                   num_steps=1,
                                                                                                   lstm_size=lstm_size,
                                                                                                   num_layers=num_layers,
                                                                                                   learning_rate=learning_rate)
            models['language_model'] = {'inputs' : temp[0]}

            init = tf.constant(np.random.rand(1, 5, 13), dtype=tf.float32)
            models['synthesis'] = {'synthesized_sentence' : tf.get_variable('var_name', initializer=init, dtype=tf.float32)}
            #models['synthesis']['synthesized_sentence'] *= tf.cast(models['language_model']['sampled_sentence'], tf.float32)
            models['a'] = tf.Variable([[[12, 23, 34, 45, 56, 67, 78, 89, 91, 1, 21, 32, 43],
                                        [54, 65, 76, 87, 98, 9, 10, 10, 29, 38, 47, 56, 78],
                                        [89, 90, 1, 12, 23, 34, 45, 56, 67, 78, 89, 90, 1],
                                        [9, 98, 87, 76, 65, 54, 43, 32, 21, 10, 9, 29, 38],
                                        [47, 56, 65, 74, 83, 92, 1, 55, 66, 77, 88, 99, 100]]], dtype=tf.float32, trainable=False)
            models['speech_model'] = {'distance' : tf.reduce_sum(tf.pow(models['synthesis']['synthesized_sentence'] - models['a'], 2))}

            #models['recognition']['recognized_sentence'] = tf.reduce_sum(models['synthesis']['synthesized_sentence'], 2) * tf.Variable(5.4)

    return models

def sample_sentence():
    return np.array([[4, 3, 2, 1, 5, 6, 7, 1, 2, 3, 4, 5, 2, 1, 3, 4]])
    pass

def sample_recording():
    pass

def synth(sentence):
    return np.random.random((1, 100, 13))
    pass

def recognize(recording, recording_seq_len, session, models):
    #TODO
    int_to_char = {}

    d, dh = session.run([models['recognition']['decoded'][0], models['recognition']['dense_hypothesis']],
                        {
                            models['recognition']['inputs']: recording,
                            models['recognition']['seq_len'] : recording_seq_len,
                            models['recognition']['input_dropout_keep'] : 1,
                            models['recognition']['output_dropout_keep']: 1,
                            models['recognition']['state_dropout_keep']: 1,
                            models['recognition']['affine_dropout_keep']: 1
                        }
                        )

    dh = [f[:np.where(f == -1)[0][0] if -1 in f else len(f)] for f in dh]
    decoded_results = [''.join([int_to_char[x] for x in hypothesis]) for hypothesis in np.asarray(dh)]
    decoded_results = [s.replace('<space>', ' ') for s in decoded_results]

    return
    j = 0
    for h in decoded_results:
        #print('{} -> {}\n'.format(data[j][1], h))
        j += 1

def estimate_sentence(sentence, models):
    return 0.1

def estimate_recording(recording, models):
    return 0.2

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        models = load_models(sess)

        sample_sentence_graph = models['language_model']['sampled_sentence']
        synthesized_sentence_graph = models['synthesis']['synthesized_sentence']
        #recognized_sentence_graph = models['recognition']['recognized_sentence']

        imm_reward_plh = models['speech_model']['distance']
        lt_reward_plh = -models['recognition']['score']
        synth_pretty_plh = tf.cast(synthesized_sentence_graph, tf.float32)
        #recog_pretty_plh = tf.cast(recognized_sentence_graph, tf.int32)

        total_reward = imm_reward_plh
        total_reward += lt_reward_plh

        loss = total_reward #* 100000
        optimizer = tf.train.RMSPropOptimizer(0.01, momentum=0.9)
        #optimizer = tf.train.AdamOptimizer()

        training_step = optimizer.minimize(loss)

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        i = 0
        while True:
            #TODO
            #sample sentence
            samples = [c for c in prime]

            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, checkpoint)
                new_state = sess.run(initial_state)
                for c in prime:
                    x = np.zeros((1, 1))
                    x[0, 0] = vocab_to_int[c]
                    feed = {inputs: x,
                            keep_probs: 1.,
                            initial_state: new_state}
                    l2, preds, new_state, xnhh = sess.run([logits2, prediction, final_state, xnh],
                                                          feed_dict=feed)
                print(l2)
                print(x)
                print(xnhh)
                c = pick_top_n(preds, len(vocab))
                samples.append(int_to_vocab[c])

                for i in range(n_samples):
                    x[0, 0] = c
                    feed = {inputs: x,
                            keep_probs: 1.,
                            initial_state: new_state}
                    preds, new_state = sess.run([prediction, final_state],
                                                feed_dict=feed)

                    c = pick_top_n(preds, len(vocab))
                    samples.append(int_to_vocab[c])

            return ''.join(samples)

            #TODO
            #synthesize speech
            middle_synthesized = sess.run(synthesized_sentence_graph)

            #TODO
            #loop

            _, loss, synthesized, distance, score, pps, mindexes = \
                sess.run([training_step, total_reward, synth_pretty_plh, imm_reward_plh, lt_reward_plh, models['recognition']['probabilities'], models['recognition']['mindex']],
                     feed_dict={
                         models['recognition']['inputs']: middle_synthesized,
                         models['recognition']['seq_len']: [np.array(middle_synthesized).shape[1]],
                         models['recognition']['input_dropout_keep']: 1,
                         models['recognition']['output_dropout_keep']: 1,
                         models['recognition']['state_dropout_keep']: 1,
                         models['recognition']['affine_dropout_keep']: 1
                     })

            print(loss)
            print(distance)
            print(-score)
            print(synthesized)
            continue
            break
            if i % 100 == 0:
                print(loss_v)
                print(synthesized)
                print(aa)
                print(recognized)
                #break
            i += 1
            continue
            score = sess.run(models['recognition']['score'],
                     feed_dict={
                         models['recognition']['inputs'] : np.random.random((1, 5, 13)),
                         models['recognition']['seq_len'] : np.array([5]),
                         models['recognition']['input_dropout_keep']: 1,
                         models['recognition']['output_dropout_keep']: 1,
                         models['recognition']['state_dropout_keep']: 1,
                         models['recognition']['affine_dropout_keep']: 1
                     })

            print(score)

main()