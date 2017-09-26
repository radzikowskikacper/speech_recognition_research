import numpy as np
import tensorflow as tf

from recognition.ctc import data as ctc_data
from . import model

def test_network(session, data_inputs, data_targets, batch_size, training_inputs_mean, training_inputs_std, data, mode,
                 decoded, dense_hypothesis, inputs, seq_len, input_dropout_keep, output_dropout_keep, state_dropout_keep,
                 affine_dropout_keep, int_to_char, model_folder_name, final_outcomes_fname):
    with open('{}/{}'.format(model_folder_name, final_outcomes_fname), 'a') as f:
        f.write('---' + mode + '---\n')
        j = 0
        for test_input, _, test_seq_len, _ in \
                ctc_data.batch_generator(data_inputs, data_targets, batch_size, training_inputs_mean,
                                         training_inputs_std):
            d, dh = session.run([decoded[0], dense_hypothesis],
                                {
                                    inputs: test_input,
                                    seq_len: test_seq_len,
                                    input_dropout_keep : 1,
                                    output_dropout_keep : 1,
                                    state_dropout_keep : 1,
                                    affine_dropout_keep : 1
                                }
                                )

            dh = [f[:np.where(f == -1)[0][0] if -1 in f else len(f)] for f in dh]
            decoded_results = [''.join([int_to_char[x] for x in hypothesis]) for hypothesis in np.asarray(dh)]
            decoded_results = [s.replace('<space>', ' ') for s in decoded_results]

            for h in decoded_results:
                f.write('{} -> {}\n'.format(data[j][1], h))
                j += 1

def test(model_folder_name):
    # Dividing the data
    training_data, training_inputs, training_targets, training_inputs_mean, training_inputs_std, \
    validation_data, validation_inputs, validation_targets, \
    testing_data, testing_inputs, testing_targets, \
    int_to_char, num_classes, num_samples = \
        ctc_data.load_data_sets('../data/umeerj/10k')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        inputs, targets, seq_len, input_dropout_keep, output_dropout_keep, state_dropout_keep, \
        affine_dropout_keep, cost, ler, ler2, ler3, dense_hypothesis, dense_targets, decoded, optimizer \
            = model.load_existing_model(model_folder_name, session)

        test_network(session, testing_inputs, testing_targets, 1, training_inputs_mean, training_inputs_std,
                     testing_data, 'testing', decoded, dense_hypothesis, inputs, seq_len, input_dropout_keep,
                     output_dropout_keep, state_dropout_keep, affine_dropout_keep, int_to_char, '.', 'testt.txt')