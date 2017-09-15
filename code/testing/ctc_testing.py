import numpy as np

from preprocessing.loading import ctc_preprocessing

def test_network(session, test_inputs, test_targets, batch_size, training_inputs_mean, training_inputs_std, data, mode,
                 decoded, dense_hypothesis, inputs, seq_len, input_dropout_keep, output_dropout_keep, state_dropout_keep,
                 affine_dropout_keep, int_to_char, model_folder_name, final_outcomes_fname):
    with open('{}/{}'.format(model_folder_name, final_outcomes_fname), 'a') as f:
        # Testing network
        i = 0
        f.write('---' + mode + '---\n')
        for test_inputs, _, test_seq_len, _ in \
                ctc_preprocessing.batch_generator(test_inputs, test_targets, batch_size, training_inputs_mean,
                                                  training_inputs_std, mode=mode):
            d, dh = session.run([decoded[0], dense_hypothesis], {inputs: test_inputs,
                                         seq_len: test_seq_len,
                                         input_dropout_keep: 1,
                                         output_dropout_keep: 1,
                                         state_dropout_keep: 1,
                                         affine_dropout_keep: 1
                                         }
                            )
            dh = [d[:np.where(d == -1)[0][0] if -1 in d else len(d)] for d in dh]
            decoded_results = [''.join([int_to_char[x] for x in hypothesis]) for hypothesis in np.asarray(dh)]
            decoded_results = [s.replace('<space>', ' ') for s in decoded_results]

            for h in decoded_results:
                f.write('{} -> {}\n'.format(data[i][1], h))
                i += 1
