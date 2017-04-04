from collections import defaultdict
import os, shutil, re, stat, wave, contextlib

class handler2:
    utt_to_text = dict()
    utt_to_file_time = dict()
    file_to_path = dict()
    utt_to_spk = dict()
    spk_to_utt = defaultdict(list)

    fname_to_text = dict()

    data_dir = ''

    def load_data(self, data_dir):
        self.data_dir = data_dir
        self.fname_to_text = dict()
        self.utt_to_text = dict()
        self.utt_to_file_time = defaultdict(list)
        self.file_to_path = dict()
        self.utt_to_spk = dict()
        self.spk_to_utt = defaultdict(list)

        for filename in ['sentence1.tab', 'word1.tab']:
            with open(os.path.join(data_dir, 'doc', 'JEcontent', 'tab', filename)) as f:
                for current_line in f.readlines():
                    parts = current_line.split('.wav')
                    self.fname_to_text[parts[0]] = parts[-1].strip()

        for root, dirs, files in os.walk(os.path.join(data_dir, 'wav', 'JE')):
            parts = root.split(os.path.sep)
            for file in files:
                path = os.path.join(root, file)
                file = file.replace('.wav', '')
                spk = '-'.join(parts[-2:])
                file_id = spk + '-' + file

                self.utt_to_text[file_id] = self.fname_to_text[file]
                self.utt_to_file_time[file_id] = (file_id, 0.0, self.wav_length(path))
                self.file_to_path[file_id] = path
                self.utt_to_spk[file_id] = spk
                self.spk_to_utt[spk].append(file_id)
        '''
        for (dirpath, dirnames, filenames) in os.walk(data_dir + '/lbl/'):
            for filename in filenames:
                if filename == 'scores.lst':
                    filename = os.sep.join([dirpath, filename])
                    parts = filename.split('/')
                    gender = 'female' in parts[len(parts) - 3]
                    pos = 2
                    if 'segmental' in parts[len(parts) - 2]:
                        pos = 0
                    elif 'intonation' in parts[len(parts) - 2] or 'accent' in parts[len(parts) - 2]:
                        pos = 1

                    with open(filename) as f:
                        for current_line in f.readlines():
                            if '#' in current_line:
                                continue
                            parts = current_line.split()
                            c = sum([int(x) for x in parts[1:]])
                            c /= len(parts) - 1
                            ff = parts[0].split('/')[2]
                            id = '-'.join(parts[0].split('/')[0:2])#parts[0].split('/')[0] + '/' + parts[0].split('/')[1]

                            for i, entry in enumerate(self.graded_data[self.fnames2[ff]]):
                                if entry[0] == parts[0]:
                                    if pos == 0:
                                        self.graded_data[self.fnames2[ff]][i] = (entry[0], entry[1], entry[2], c, entry[4], entry[5])
                                    elif pos == 1:
                                        self.graded_data[self.fnames2[ff]][i] = (entry[0], entry[1], entry[2], entry[3], c, entry[5])
                                    else:
                                        self.graded_data[self.fnames2[ff]][i] = (entry[0], entry[1], entry[2], entry[3], entry[4], c)
                                    break

                            self.graded_data[self.fnames2[ff]].append(
                                (parts[0], id, gender, c if pos == 0 else -1.0, c if pos == 1 else -1.0, c if pos == 2 else -1.0))
        '''

    def wav_length(self, path):
        with contextlib.closing(wave.open(path, 'r')) as f:
            return f.getnframes() / float(f.getframerate())

    def adjust_lexicon(self, lexicon_in, words, lexicon_out):
        ref = dict()

        with open(lexicon_in) as f:
            for line in f:
                line = line.strip()
                columns = line.split("\t", 1)
                word = columns[0]
                pron = columns[1]
                try:
                    ref[word].append(pron)
                except:
                    ref[word] = list()
                    ref[word].append(pron)

        lex = open(lexicon_out, "wb")

        with open(words) as f:
            lex.write("OOV OOV\n")
            for line in f:
                line = line.strip()
                if line in ref.keys():
                    for pron in ref[line]:
                        lex.write(line + " " + pron + "\n")
        lex.close()

    def generate_phones(self, lexicon_in, phones_out):
        phones = set()
        with open(lexicon_in) as f:
            for line in f.readlines()[1:]:
                phones.update(line.split()[1:])
        ftr = open(phones_out, 'w')
        for p in sorted(phones):
            ftr.write(p + '\n')
        ftr.close()

    def transformData_Kaldi_easier_tut(self, kaldi_dir, output_dir, test_speakers):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(os.path.join(output_dir, 'exp'))
        os.makedirs(os.path.join(output_dir, 'conf'))
        os.makedirs(os.path.join(output_dir, 'data', 'lang'))
        os.makedirs(os.path.join(output_dir, 'data', 'train'))
        os.makedirs(os.path.join(output_dir, 'data', 'local', 'lang'))

        os.symlink(os.path.join(kaldi_dir, 'egs', 'wsj', 's5', 'utils'), os.path.join(output_dir, 'utils'))
        os.symlink(os.path.join(kaldi_dir, 'egs', 'wsj', 's5', 'steps'), os.path.join(output_dir, 'steps'))
        os.symlink(os.path.join(kaldi_dir, 'src'), os.path.join(output_dir, 'src'))

        ftr = open('{}/path.sh'.format(output_dir), 'w')
        ftr.write('''export KALDI_ROOT={}
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
'''.format(kaldi_dir))
        ftr.close()

        ftr = open(os.path.join(output_dir, 'cmd.sh'), 'w')
        ftr.write('''export train_cmd="run.pl"
export decode_cmd="run.pl --mem 2G"
        '''.format(kaldi_dir))
        ftr.close()
        st = os.stat(os.path.join(output_dir, 'cmd.sh'))
        os.chmod(os.path.join(output_dir, 'cmd.sh'), st.st_mode | stat.S_IEXEC)

        ftr = open('{}/conf/mfcc.conf'.format(output_dir), 'w')
        ftr.write('--use-energy=false')
        ftr.write('--sample-frequency=16000')
        ftr.close()

        ftr = open(os.path.join(output_dir, 'data', 'train', 'text'), 'w')
        for k, v in self.utt_to_text.iteritems():
            ftr.write('{} {}\n'.format(k, v.upper()))
        ftr.close()

        words = set()
        for v in self.utt_to_text.itervalues():
            words.update([re.sub(r'[^a-z A-Z\']', '', k).upper() for k in v.split()])
        ftr = open(os.path.join(output_dir, 'data', 'train', 'words.txt'), 'w')
        for k in sorted(words):
            ftr.write('{}\n'.format(k))
        ftr.close()

        ftr = open(os.path.join(output_dir, 'data', 'train', 'segments'), 'w')
        for k, v in self.utt_to_file_time.iteritems():
            ftr.write('{} {} {} {}\n'.format(k, *v))
        ftr.close()

        ftr = open(os.path.join(output_dir, 'data', 'train', 'wav.scp'), 'w')
        for k, v in self.file_to_path.iteritems():
            ftr.write('{} {}\n'.format(k, v))
        ftr.close()

        ftr = open(os.path.join(output_dir, 'data', 'train', 'utt2spk'), 'w')
        for k, v in self.utt_to_spk.iteritems():
            ftr.write('{} {}\n'.format(k, v))
        ftr.close()

        ftr = open(os.path.join(output_dir, 'data', 'train', 'spk2utt'), 'w')
        for k, v in self.spk_to_utt.iteritems():
            ftr.write('{} {}\n'.format(k, ' '.join(v)))
        ftr.close()


        self.adjust_lexicon(os.path.join(self.data_dir, '..', 'lexicon1.txt'),
                            os.path.join(output_dir, 'data', 'train', 'words.txt'),
                            os.path.join(output_dir, 'data', 'local', 'lang', 'lexicon.txt'))

        self.generate_phones(os.path.join(output_dir, 'data', 'local', 'lang', 'lexicon.txt'),
                             os.path.join(output_dir, 'data', 'local', 'lang', 'nonsilence_phones.txt'))

        ftr = open(os.path.join(output_dir, 'data', 'local', 'lang', 'silence_phones.txt'), 'w')
        ftr.write('SIL\nOOV\n')
        ftr.close()

        ftr = open(os.path.join(output_dir, 'data', 'local', 'lang', 'optional_silence.txt'), 'w')
        ftr.write('SIL\n')
        ftr.close()