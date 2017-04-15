import os, shutil, re, stat
from collections import defaultdict


class data:
    utt_to_text = dict()
    utt_to_file_time = dict()
    file_to_path = dict()
    utt_to_spk = dict()
    spk_to_utt = defaultdict(list)
    fname_to_text = dict()

    data_dir = ''


class kaldi_formatter():
    def __init__(self, data):
        self.data = data

    def __adjust_lexicon(self, lexicon_in, words, lexicon_out):
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

        with open(words) as f, open(lexicon_out, "wb") as lex:
            lex.write("OOV OOV\n")
            for line in f:
                line = line.strip()
                if line in ref.keys():
                    for pron in ref[line]:
                        lex.write(line + " " + pron + "\n")

    def __generate_phones(self, lexicon_in, phones_out):
        phones = set()
        with open(lexicon_in) as f:
            for line in f.readlines()[1:]:
                phones.update(line.split()[1:])
        with open(phones_out, 'w') as ftr:
            for p in sorted(phones):
                ftr.write(p + '\n')

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

        with open(os.path.join(output_dir, 'data', 'train', 'text'), 'w') as ftr:
            for k in sorted(self.data.utt_to_text.iterkeys()):
                ftr.write('{} {}\n'.format(k, self.data.utt_to_text[k].upper()))

        words = set()
        for v in self.data.utt_to_text.itervalues():
            words.update([re.sub(r'[^a-z A-Z\']', '', k).upper() for k in v.split()])
        with open(os.path.join(output_dir, 'data', 'train', 'words.txt'), 'w') as ftr:
            for k in sorted(words):
                ftr.write('{}\n'.format(k))

        with open(os.path.join(output_dir, 'data', 'train', 'segments'), 'w') as ftr:
            for k in sorted(self.data.utt_to_file_time.iterkeys()):
                ftr.write('{} {} {} {}\n'.format(k, *(self.data.utt_to_file_time[k])))

        with open(os.path.join(output_dir, 'data', 'train', 'wav.scp'), 'w') as ftr:
            for k in sorted(self.data.file_to_path.iterkeys()):
                ftr.write('{} {}\n'.format(k, self.data.file_to_path[k]))

        with open(os.path.join(output_dir, 'data', 'train', 'utt2spk'), 'w') as ftr:
            for k in sorted(self.data.utt_to_spk.iterkeys()):
                ftr.write('{} {}\n'.format(k, self.data.utt_to_spk[k]))

        with open(os.path.join(output_dir, 'data', 'train', 'spk2utt'), 'w') as ftr:
            for k in sorted(self.data.spk_to_utt.iterkeys()):
                ftr.write('{} {}\n'.format(k, ' '.join(sorted(self.data.spk_to_utt[k]))))

        with open(os.path.join(output_dir, 'data', 'local', 'lang', 'silence_phones.txt'), 'w') as ftr:
            ftr.write('SIL\nOOV\n')

        with open(os.path.join(output_dir, 'data', 'local', 'lang', 'optional_silence.txt'), 'w') as ftr:
            ftr.write('SIL\n')

        self.__adjust_lexicon(os.path.join(self.data.data_dir, '..', 'lexicon1.txt'),
                            os.path.join(output_dir, 'data', 'train', 'words.txt'),
                            os.path.join(output_dir, 'data', 'local', 'lang', 'lexicon.txt'))

        self.__generate_phones(os.path.join(output_dir, 'data', 'local', 'lang', 'lexicon.txt'),
                             os.path.join(output_dir, 'data', 'local', 'lang', 'nonsilence_phones.txt'))

    def prepare_training_script_Kaldi(self, kaldi_dir, output_dir, output_file):
        with open('{}/path.sh'.format(output_dir), 'w') as ftr:
            ftr.write('''export KALDI_ROOT={}
            [ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
            export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
            [ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
            . $KALDI_ROOT/tools/config/common_path.sh
            export LC_ALL=C
            '''.format(kaldi_dir))

        with open('{}/conf/mfcc.conf'.format(output_dir), 'w') as ftr:
            ftr.write('--use-energy=false\n--sample-frequency=16000\n')

        with open(output_file, 'w') as f:
            f.write("#!/bin/bash\nset -e\n")
            f.write("utils/prepare_lang.sh data/local/lang 'OOV' data/local/ data/lang\n")
            f.write('train_cmd="run.pl"\ndecode_cmd="run.pl --mem 2G"\n')
            f.write('''mfccdir=mfcc
    x=data/train
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 16 $x exp/make_mfcc/$x $mfccdir
    steps/compute_cmvn_stats.sh $x exp/make_mfcc/$x $mfccdir\n''')
            f.write('utils/subset_data_dir.sh --first data/train 10000 data/train_10k\n')
        st = os.stat(output_file)
        os.chmod(output_file, st.st_mode | stat.S_IEXEC)
