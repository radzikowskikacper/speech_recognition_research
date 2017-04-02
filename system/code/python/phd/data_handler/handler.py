from collections import defaultdict
import os, shutil, re

class handler:
    data_path = ''
    data = dict()
    graded_data = dict()
    fnames = defaultdict(list)
    fnames2 = dict()

    def load_data(self, data_dir):
        self.data_path = data_dir
        #file name, ID, gender, score 1, score 2, score 3
        self.graded_data = dict();
        self.data = dict()
        self.fnames = defaultdict(list)
        self.speakers = dict()

        for filename in ['sentence1.tab', 'word1.tab']:
            with open(data_dir + '/doc/JEcontent/tab/' + filename) as f:
                for current_line in f.readlines():
                    parts = current_line.split('.wav')
                    parts[len(parts) - 1] = parts[len(parts) - 1].strip()

                    if parts[len(parts) - 1] not in self.graded_data:
                        self.graded_data[parts[len(parts) - 1]] = list()
                        self.data[parts[len(parts) - 1]] = list()
                    self.fnames[parts[len(parts) - 1]].append(parts[0] + '.wav')
                    self.fnames2[parts[0] + '.wav'] = parts[-1]
        '''
        for root, dirs, files in os.walk(data_dir + '/wav/JE'):
            path = root.split(os.sep)
            print((len(path) - 1) * '---', os.path.basename(root))

            for file in files:
                parts = root.split('/')
                g = True if 'F' in parts[-1] else False
                id = '-'.join(parts[-2:])
                for k, v in self.fnames.iteritems():
                    if file in v:
                        self.data[k].append(('/'.join(parts[-2:] + [file]), g, id))
                        break
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

    def get_phonemes(self, word):
        return [word]

    def transformData_Kaldi(self, output_dir, test_speakers):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs('{}/local'.format(output_dir))
        os.makedirs('{}/conf'.format(output_dir))
        os.makedirs('{}/audio/test'.format(output_dir))
        os.makedirs('{}/audio/train'.format(output_dir))
        os.makedirs('{}/data/test'.format(output_dir))
        os.makedirs('{}/data/train'.format(output_dir))
        os.makedirs('{}/data/local/dict'.format(output_dir))

        for s in test_speakers:
            os.makedirs("{}/audio/test/{}".format(output_dir, str(s)))
            #shutil.copytree('{}/wav/JE/{}'.format(self.data_path, s.replace('-', '/')),
            #                '{}/audio/test/{}'.format(output_dir, s))

        te_utt = defaultdict(list)
        tr_utt = defaultdict(list)
        for root, dirs, files in os.walk(self.data_path + '/wav/JE'):
            path = root.split(os.sep)
            #print((len(path) - 1) * '---', os.path.basename(root))

            for file in files:
                parts = root.split('/')
                g = 'F' in parts[-1]
                id = '-'.join(parts[-2:])
                self.speakers[id] = g

                if id not in test_speakers:
                    if not os.path.exists('{}/audio/{}/{}'.format(output_dir, 'train', id)):
                        os.makedirs('{}/audio/train/{}'.format(output_dir, id))
                    tr_utt[id + '-' + file.replace('.wav', '')].extend(['{}/audio/train/{}/{}'.format(output_dir, id, file), self.fnames2[file], id])
                else:
                    te_utt[id + '-' + file.replace('.wav', '')].extend(['{}/audio/test/{}/{}'.format(output_dir, id, file), self.fnames2[file], id])
                #shutil.copyfile('{}/wav/JE/{}'.format(self.data_path, '/'.join(parts[-2:] + [file])),
                 #               '{}/audio/{}/{}/{}'.format(output_dir, 'test' if id in test_speakers else 'train', id, file))

        fte = open('{}/data/test/wav.scp'.format(output_dir), 'w')
        ftr = open('{}/data/train/wav.scp'.format(output_dir), 'w')
        for k in sorted(te_utt.iterkeys()):
            fte.write('{} {}\n'.format(k, te_utt[k][0]))
        for k in sorted(tr_utt.iterkeys()):
            ftr.write('{} {}\n'.format(k, tr_utt[k][0]))
        fte.close()
        ftr.close()

        fte = open('{}/data/test/text'.format(output_dir), 'w')
        ftr = open('{}/data/train/text'.format(output_dir), 'w')
        for k in sorted(te_utt.iterkeys()):
            fte.write('{} {}\n'.format(k, te_utt[k][1]))
        for k in sorted(tr_utt.iterkeys()):
            ftr.write('{} {}\n'.format(k, tr_utt[k][1]))
        fte.close()
        ftr.close()

        fte = open('{}/data/test/utt2spk'.format(output_dir), 'w')
        ftr = open('{}/data/train/utt2spk'.format(output_dir), 'w')
        for k in sorted(te_utt.iterkeys()):
            fte.write('{} {}\n'.format(k, te_utt[k][2]))
        for k in sorted(tr_utt.iterkeys()):
            ftr.write('{} {}\n'.format(k, tr_utt[k][2]))
        fte.close()
        ftr.close()

        fte = open('{}/data/test/spk2gender'.format(output_dir), 'w')
        ftr = open('{}/data/train/spk2gender'.format(output_dir), 'w')
        for k in sorted(self.speakers.iterkeys()):
            if k in test_speakers:
                fte.write('{} {}\n'.format(k, 'f' if self.speakers[k] else 'm'))
            else:
                ftr.write('{} {}\n'.format(k, 'f' if self.speakers[k] else 'm'))
        fte.close()
        ftr.close()

        ftr = open('{}/data/local/corpus.txt'.format(output_dir), 'w')
        for k in sorted(self.fnames.iterkeys(), key = lambda a: a.lower()):
            ftr.write('{}\n'.format(k))
        ftr.close()

        temp = dict()
        ftr = open('{}/data/local/dict/lexicon.txt'.format(output_dir), 'w')
        ftr.write('!SIL sil\n<UNK> spn\n')
        for k in self.fnames.iterkeys():
            k = re.sub(r'[^a-z A-Z]','', k)
            for w in k.lower().replace('.', '').replace(',', '').split():
                temp[w] = self.get_phonemes(w)
        for k in sorted(temp.iterkeys()):
            ftr.write('{} {}\n'.format(k, ' '.join(temp[k])))
        ftr.close()

        phonemes = list()
        ftr = open('{}/data/local/dict/nonsilence_phones.txt'.format(output_dir), 'w')
        for v in temp.itervalues():
            phonemes.extend(v)
        for p in sorted(set(phonemes)):
            ftr.write('{}\n'.format(p))
        ftr.close()

        ftr = open('{}/data/local/dict/silence_phones.txt'.format(output_dir), 'w')
        ftr.write('sil\nspn\n')
        ftr.close()

        ftr = open('{}/data/local/dict/optional_silence.txt'.format(output_dir), 'w')
        ftr.write('spn\n')
        ftr.close()

        os.symlink('{}/../../../kaldi/egs/wsj/s5/utils'.format(output_dir), '{}/utils'.format(output_dir))
        os.symlink('{}/../../../kaldi/egs/wsj/s5/steps'.format(output_dir), '{}/steps'.format(output_dir))
        os.symlink('{}/../../../kaldi/egs/voxforge/s5/local/score.sh'.format(output_dir), '{}/local/score.sh'.format(output_dir))

        ftr = open('{}/conf/decode.config'.format(output_dir), 'w')
        ftr.write('first_beam=10.0\nbeam=13.0\nlattice_beam=6.0')
        ftr.close()
        ftr = open('{}/conf/mfcc.conf'.format(output_dir), 'w')
        ftr.write('--use-energy=false')
        ftr.close()

        ftr = open('{}/cmd.sh'.format(output_dir), 'w')
        ftr.write('# Setting local system jobs (local CPU - no external clusters\nexport train_cmd=run.pl\nexport decode_cmd=run.pl')
        ftr.close()

        ftr = open('{}/path.sh'.format(output_dir), 'w')
        ftr.write('''# Defining Kaldi root directory\nexport KALDI_ROOT=`pwd`/../../../kaldi\n\n# Setting paths to useful tools\n''' \
                    '''export\nPATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:''' \
                    '''$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/src/sgmm2bin/:''' \
                    '''$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$PWD:$PATH\n\n'''
                    '''# Defining audio data directory (modify it for your installation directory!)\n''' \
                    '''export DATA_ROOT="{}/projects/research/phd/data/kaldi/audio"\n\n# Enable SRILM\n''' \
                    '''source $KALDI_ROOT/tools/env.sh\n\n# Variable needed for proper data sorting\nexport LC_ALL=C'''.format(os.path.expanduser('~')))
        ftr.close()

        ftr = open('{}/run.sh'.format(output_dir), 'w')
        ftr.write('''#!/bin/bash

. ./path.sh || exit 1
. ./cmd.sh || exit 1

nj=1       # number of parallel jobs - 1 is perfect for such a small data set
lm_order=1 # language model order (n-gram quantity) - 1 is enough for digits grammar

# Safety mechanism (possible running this script with modified arguments)
. utils/parse_options.sh || exit 1
[[ $# -ge 1 ]] && { echo "Wrong arguments!"; exit 1; }

# Removing previously created data (from last run.sh execution)
rm -rf exp mfcc data/train/spk2utt data/train/cmvn.scp data/train/feats.scp data/train/split1 data/test/spk2utt data/test/cmvn.scp data/test/feats.scp data/test/split1 data/local/lang data/lang data/local/tmp data/local/dict/lexiconp.txt

echo
echo "===== PREPARING ACOUSTIC DATA ====="
echo

# Needs to be prepared by hand (or using self written scripts):

# spk2gender  [<speaker-id> <gender>]
# wav.scp     [<uterranceID> <full_path_to_audio_file>]
# text           [<uterranceID> <text_transcription>]
# utt2spk     [<uterranceID> <speakerID>]
# corpus.txt  [<text_transcription>]

# Making spk2utt files
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

echo
echo "===== FEATURES EXTRACTION ====="
echo

# Making feats.scp files
mfccdir=mfcc
# Uncomment and modify arguments in scripts below if you have any problems with data sorting
# utils/validate_data_dir.sh data/train     # script for checking prepared data - here: for data/train directory
# utils/fix_data_dir.sh data/train          # tool for data proper sorting if needed - here: for data/train directory
steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir
steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/test exp/make_mfcc/test $mfccdir

# Making cmvn.scp files
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir
steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test $mfccdir

echo
echo "===== PREPARING LANGUAGE DATA ====="
echo

# Needs to be prepared by hand (or using self written scripts):

# lexicon.txt           [<word> <phone 1> <phone 2> ...]
# nonsilence_phones.txt    [<phone>]
# silence_phones.txt    [<phone>]
# optional_silence.txt  [<phone>]

# Preparing language data
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang

echo
echo "===== LANGUAGE MODEL CREATION ====="
echo "===== MAKING lm.arpa ====="
echo

loc=`which ngram-count`;
if [ -z $loc ]; then
   if uname -a | grep 64 >/dev/null; then
           sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64
   else
                   sdir=$KALDI_ROOT/tools/srilm/bin/i686
   fi
   if [ -f $sdir/ngram-count ]; then
                   echo "Using SRILM language modelling tool from $sdir"
                   export PATH=$PATH:$sdir
   else
                   echo "SRILM toolkit is probably not installed.
                           Instructions: tools/install_srilm.sh"
                   exit 1
   fi
fi

local=data/local
mkdir $local/tmp
ngram-count -order $lm_order -write-vocab $local/tmp/vocab-full.txt -wbdiscount -text $local/corpus.txt -lm $local/tmp/lm.arpa

echo
echo "===== MAKING G.fst ====="
echo

lang=data/lang
arpa2fst --disambig-symbol=#0 --read-symbol-table=$lang/words.txt $local/tmp/lm.arpa $lang/G.fst

echo
echo "===== MONO TRAINING ====="
echo

steps/train_mono.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono  || exit 1

echo
echo "===== MONO DECODING ====="
echo

utils/mkgraph.sh --mono data/lang exp/mono exp/mono/graph || exit 1
steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" exp/mono/graph data/test exp/mono/decode

echo
echo "===== MONO ALIGNMENT ====="
echo

steps/align_si.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_ali || exit 1

echo
echo "===== TRI1 (first triphone pass) TRAINING ====="
echo

steps/train_deltas.sh --cmd "$train_cmd" 2000 11000 data/train data/lang exp/mono_ali exp/tri1 || exit 1

echo
echo "===== TRI1 (first triphone pass) DECODING ====="
echo

utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph || exit 1
steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" exp/tri1/graph data/test exp/tri1/decode

echo
echo "===== run.sh script is finished ====="
echo''')
        ftr.close()

'''
        i = 0
        for k, v in self.graded_data.iteritems():
            for entry in v:
                if entry[1] in test_speakers:
                    print '[{}] put {} in {}'.format(i, entry[0], entry[1])
                    shutil.copyfile('{}/wav/JE/{}'.format(self.data_path, entry[0]), '{}/audio/test/{}/{}'.format(output_dir, entry[1], entry[0].split('/')[2]))
                    pass
                    i+=1
'''