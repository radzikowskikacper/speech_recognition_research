from collections import defaultdict
import os, shutil

class handler:
    data_path = ''
    data = dict()
    graded_data = dict()
    fnames = defaultdict(list)

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
                            f = parts[0].split('/')[2]
                            id = '-'.join(parts[0].split('/')[0:2])#parts[0].split('/')[0] + '/' + parts[0].split('/')[1]

                            found = False
                            for k, v in self.fnames.iteritems():
                                if f in v:
                                    for (i, entry) in enumerate(self.graded_data[k]):
                                        if entry[0] == parts[0]:
                                            if pos == 0:
                                                self.graded_data[k][i] = (entry[0], entry[1], entry[2], c, entry[4], entry[5])
                                            elif pos == 1:
                                                self.graded_data[k][i] = (entry[0], entry[1], entry[2], entry[3], c, entry[5])
                                            else:
                                                self.graded_data[k][i] = (entry[0], entry[1], entry[2], entry[3], entry[4], c)
                                            found = True
                                            break
                                    if found:
                                        break

                                    self.graded_data[k].append(
                                        (parts[0], id, gender, c if pos == 0 else -1.0, c if pos == 2 else -1.0, c if pos == 3 else -1.0))

    def transformData_Kaldi(self, output_dir, test_speakers):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs('{}/audio/test'.format(output_dir))
        os.makedirs('{}/audio/train'.format(output_dir))
        os.makedirs('{}/data/test'.format(output_dir))
        os.makedirs('{}/data/train'.format(output_dir))

        for s in test_speakers:
            os.makedirs("{}/audio/test/{}".format(output_dir, str(s)))
            #shutil.copytree('{}/wav/JE/{}'.format(self.data_path, s.replace('-', '/')),
            #                '{}/audio/test/{}'.format(output_dir, s))

        for root, dirs, files in os.walk(self.data_path + '/wav/JE'):
            path = root.split(os.sep)
            #print((len(path) - 1) * '---', os.path.basename(root))

            for file in files:
                parts = root.split('/')
                g = 'F' in parts[-1]
                id = '-'.join(parts[-2:])
                self.speakers[id] = g
                for k, v in self.fnames.iteritems():
                    if file in v:
                        #self.data[k].append(('/'.join(parts[-2:] + [file]), g, id))
                        if id not in test_speakers and not os.path.exists('{}/audio/{}/{}'.format(output_dir, 'train', id)):
                            os.makedirs('{}/audio/train/{}'.format(output_dir, id))
                        #shutil.copyfile('{}/wav/JE/{}'.format(self.data_path, '/'.join(parts[-2:] + [file])),
                         #               '{}/audio/{}/{}/{}'.format(output_dir, 'test' if id in test_speakers else 'train', id, file))
                        break

        fte = open('{}/data/test/spk2gender'.format(output_dir))
        ftr = open('{}/data/train/spk2gender'.format(output_dir))
        for k in sorted(self.speakers.iterkeys()):
            if k in test_speakers:
                fte.write('{} {}\n'.format(k, 'f' if self.speakers[k] else 'm'))
            else:
                ftr.write('{} {}\n'.format(k, 'f' if self.speakers[k] else 'm'))
        fte.close()
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


        