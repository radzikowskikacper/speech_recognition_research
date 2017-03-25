from collections import defaultdict
import os, shutil

class handler:
    data = dict()
    fnames = defaultdict(list)

    def load_data(self, data_dir):
        self.data = dict();
        self.fnames = defaultdict(list)

        for filename in ['sentence1.tab', 'word1.tab']:
            with open(data_dir + '/doc/JEcontent/tab/' + filename) as f:
                for current_line in f.readlines():
                    parts = current_line.split('.wav')
                    parts[len(parts) - 1] = parts[len(parts) - 1].strip()

                    if parts[len(parts) - 1] not in self.data:
                        self.data[parts[len(parts) - 1]] = list()
                    self.fnames[parts[len(parts) - 1]].append(parts[0] + '.wav')

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
                            id = parts[0].split('/')[0] + '/' + parts[0].split('/')[1]

                            found = False
                            for k, v in self.fnames.iteritems():
                                if f in v:
                                    for (i, entry) in enumerate(self.data[k]):
                                        if entry[0] == parts[0]:
                                            print("found")
                                            if pos == 0:
                                                self.data[k][i] = (entry[0], entry[1], entry[2], c, entry[4], entry[5])
                                            elif pos == 1:
                                                self.data[k][i] = (entry[0], entry[1], entry[2], entry[3], c, entry[5])
                                            else:
                                                self.data[k][i] = (entry[0], entry[1], entry[2], entry[3], entry[4], c)
                                            found = True
                                            break
                                    if found:
                                        break

                                    self.data[k].append(
                                        (parts[0], id, gender, c if pos == 0 else -1.0, c if pos == 2 else -1.0, c if pos == 3 else -1.0))


    def transformData_Kaldi(self, output_dir, test_speakers):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir + '/audio/test')
        os.makedirs(output_dir + '/audio/train')
        os.makedirs(output_dir + '/data/test')
        os.makedirs(output_dir + '/data/train')

        