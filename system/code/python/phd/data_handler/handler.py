from collections import defaultdict
import os

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
                    self.fnames[parts[len(parts) - 1]].append(parts[0])

        pass

        list_of_files = []
        for (dirpath, dirnames, filenames) in os.walk(data_dir + '/lbl/'):
            for filename in filenames:
                if filename == 'scores.lst':
                    print(filename)
                    parts = filename.split('/')
                    gender = 'female' in parts[len(parts) - 3]
                    pos = 2
                    if 'segmental' in parts[len(parts) - 2]:
                        pos = 0
                    elif 'intonation' in parts[len(parts) - 2] or 'accent' in parts[len(parts) - 2]:
                        pos = 1

                    list_of_files.append(os.sep.join([dirpath, filename]))

                    with open(os.sep.join([dirpath, filename])) as f:
                        for current_line in f.readlines():
                            if '#' in current_line:
                                continue
                            parts = current_line.split('\\s+')
                            c = sum([int(x) for x in parts[1:]])
                            c /= len(parts) - 1
                            f = parts[0].split('/')[2]
                            id = parts[0].split('/')[0] + '/' + parts[0].split('/')[1]

                            for k, v in self.fnames.iteritems():
                                if f in v:
                                    for k2, v2 in self.data.iteritems():
                                        

        pass
    def transformData_Kaldi(self, output_dir):
        pass