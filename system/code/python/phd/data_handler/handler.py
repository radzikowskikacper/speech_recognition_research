from collections import defaultdict

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

        for i in range(100):
            print("hello")

        pass

    def transformData_Kaldi(self, output_dir):
        pass