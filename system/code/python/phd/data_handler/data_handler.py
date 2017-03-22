class data_handler:
    data = dict()
    fnames = dict()

    def load_data(self, data_dir):
        self.data = dict();
        self.fnames = dict();

        for filename in ['sentence1.tab', 'word1.tab']:
            with open(data_dir + '/doc/JEcontent/tab/' + filename) as f:
                for current_line in f.readlines():
                    parts = current_line.split('\\.wav')
                    parts[len(parts) - 1] = parts[len(parts) - 1].strip()


    def transformData_Kaldi(self, output_dir):
        pass