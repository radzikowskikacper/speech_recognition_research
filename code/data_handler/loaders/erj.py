import os, wave, contextlib


class erj_loader():
    def __init__(self, data):
        self.data = data

    def __wav_length(self, path):
        with contextlib.closing(wave.open(path, 'r')) as f:
            return f.getnframes() / float(f.getframerate())

    def load_data(self, data_dir):
        self.data.data_dir = data_dir

        for filename in ['sentence1.tab', 'word1.tab']:
            with open(os.path.join(data_dir, 'doc', 'JEcontent', 'tab', filename)) as f:
                for current_line in f.readlines():
                    parts = current_line.split('.wav')
                    self.data.fname_to_text[parts[0]] = parts[-1].strip()

        for root, dirs, files in os.walk(os.path.join(data_dir, 'wav', 'JE')):
            parts = root.split(os.path.sep)
            for file in files:
                path = os.path.join(root, file)
                file = file.replace('.wav', '')
                spk = '-'.join(parts[-2:])
                file_id = spk + '-' + file
                tm = self.__wav_length(path)
                if tm == 0.0:
                    continue

                self.data.utt_to_text[file_id] = self.data.fname_to_text[file]
                self.data.utt_to_file_time[file_id] = (file_id, 0.0, tm)
                self.data.file_to_path[file_id] = path
                self.data.utt_to_spk[file_id] = spk
                self.data.spk_to_utt[spk].append(file_id)