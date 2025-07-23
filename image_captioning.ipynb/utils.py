# utils.py

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx.get('<unk>', 0))

    def get_word(self, idx):
        return self.idx2word.get(idx, '<unk>')

    def __len__(self):
        return len(self.word2idx)
