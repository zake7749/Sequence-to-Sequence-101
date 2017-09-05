class DataHelper(object):

    def __init__(self):
        self.word2idx = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 4}
        self.idx2word = {0: 'SOS', 1: 'EOS', 2: 'PAD', 4: 'UNK'}
        self.num_words = 4

    def build_vocab(self, data_path):
        """Construct the relation between words and indices"""
        with open(data_path, 'r', encoding='utf-8') as dataset:
            for sequence in dataset:
                sequence = sequence.strip('\n')
                words = self.split_sentence(sequence)
                for word in words:
                    if word not in self.word2idx:
                        self.word2idx[word] = self.num_words
                        self.idx2word[self.num_words] = word
                        self.num_words += 1

    def sentence_to_indices(self, sentence, add_EOS=False, add_SOS=False):
        """Transform a char sequence to index sequence
            :param sentence: a string composed with chars
            :param add_EOS: if true, add the <EOS> tag at the end of given sentence
            :param add_SOS: if true, add the <SOS> tag at the beginning of given sentence
        """
        index_sequence = [self.word2idx['SOS']] if add_SOS else []
        for word in self.split_sentence(sentence):
            index_sequence.append(self.word2idx[word])
        if add_EOS:
            index_sequence.append(self.word2idx['EOS'])
        return index_sequence

    def indices_to_setence(self, indices):
        """Transform a list of indices
            :param indices: a list
        """
        sentence = ""
        for id in indices:
            sentence += self.idx2word[id] + ' '
        return sentence

    def split_sentence(self, sentence):
        """Vary from languages. In our task, we simply split the sequence by space"""
        return sentence.split(' ')

    def __str__(self):
        str = "Vocab information:\n"
        for idx, word in self.idx2word.items():
            str += "Char: %s Index: %d\n" % (word, idx)
        return str

if __name__ == '__main__':
    data_helper = DataHelper()
    data_helper.build_vocab('Alphabet.txt')
    print(data_helper)

    test = "A C D E F G H I"
    print("Sentence before transformed:", test)
    ids = data_helper.sentence_to_indices(test)
    print("Indices sequence:", ids)
    sent = data_helper.indices_to_setence(ids)
    print("Sentence  after transformed:",sent)