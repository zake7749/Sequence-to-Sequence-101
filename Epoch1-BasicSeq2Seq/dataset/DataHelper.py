import torch
import numpy as np

from torch.autograd import Variable


class Vocabulary(object):

    def __init__(self):
        self.char2idx = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
        self.idx2char = {0: 'SOS', 1: 'EOS', 2: 'PAD', 3: 'UNK'}
        self.num_chars = 4
        self.max_length = 0
        self.word_list = []

    def build_vocab(self, data_path):
        """Construct the relation between words and indices"""
        with open(data_path, 'r', encoding='utf-8') as dataset:
            for word in dataset:
                word = word.strip('\n')

                self.word_list.append(word)
                if self.max_length < len(word):
                    self.max_length = len(word)

                chars = self.split_sequence(word)
                for char in chars:
                    if char not in self.char2idx:
                        self.char2idx[char] = self.num_chars
                        self.idx2char[self.num_chars] = char
                        self.num_chars += 1

    def sequence_to_indices(self, sequence, add_eos=False, add_sos=False):
        """Transform a char sequence to index sequence
            :param sequence: a string composed with chars
            :param add_eos: if true, add the <EOS> tag at the end of given sentence
            :param add_sos: if true, add the <SOS> tag at the beginning of given sentence
        """
        index_sequence = [self.char2idx['SOS']] if add_sos else []

        for char in self.split_sequence(sequence):
            if char not in self.char2idx:
                index_sequence.append((self.char2idx['UNK']))
            else:
                index_sequence.append(self.char2idx[char])

        if add_eos:
            index_sequence.append(self.char2idx['EOS'])

        return index_sequence

    def indices_to_sequence(self, indices):
        """Transform a list of indices
            :param indices: a list
        """
        sequence = ""
        for idx in indices:
            char = self.idx2char[idx]
            if char == "EOS":
                break
            else:
                sequence += char
        return sequence

    def split_sequence(self, sequence):
        """Vary from languages and tasks. In our task, we simply return chars in given sentence
        For example:
            Input : alphabet
            Return: [a, l, p, h, a, b, e, t]
        """
        return [char for char in sequence]

    def __str__(self):
        str = "Vocab information:\n"
        for idx, char in self.idx2char.items():
            str += "Char: %s Index: %d\n" % (char, idx)
        return str


class DataTransformer(object):

    def __init__(self, path, use_cuda):
        self.indices_sequences = []
        self.use_cuda = use_cuda

        # Load and build the vocab
        self.vocab = Vocabulary()
        self.vocab.build_vocab(path)
        self.PAD_ID = self.vocab.char2idx["PAD"]
        self.SOS_ID = self.vocab.char2idx["SOS"]
        self.vocab_size = self.vocab.num_chars
        self.max_length = self.vocab.max_length

        self._build_training_set(path)

    def _build_training_set(self, path):
        # Change sentences to indices, and append <EOS> at the end of all pairs
        for word in self.vocab.word_list:
            indices_seq = self.vocab.sequence_to_indices(word, add_eos=True)
            # input and target are the same in auto-encoder
            self.indices_sequences.append([indices_seq, indices_seq[:]])

    def mini_batches(self, batch_size):
        input_batches = []
        target_batches = []

        np.random.shuffle(self.indices_sequences)
        mini_batches = [
            self.indices_sequences[k: k + batch_size]
            for k in range(0, len(self.indices_sequences), batch_size)
        ]

        for batch in mini_batches:
            seq_pairs = sorted(batch, key=lambda seqs: len(seqs[0]), reverse=True)  # sorted by input_lengths
            input_seqs = [pair[0] for pair in seq_pairs]
            target_seqs = [pair[1] for pair in seq_pairs]

            input_lengths = [len(s) for s in input_seqs]
            in_max = input_lengths[0]
            input_padded = [self.pad_sequence(s, in_max) for s in input_seqs]

            target_lengths = [len(s) for s in target_seqs]
            out_max = target_lengths[0]
            target_padded = [self.pad_sequence(s, out_max) for s in target_seqs]

            input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)  # time * batch
            target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)  # time * batch

            if self.use_cuda:
                input_var = input_var.cuda()
                target_var = target_var.cuda()

            yield (input_var, input_lengths), (target_var, target_lengths)

    def pad_sequence(self, sequence, max_length):
        sequence += [self.PAD_ID for i in range(max_length - len(sequence))]
        return sequence

    def evaluation_batch(self, words):
        """
        Prepare a batch of var for evaluating
        :param words: a list, store the testing data 
        :return: evaluation_batch
        """
        evaluation_batch = []

        for word in words:
            indices_seq = self.vocab.sequence_to_indices(word, add_eos=True)
            evaluation_batch.append([indices_seq])

        seq_pairs = sorted(evaluation_batch, key=lambda seqs: len(seqs[0]), reverse=True)
        input_seqs = [pair[0] for pair in seq_pairs]
        input_lengths = [len(s) for s in input_seqs]
        in_max = input_lengths[0]
        input_padded = [self.pad_sequence(s, in_max) for s in input_seqs]

        input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)  # time * batch

        if self.use_cuda:
            input_var = input_var.cuda()

        return input_var, input_lengths

if __name__ == '__main__':
    vocab = Vocabulary()
    vocab.build_vocab('Google-10000-English.txt')
    print(vocab)

    test = "helloworld"
    print("Sequence before transformed:", test)
    ids = vocab.sequence_to_indices(test)
    print("Indices sequence:", ids)
    sent = vocab.indices_to_sequence(ids)
    print("Sequence after transformed:",sent)

    data_transformer = DataTransformer('Google-10000-English.txt', use_cuda=False)

    for ib, tb in data_transformer.mini_batches(batch_size=3):
        print("B0-0")
        print(ib, tb)
        break
