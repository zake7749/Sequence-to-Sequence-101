# To generate the training set, counting.txt, for Epoch1.
import random
import string

# Config
data_size = 3000
flip_ratio = .5
max_len = 12
output_file_name = 'Alphabet.txt'
curr_set = set()

with open(output_file_name, 'w', encoding='utf-8') as dataset:
    cur_data_num = 0

    while cur_data_num < data_size:
        seq = []
        for char in string.ascii_uppercase[:max_len]:
            if random.uniform(0, 1) > flip_ratio:
                seq.append(char)
        seq = " ".join(seq)
        if len(seq) == 0 or seq in curr_set:
            continue
        else:
            dataset.write(seq + '\n')
            curr_set.add(seq)
            cur_data_num += 1