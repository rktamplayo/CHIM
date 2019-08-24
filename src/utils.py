import numpy as np
import fastText as fasttext
from tqdm import tqdm

def get_vectors(train_file, num_categories, vec_file, word_dim, with_label=True, count=5):
    label_dict = {}
    category_dicts = [{'<unk>': 0} for _ in range(num_categories)]
    word_count = {}

    f = open(train_file, 'r', encoding='utf-8', errors='ignore')
    for line in f:
        spl = line.strip().lower().split('\t\t')
        
        # get categories
        for i in range(num_categories):
            categories = spl[i].split(',')
            for category in categories:
                if category not in category_dicts[i]:
                    category_dicts[i][category] = len(category_dicts[i])

        # get labels
        if with_label:
            labels = spl[num_categories+1].split(" ")
            for label in labels:
                if label not in label_dict:
                    label_dict[label] = len(label_dict)

        # get texts
        text = spl[num_categories].split()
        for token in text:
            if token not in word_count:
                word_count[token] = 0
            word_count[token] += 1

    word_dict = {}
    word_dict['<pad>'] = len(word_dict)
    word_dict['<unk>'] = len(word_dict)

    for word in word_count:
        if word_count[word] >= count:
            if word not in word_dict:
                word_dict[word] = len(word_dict)

    print('Number of Words:', len(word_dict))

    word_vectors = np.random.uniform(-0.1, 0.1, (len(word_dict), word_dim))
    
    f = open(vec_file, 'r', encoding='utf-8', errors='ignore')
    #f.readline()
    for line in tqdm(f):
        spl = line.split()
        token = spl[0]
        vector = [float(x) for x in spl[-word_dim:]]
        if token in word_dict:
            word_vectors[word_dict[token]] = np.array(vector)
    f.close()

    return word_vectors, word_dict, category_dicts, label_dict

def get_data(file, word_dict, category_dicts, label_dict, with_label=True):
    data = {}
    data['x'] = []
    data['c'] = []
    data['y'] = []
    num_categories = len(category_dicts)

    f = open(file, 'r', encoding='utf-8', errors='ignore')
    for line in tqdm(f):
        spl = line.strip().lower().split('\t\t')

        # get labels
        if with_label:
            label = spl[num_categories+1]
            if label not in label_dict:
                continue
            y = label_dict[label]
            data['y'].append(y)

        # get categories
        c = []
        for i in range(num_categories):
            category = spl[i]
            if category in category_dicts[i]:
                c.append(category_dicts[i][category])
            else:
                c.append(category_dicts[i]['<unk>'])
        data['c'].append(c)

        # get texts
        text = spl[num_categories].split()
        x = []
        for token in text:
            if token in word_dict:
                x.append(word_dict[token])
            else:
                x.append(word_dict['<unk>'])
        data['x'].append(x)

    f.close()

    return data


def pad(batch, pad_id=0, left_pad=True):
    max_length = max(len(x) for x in batch)
    new_batch = []
    mask_batch = []
    for x in batch:
        x = x + [pad_id] * (max_length - len(x))
        mask = [1.0] * len(x) + [0.0] * (max_length - len(x))
        new_batch.append(x[:max_length])
        mask_batch.append(mask[:max_length])
    new_batch = np.array(new_batch)
    mask_batch = np.array(mask_batch)
    return new_batch, mask_batch