import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle
from tqdm import tqdm
import sys
import os

dim = int(sys.argv[5])
word_dim = 300
embed_dim = dim
hidden_dim = dim
category_dim = dim

num_epoch = 300
batch_size = 32
evaluate_every = 10000
training_stopper = 30
best_dev_acc = 0

num_categories = 2
basis = 0

data_type = sys.argv[1]
lang = sys.argv[2]
inject_type = sys.argv[3] # none, bias, weight, weight.imp, etc.
if 'chunk' in inject_type:
    basis = int(sys.argv[6])
inject_locs = sys.argv[4].split(',') # embed, encode, pool, classify
device = sys.argv[-1].split(':')[-1]
os.environ['CUDA_VISIBLE_DEVICES'] = device

file_dir = 'data/' + data_type + '/'
test_file = file_dir + '/test.txt'
vec_file = 'data/glove.txt'

pickle_ext = file_dir + '/data'
vector_pickle = pickle_ext + '.vec.p'
test_pickle = pickle_ext + '.test.p'
model_file = file_dir + '/model.' + inject_type + '.' + '_'.join(inject_locs) + '.' + str(basis) + '.' + str(dim)

if 'none' in inject_locs:
    inject_locs = []

from model import Classifier as Model
import utils

def to_tensor(x, device=device):
    x = np.array(x)
    x = torch.from_numpy(x)
    return x.cuda()

if os.path.isfile(vector_pickle):
    word_vectors, word_dict, category_dicts, label_dict = pickle.load(open(vector_pickle, 'rb'))
else:
    word_vectors, word_dict, category_dicts, label_dict = utils.get_vectors(train_file, num_categories, vec_file, word_dim)
    pickle.dump([word_vectors, word_dict, category_dicts, label_dict], open(vector_pickle, 'wb'), protocol=4)

if os.path.isfile(test_pickle):
    test_data = pickle.load(open(test_pickle, 'rb'))
else:
    test_data = utils.get_data(test_file, word_dict, category_dicts, label_dict)
    pickle.dump(test_data, open(test_pickle, 'wb'), protocol=4)

category_sizes = [len(category_dict) for category_dict in category_dicts]
label_size = len(label_dict)
word_size = len(word_dict)

best_point = torch.load(model_file)
model = Model(word_size, label_size, category_sizes,
              word_dim, embed_dim, hidden_dim, category_dim,
              inject_type, inject_locs, basis)
model.cuda()
model.load_state_dict(best_point['state_dict'])

x_test = test_data['x']
c_test = test_data['c']
y_test = test_data['y']

test_acc = 0
test_rmse = 0
rev_label_dict = {label_dict[label]:label for label in label_dict}
with torch.no_grad():
    for j in tqdm(range(0, len(x_test), batch_size)):
        model.eval()

        x_batch, m_batch = utils.pad(x_test[j:j+batch_size])
        x_batch = to_tensor(x_batch)
        m_batch = to_tensor(m_batch).float()
        c_batch = to_tensor(c_test[j:j+batch_size])
        y_batch = to_tensor(y_test[j:j+batch_size])

        p_batch = model(x_batch, m_batch, c_batch).max(1)[1]
        test_acc += p_batch.eq(y_batch).long().sum().item()

        p_batch = p_batch.cpu().detach().numpy()
        p_batch = np.array([int(rev_label_dict[p]) for p in p_batch])
        y_batch = y_batch.cpu().detach().numpy()
        y_batch = np.array([int(rev_label_dict[y]) for y in y_batch])
        test_rmse += np.sum((p_batch-y_batch) ** 2)
test_acc /= len(x_test)
test_rmse = np.sqrt(test_rmse/len(x_test))

print(test_acc, test_rmse)