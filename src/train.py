import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle
from tqdm import tqdm
import sys
import os

dim = int(sys.argv[4])
word_dim = 300
embed_dim = dim
hidden_dim = dim
category_dim = dim

num_epoch = 300
batch_size = 32
evaluate_every = 10000
training_stopper = 30
best_dev_acc = 0
best_rmse = 100

num_categories = 2
basis = 0

data_type = sys.argv[1]
inject_type = sys.argv[2] # none, bias, weight, weight.imp, etc.
if 'chunk' in inject_type:
    basis = int(sys.argv[5])
inject_locs = sys.argv[3].split(',') # embed, encode, pool, classify
device = sys.argv[-1].split(':')[-1]
os.environ['CUDA_VISIBLE_DEVICES'] = device

file_dir = 'data/' + data_type + '/'
train_file = file_dir + '/train.txt'
dev_file = file_dir + '/dev.txt'
vec_file = 'data/glove.txt'

pickle_ext = file_dir + '/data'
vector_pickle = pickle_ext + '.vec.p'
train_pickle = pickle_ext + '.train.p'
dev_pickle = pickle_ext + '.dev.p'
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

if os.path.isfile(train_pickle):
    train_data = pickle.load(open(train_pickle, 'rb'))
else:
    train_data = utils.get_data(train_file, word_dict, category_dicts, label_dict)
    pickle.dump(train_data, open(train_pickle, 'wb'), protocol=4)

if os.path.isfile(dev_pickle):
    dev_data = pickle.load(open(dev_pickle, 'rb'))
else:
    dev_data = utils.get_data(dev_file, word_dict, category_dicts, label_dict)
    pickle.dump(dev_data, open(dev_pickle, 'wb'), protocol=4)

category_sizes = [len(category_dict) for category_dict in category_dicts]
label_size = len(label_dict)
word_size = len(word_dict)

model = Model(word_size, label_size, category_sizes,
              word_dim, embed_dim, hidden_dim, category_dim,
              inject_type, inject_locs, chunk_ratio=basis, basis=basis)
with torch.no_grad():
    model.embedding.weight.set_(torch.from_numpy(word_vectors).float())
model.cuda()
optimizer = torch.optim.Adadelta(model.parameters())
#if os.path.exists(model_file):
#    best_point = torch.load(model_file)
#    model.load_state_dict(best_point['state_dict'])
#    optimizer.load_state_dict(best_point['optimizer'])
#    best_dev_acc = best_point['dev_acc']
print("Total Parameters:", sum(p.numel() for p in model.parameters()))

x_train = train_data['x']
c_train = train_data['c']
y_train = train_data['y']
x_dev = dev_data['x']
c_dev = dev_data['c']
y_dev = dev_data['y']

evaluate_every = len(x_train) // 10

eval_at = evaluate_every
stop_at = training_stopper

rev_label_dict = {label_dict[label]:label for label in label_dict}
for epoch in range(num_epoch):
    if stop_at <= 0:
        break

    shuffle_indices = np.random.permutation(np.arange(len(x_train)))
    x_shuffle = np.array(x_train)[shuffle_indices]
    c_shuffle = np.array(c_train)[shuffle_indices]
    y_shuffle = np.array(y_train)[shuffle_indices]

    losses = []
    for i in tqdm(range(0, len(x_train), batch_size)):
        model.train()

        x_batch, m_batch = utils.pad(x_shuffle[i:i+batch_size])
        x_batch = to_tensor(x_batch)
        m_batch = to_tensor(m_batch).float()
        c_batch = to_tensor(c_shuffle[i:i+batch_size])
        y_batch = to_tensor(y_shuffle[i:i+batch_size])

        p_batch = model(x_batch, m_batch, c_batch)

        batch_loss = F.cross_entropy(p_batch, y_batch)
        losses.append(batch_loss.item())

        optimizer.zero_grad()
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()

        eval_at -= len(x_batch)
        if eval_at <= 0:
            avg_loss = np.mean(losses)
            dev_acc = 0
            dev_rmse = 0
            with torch.no_grad():
                for j in tqdm(range(0, len(x_dev), batch_size)):
                    model.eval()

                    x_batch, m_batch = utils.pad(x_dev[j:j+batch_size])
                    x_batch = to_tensor(x_batch)
                    m_batch = to_tensor(m_batch).float()
                    c_batch = to_tensor(c_dev[j:j+batch_size])
                    y_batch = to_tensor(y_dev[j:j+batch_size])

                    p_batch = model(x_batch, m_batch, c_batch).max(1)[1]
                    dev_acc += p_batch.eq(y_batch).long().sum().item()

                    p_batch = p_batch.cpu().detach().numpy()
                    p_batch = np.array([int(rev_label_dict[p]) for p in p_batch])
                    y_batch = y_batch.cpu().detach().numpy()
                    y_batch = np.array([int(rev_label_dict[y]) for y in y_batch])
                    dev_rmse += np.sum((p_batch-y_batch) ** 2)
            dev_acc /= len(x_dev)
            dev_rmse = np.sqrt(dev_rmse/len(x_dev))

            if best_dev_acc <= dev_acc:
                best_dev_acc = dev_acc
                best_rmse = dev_rmse
                stop_at = training_stopper
                torch.save({
                    'state_dict': model.state_dict(),
                    'dev_acc': dev_acc,
                    'optimizer': optimizer.state_dict()
                }, model_file)
            else:
                stop_at -= 1

            print("Epoch: %d, Batch: %d, Loss: %.4f, Dev Acc: %.2f, RMSE: %.3f" % (epoch, i, avg_loss, dev_acc*100, dev_rmse))
            losses = []
            eval_at = evaluate_every

print(best_dev_acc, best_rmse)
