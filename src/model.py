import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np

torch.manual_seed(1)

def to_tensor(x):
    x = np.array(x)
    x = torch.from_numpy(x)
    return x.cuda()

class Classifier(nn.Module):

    def __init__(self, word_size, label_size, category_sizes,
                 word_dim, embed_dim, hidden_dim, category_dim,
                 inject_type, inject_locs, chunk_ratio=1,
                 basis=-1, drop_prob=0.1):
        super(Classifier, self).__init__()
        self.label_size = label_size
        self.category_sizes = category_sizes
        self.word_dim = word_dim
        self.embed_dim = embed_dim
        hidden_dim = hidden_dim//2
        self.hidden_dim = hidden_dim
        classify_dim = hidden_dim*2
        self.classify_dim = classify_dim
        self.category_dim = category_dim
        self.inject_type = inject_type
        self.inject_locs = inject_locs
        self.chunk_ratio = chunk_ratio
        self.basis = basis

        category_embeds = []
        for category_size in category_sizes:
            category_embeds.append(nn.Embedding(category_size, category_dim//len(category_sizes)))
        self.category_embeds = nn.ModuleList(category_embeds)

        self.embedding = nn.Embedding(word_size, word_dim, padding_idx=0)
        if 'embed' in inject_locs:
            if inject_type == 'bias':
                self.embed_transform = nn.Linear(category_dim, embed_dim)
                self.embed_weight = nn.Parameter(torch.Tensor(word_dim, embed_dim))
            if 'weight' in inject_type:
                if 'chunk' in inject_type:
                    self.embed_transform = nn.Linear(category_dim, word_dim*embed_dim//(chunk_ratio**2))
                else:
                    self.embed_transform = nn.Linear(category_dim, word_dim*embed_dim)
                if 'imp' in inject_type:
                    self.embed_weight = nn.Parameter(torch.Tensor(word_dim, embed_dim))
                self.embed_bias = nn.Parameter(torch.Tensor(embed_dim))
        else:
            self.embed_transform = nn.Linear(word_dim, embed_dim)

        if 'encode' in inject_locs:
            if inject_type == 'bias':
                self.encode_f_transform = nn.Linear(category_dim, 4*hidden_dim)
                self.encode_f_weight = nn.Parameter(torch.Tensor(embed_dim+hidden_dim, 4*hidden_dim))
                self.encode_b_transform = nn.Linear(category_dim, 4*hidden_dim)
                self.encode_b_weight = nn.Parameter(torch.Tensor(embed_dim+hidden_dim, 4*hidden_dim))
            if 'weight' in inject_type:
                if 'chunk' in inject_type:
                    self.encode_f_transform = nn.Linear(category_dim, (embed_dim+hidden_dim)*4*hidden_dim//(chunk_ratio**2))
                    self.encode_b_transform = nn.Linear(category_dim, (embed_dim+hidden_dim)*4*hidden_dim//(chunk_ratio**2))
                else:
                    self.encode_f_transform = nn.Linear(category_dim, (embed_dim+hidden_dim)*4*hidden_dim)
                    self.encode_b_transform = nn.Linear(category_dim, (embed_dim+hidden_dim)*4*hidden_dim)
                if 'imp' in inject_type:
                    self.encode_f_weight = nn.Parameter(torch.Tensor(embed_dim+hidden_dim, 4*hidden_dim))
                    self.encode_b_weight = nn.Parameter(torch.Tensor(embed_dim+hidden_dim, 4*hidden_dim))
                self.encode_f_bias = nn.Parameter(torch.Tensor(4*hidden_dim))
                self.encode_b_bias = nn.Parameter(torch.Tensor(4*hidden_dim))
        else:
            self.encode_f_transform = nn.Linear(embed_dim+hidden_dim, 4*hidden_dim)
            self.encode_b_transform = nn.Linear(embed_dim+hidden_dim, 4*hidden_dim)

        self.pool_latent = nn.Linear(hidden_dim, 1)
        if 'pool' in inject_locs:
            if inject_type == 'bias':
                self.pool_transform = nn.Linear(category_dim, hidden_dim)
                self.pool_weight = nn.Parameter(torch.Tensor(2*hidden_dim, hidden_dim))
            if 'weight' in inject_type:
                if 'chunk' in inject_type:
                    self.pool_transform = nn.Linear(category_dim, 2*hidden_dim*hidden_dim//(chunk_ratio**2))
                else:
                    self.pool_transform = nn.Linear(category_dim, 2*hidden_dim*hidden_dim)
                if 'imp' in inject_type:
                    self.pool_weight = nn.Parameter(torch.Tensor(2*hidden_dim, hidden_dim))
                self.pool_bias = nn.Parameter(torch.Tensor(hidden_dim))
        else:
            self.pool_transform = nn.Linear(2*hidden_dim, hidden_dim)

        if 'classify' in inject_locs:
            if inject_type == 'bias':
                self.classify_transform = nn.Linear(category_dim, label_size)
                self.classify_weight = nn.Parameter(torch.Tensor(classify_dim, label_size))
            if 'weight' in inject_type:
                if 'chunk' in inject_type:
                    self.classify_transform = nn.Linear(category_dim, classify_dim*label_size//chunk_ratio)
                else:
                    self.classify_transform = nn.Linear(category_dim, classify_dim*label_size)
                if 'imp' in inject_type:
                    self.classify_weight = nn.Parameter(torch.Tensor(classify_dim, label_size))
                self.classify_bias = nn.Parameter(torch.Tensor(label_size))
        else:
            self.classify_transform = nn.Linear(classify_dim, label_size)

        self.dropout = nn.Dropout(drop_prob)

        for _, p in self.named_parameters():
            if len(p.size()) == 1:
                nn.init.constant_(p, 0)
            else:
                nn.init.kaiming_normal_(p)

    def forward(self, input, mask, categories):
        batch_size, sequence_len = input.size()

        if self.inject_type != 'none':
            cs = []
            for i, category_embed in enumerate(self.category_embeds):
                cs.append(category_embed(categories[:,i]))
            cs = torch.cat(cs, dim=-1)
            cs = self.dropout(cs)

        ws = self.embedding(input)
        ws = self.dropout(ws)
        if 'embed' in self.inject_locs:
            if self.inject_type == 'bias':
                embed_bias = self.embed_transform(cs).view(batch_size, 1, self.embed_dim)
                ws = torch.tanh(torch.matmul(ws, self.embed_weight) + embed_bias)
            if 'weight' in self.inject_type:
                if 'chunk' in self.inject_type:
                    embed_weight = self.embed_transform(cs).view(batch_size, self.word_dim//self.chunk_ratio, self.embed_dim//self.chunk_ratio)
                    embed_weight = embed_weight.repeat(1, self.chunk_ratio, self.chunk_ratio)
                else:
                    embed_weight = self.embed_transform(cs).view(batch_size, self.word_dim, self.embed_dim)
                if 'imp' in self.inject_type:
                    embed_weight = torch.sigmoid(embed_weight) * self.embed_weight
                ws = torch.tanh(torch.matmul(ws, embed_weight) + self.embed_bias)
        else:
            ws = torch.tanh(self.embed_transform(ws))
        ws = self.dropout(ws)

        hl = cl = ws.new_zeros(batch_size, self.hidden_dim)
        hls = []
        if 'encode' in self.inject_locs:
            if self.inject_type == 'bias':
                encode_f_bias = self.encode_f_transform(cs).view(batch_size, 4*self.hidden_dim)
            if 'weight' in self.inject_type:
                if 'chunk' in self.inject_type:
                    encode_f_weight = self.encode_f_transform(cs).view(batch_size, (self.embed_dim+self.hidden_dim)//self.chunk_ratio, 4*self.hidden_dim//self.chunk_ratio)
                    encode_f_weight = encode_f_weight.repeat(1, self.chunk_ratio, self.chunk_ratio)
                else:
                    encode_f_weight = self.encode_f_transform(cs).view(batch_size, self.embed_dim+self.hidden_dim, 4*self.hidden_dim)
                if 'imp' in self.inject_type:
                    encode_f_weight = torch.sigmoid(encode_f_weight) * self.encode_f_weight
        for t in range(sequence_len):
            wt = ws[:,t]
            wht = torch.cat([wt,hl], dim=-1)
            if 'encode' in self.inject_locs:
                if self.inject_type == 'bias':
                    ft, it, ot, ut = (torch.matmul(wht, self.encode_f_weight) + encode_f_bias).chunk(4, -1)
                if 'weight' in self.inject_type:
                    wht = wht.view(batch_size, 1, -1)
                    ft, it, ot, ut = (torch.matmul(wht, encode_f_weight) + self.encode_f_bias).squeeze(1).chunk(4, -1)
            else:
                ft, it, ot, ut = self.encode_f_transform(wht).chunk(4, -1)
            hl, cl = self.lstm_cell(ft, it, ot, ut, cl)
            hls = hls + [hl]
        hls = torch.stack(hls, dim=1)

        hr = cr = ws.new_zeros(batch_size, self.hidden_dim)
        hrs = []
        if 'encode' in self.inject_locs:
            if self.inject_type == 'bias':
                encode_b_bias = self.encode_b_transform(cs).view(batch_size, 4*self.hidden_dim)
            if 'weight' in self.inject_type:
                if 'chunk' in self.inject_type:
                    encode_b_weight = self.encode_b_transform(cs).view(batch_size, (self.embed_dim+self.hidden_dim)//self.chunk_ratio, 4*self.hidden_dim//self.chunk_ratio)
                    encode_b_weight = encode_b_weight.repeat(1, self.chunk_ratio, self.chunk_ratio)
                else:
                    encode_b_weight = self.encode_b_transform(cs).view(batch_size, self.embed_dim+self.hidden_dim, 4*self.hidden_dim)
                if 'imp' in self.inject_type:
                    encode_b_weight = torch.sigmoid(encode_b_weight) * self.encode_b_weight
        for t in reversed(range(sequence_len)):
            wt = ws[:,t]
            wht = torch.cat([wt,hr], dim=-1)
            if 'encode' in self.inject_locs:
                if self.inject_type == 'bias':
                    ft, it, ot, ut = (torch.matmul(wht, self.encode_b_weight) + encode_b_bias).chunk(4, -1)
                if 'weight' in self.inject_type:
                    wht = wht.view(batch_size, 1, -1)
                    ft, it, ot, ut = (torch.matmul(wht, encode_b_weight) + self.encode_b_bias).squeeze(1).chunk(4, -1)
            else:
                ft, it, ot, ut = self.encode_b_transform(wht).chunk(4, -1)
            hr, cr = self.lstm_cell(ft, it, ot, ut, cr)
            hrs = [hr] + hrs
        hrs = torch.stack(hrs, dim=1)
        hs = torch.cat([hls, hrs], dim=-1)

        if 'pool' in self.inject_locs:
            if self.inject_type == 'bias':
                pool_bias = self.pool_transform(cs).view(batch_size, 1, self.hidden_dim)
                e = torch.tanh(torch.matmul(hs, self.pool_weight) + pool_bias)
            if 'weight' in self.inject_type:
                if 'chunk' in self.inject_type:
                    pool_weight = self.pool_transform(cs).view(batch_size, 2*self.hidden_dim//self.chunk_ratio, self.hidden_dim//self.chunk_ratio)
                    pool_weight = pool_weight.repeat(1, self.chunk_ratio, self.chunk_ratio)
                else:
                    pool_weight = self.pool_transform(cs).view(batch_size, 2*self.hidden_dim, self.hidden_dim)
                if 'imp' in self.inject_type:
                    pool_weight = torch.sigmoid(pool_weight) * self.pool_weight
                e = torch.tanh(torch.matmul(hs, pool_weight) + self.pool_bias)
        else:
            e = torch.tanh(self.pool_transform(hs))
        e = self.dropout(e)
        a = F.softmax(self.pool_latent(e), dim=1) * mask.unsqueeze(-1)
        a = a / a.sum(dim=1, keepdim=True)
        h = torch.sum(a*hs, dim=1)
        h = self.dropout(h)

        if 'classify' in self.inject_locs:
            if self.inject_type == 'bias':
                classify_bias = self.classify_transform(cs).view(batch_size, self.label_size)
                p = torch.matmul(h, self.classify_weight) + classify_bias
            if 'weight' in self.inject_type:
                if 'chunk' in self.inject_type:
                    classify_weight = self.classify_transform(cs).view(batch_size, self.classify_dim//self.chunk_ratio, self.label_size)
                    classify_weight = classify_weight.repeat(1, self.chunk_ratio, 1)
                else:
                    classify_weight = self.classify_transform(cs).view(batch_size, self.classify_dim, self.label_size)
                if 'imp' in self.inject_type:
                    classify_weight = torch.sigmoid(classify_weight) * self.classify_weight
                h = h.view(batch_size, 1, self.classify_dim)
                p = torch.matmul(h, classify_weight).squeeze(1) + self.classify_bias
        else:
            p = self.classify_transform(h)

        return p

    def lstm_cell(self, ft, it, ot, ut, cx):
        fx = torch.sigmoid(ft)
        ix = torch.sigmoid(it)
        ox = torch.sigmoid(ot)
        ux = torch.tanh(ut)

        cx = fx * cx + ix * ux
        hx = ox * torch.tanh(cx)

        return hx, cx