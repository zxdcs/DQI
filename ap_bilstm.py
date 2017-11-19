import time
import torch
import numpy
from torch import nn
from torch.nn import functional as func
from torch.autograd import Variable
from torch.nn import init
from torch.nn.utils import rnn
import pickle
from sklearn.metrics import accuracy_score
from utils import Dataset, get_script_short_name

hparams = {
    'learning_rate': 0.001,
    'max_epoch': 10,
    'display_step': 1000,
    'emb_dim': 300,
    'conv_win': 3,
    'repr_dim': 150,
    'fc1_dim': 300,
    'n_classes': 2,
    'batch_size': 100
}


class LSTM(nn.Module):
    def __init__(self, emb_layer):
        super(LSTM, self).__init__()
        self.emb_layer = emb_layer
        self.lstm_layer = nn.LSTM(input_size=hparams['emb_dim'], hidden_size=int(hparams['repr_dim'] / 2),
                                  batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        embs = self.emb_layer(x)
        embs_sort, lengths_sort, unsort_idx = self.sort_batch(embs, lengths)
        seq = rnn.pack_padded_sequence(embs_sort, lengths_sort.cpu().numpy(), batch_first=True)
        hs, _ = self.lstm_layer(seq)
        hs, _ = rnn.pad_packed_sequence(hs, batch_first=True)
        return hs[unsort_idx]

    @staticmethod
    def sort_batch(x, l):
        l = torch.from_numpy(numpy.asarray(l)).cuda()
        l_sorted, sidx = l.sort(0, descending=True)
        x_sorted = x[sidx]
        _, unsort_idx = sidx.sort()
        return x_sorted, l_sorted, unsort_idx


class Model(nn.Module):
    def __init__(self, emb_layer):
        super(Model, self).__init__()
        self.lstm = LSTM(emb_layer)
        self.u = nn.Parameter(init.xavier_uniform(torch.FloatTensor(hparams['repr_dim'], hparams['repr_dim'])))
        self.fc1 = nn.Linear(hparams['repr_dim'] * 2, hparams['fc1_dim'])
        self.fc2 = nn.Linear(hparams['fc1_dim'], hparams['n_classes'])

    def forward(self, q1, q2, q1_len, q2_len):
        r1 = self.lstm(q1, q1_len)
        r2 = self.lstm(q2, q2_len)
        h1s, h2s = [], []
        for i in range(len(q1_len)):
            r1i = r1[i, :q1_len[i]]
            r2i = r2[i, :q2_len[i]]
            g = torch.tanh(r1i.mm(self.u).mm(r2i.t()))
            g1, _ = g.max(1)
            g2, _ = g.max(0)
            g1 = func.softmax(g1.t()).t().expand_as(r1i)
            g2 = func.softmax(g2).t().expand_as(r2i)
            h1 = (r1i * g1).sum(0).squeeze()
            h2 = (r2i * g2).sum(0).squeeze()
            h1s.append(h1)
            h2s.append(h2)
        r1 = torch.stack(h1s)
        r2 = torch.stack(h2s)
        joint = torch.cat(((r1 - r2).abs(), r1 * r2), dim=1)
        joint = torch.tanh(self.fc1(joint))
        out = self.fc2(joint)
        return out


def run_batch(b_data, b_lengths, model, optimizer=None):
    q1 = Variable(torch.from_numpy(b_data['q1']).cuda())
    q2 = Variable(torch.from_numpy(b_data['q2']).cuda())
    outputs = model(q1, q2, b_lengths['q1'], b_lengths['q2'])
    if optimizer:
        y = Variable(torch.from_numpy(b_data['y']).cuda())
        optimizer.zero_grad()
        loss = func.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()
        return loss.data[0]
    else:
        _, predicted = outputs.data.max(1)
        prob = func.softmax(outputs).data
        return predicted, prob[:, 1]


def run_epoch_eval(dataset, model, output_file=''):
    all_plabels, all_pscores = [], []
    batches, batch_lengths = dataset.get_batches(hparams['batch_size'], ('q1', 'q2', 'y'))
    for b_data, b_lengths in zip(batches, batch_lengths):
        plabels, pscores = run_batch(b_data, b_lengths, model)
        all_plabels.extend(plabels.cpu().numpy().tolist())
        all_pscores.extend(pscores.cpu().numpy().tolist())
    if output_file:
        with open(output_file, 'w') as f:
            for s in all_pscores:
                f.write(f'{s:.6f}\n')
    return accuracy_score(dataset.get_data_item('y'), all_plabels)


def run():
    print('Loading data...')
    with open('data/data_emb', 'rb') as f:
        all_sets, embs, word2idx = pickle.load(f)
    emb_layer = nn.Embedding(embs.shape[0], embs.shape[1])
    emb_layer.weight = nn.Parameter(torch.from_numpy(embs))
    model = Model(emb_layer).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    train_set = Dataset(all_sets[0], shuffle=True, pad_keys=('q1', 'q2'))
    dev_set = Dataset(all_sets[1], shuffle=False, pad_keys=('q1', 'q2'))
    test_set = Dataset(all_sets[2], shuffle=False, pad_keys=('q1', 'q2'))
    step = 0
    sum_loss = 0
    dev_best = 0
    test_score = 0
    print("Starting training...")
    print(hparams)
    start_time = time.time()
    for epoch in range(hparams['max_epoch']):
        batches, batch_lengths = train_set.get_batches(hparams['batch_size'], ('q1', 'q2', 'y'))
        for b_data, b_lengths in zip(batches, batch_lengths):
            sum_loss += run_batch(b_data, b_lengths, model, optimizer)
            step += 1
            if step % hparams['display_step'] == 0:
                avg_loss = sum_loss / hparams['display_step']
                sum_loss = 0
                dev_score = run_epoch_eval(dev_set, model)
                out_str = f'Epoc {epoch} iter {step} took {time.time() - start_time:.1f}s\n' \
                          f'loss:\t{avg_loss:.5f}\tdev score:\t{dev_score:.4f}'
                if dev_score > dev_best:
                    dev_best = dev_score
                    output_file = f'pred/{get_script_short_name(__file__)}.pred'
                    test_score = run_epoch_eval(test_set, model, output_file)
                    out_str += f'\t*** New best dev ***\ttest score:\t{test_score:.4f}'
                print(out_str)
                start_time = time.time()
    print('Best model on dev: dev:{:.4f}\ttest:{:.4f}'.format(dev_best, test_score))


if __name__ == '__main__':
    run()
