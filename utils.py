import numpy
import os


def pad(var_arr, max_len=None, fix_len=False):
    if max_len is None:
        seq_len = [len(x) for x in var_arr]
        max_len = max(seq_len)
    else:
        assert max_len > 0
        seq_len = [min(len(x), max_len) for x in var_arr]
        if not fix_len:
            max_len = max(seq_len)
    fixed_var = numpy.zeros((len(var_arr), max_len), dtype=numpy.int64)
    for idx, x in enumerate(var_arr):
        fixed_var[idx][0:seq_len[idx]] = x[0:seq_len[idx]]
    for i in range(len(seq_len)):
        if seq_len[i] < 1:
            seq_len[i] = 1
    return fixed_var, seq_len


def pad2d(var_arr):
    max_len1 = max_len2 = 0
    for arrs in var_arr:
        max_len1 = max(len(arrs), max_len1)
        for arr in arrs:
            max_len2 = max(len(arr), max_len2)
    fixed_arr = numpy.zeros((len(var_arr), max_len1, max_len2), dtype=numpy.int64)
    for i, arrs in enumerate(var_arr):
        for j, arr in enumerate(arrs):
            for k, val in enumerate(arr):
                fixed_arr[i, j, k] = val
    return fixed_arr


def pad_vec(var_arr):  # Sparse matrix
    vec_len = var_arr[0].shape[-1]
    assert vec_len != 0
    seq_len = [x.shape[0] for x in var_arr]
    max_len = max(seq_len)
    fixed_var = numpy.zeros((len(var_arr), max_len, vec_len), dtype=numpy.float32)
    for i, x in enumerate(var_arr):
        for j, k in zip(*x.nonzero()):
            fixed_var[i, j, k] = x[j, k]
    for i in range(len(seq_len)):
        if seq_len[i] < 1:
            seq_len[i] = 1
    return fixed_var, seq_len


class Dataset:
    def __init__(self, data, shuffle=False, pad_keys=(), pad2d_keys=()):
        self._data = {key: numpy.array(val) for key, val in data.items()}
        self._epochs_completed = 0
        self._num_examples = next(iter(self._data.values())).shape[0]
        self._index = numpy.arange(self._num_examples)
        self._shuffle = shuffle
        self._batches = []
        self._pad_keys = pad_keys
        self._seq_lens = []
        self._pad2d_keys = pad2d_keys

    def epochs_completed(self):
        return self._epochs_completed

    def get_batches(self, batch_size, keys):
        if self._shuffle or not self._batches:
            if self._shuffle:
                numpy.random.shuffle(self._index)
            self._build_batches(batch_size, keys)
        self._epochs_completed += 1
        return self._batches, self._seq_lens

    def _build_batches(self, batch_size, keys):
        self._batches = []
        self._seq_lens = []
        for start in range(0, self._num_examples, batch_size):
            batch = {}
            batch_seq_lens = {}
            for key in keys:
                batch_val = self._data[key][self._index[start:start + batch_size]]
                if key in self._pad_keys:
                    batch_val, seq_lens_val = pad(batch_val)
                    batch_seq_lens[key] = seq_lens_val
                elif key in self._pad2d_keys:
                    batch_val = pad2d(batch_val)
                batch[key] = batch_val
            self._batches.append(batch)
            self._seq_lens.append(batch_seq_lens)

    def get_singles(self, keys):
        if self._shuffle or not self._batches:
            if self._shuffle:
                numpy.random.shuffle(self._index)
            self._build_batches(1, keys)
        for single in self._batches:
            for key, val in single.items():
                single[key] = val[0]
        self._epochs_completed += 1
        return self._batches

    def get_data_item(self, key):
        return self._data[key]


def mask(lengths):
    lengths = numpy.array(lengths)
    lengths = lengths.reshape(lengths.shape[0], -1)
    max_lens = lengths.max(axis=0)
    batch_size = lengths.shape[0]
    mask_size = [batch_size] + max_lens.tolist()
    masks = numpy.zeros(mask_size, dtype=numpy.float32)
    num_dim = len(max_lens)
    assert num_dim < 3
    for sample, length in zip(masks, lengths):
        if num_dim == 1:
            sample[:length[0]] = 1
        elif num_dim == 2:
            sample[:length[0], :length[1]] = 1
    return masks


def copy_file(ori_file, dest_file):
    with open(ori_file) as fin, open(dest_file, 'w') as fout:
        for line in fin:
            fout.write(line)


def get_script_short_name(name):
    name = os.path.basename(name)
    name = name[:name.find('.')]
    return name


class Accuracy:
    def __init__(self):
        self.total = 0
        self.count = 0

    def update_batch(self, truth, pred):
        assert len(truth) == len(pred)
        for t, p in zip(truth, pred):
            if t == p:
                self.count += 1
        self.total += len(truth)

    def get_score(self):
        return self.count / self.total
