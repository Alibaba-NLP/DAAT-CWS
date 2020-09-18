import os
import numpy as np
import pickle
from itertools import izip
import tensorflow as tf

def fresh_dir(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
    tf.gfile.MakeDirs(path) 

def create_dic(item_list, add_unk=False, add_pad=False):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) in (list, tuple)
    dic = {}
    for items in item_list:
        for item in items:
            if item not in dic:
                dic[item] = 1
            else:
                dic[item] += 1
    # Make sure that <PAD> have a id 0.
    if add_pad:
        dic['<PAD>'] = 1e20
    # If specified, add a special item <UNK>.
    if add_unk:
        dic['<UNK>'] = 1e10
    return dic


def create_mapping(items):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    if type(items) is dict:
        sorted_items = sorted(items.items(), key=lambda x: (-x[1], x[0]))
        id2item = {i: v[0] for i, v in enumerate(sorted_items)}
        item2id = {v: k for k, v in id2item.items()}
        return item2id, id2item
    elif type(items) is list:
        id2item = {i: v for i, v in enumerate(items)}
        item2id = {v: k for k, v in id2item.items()}
        return item2id, id2item


def create_input(batch):
    """
    Take each sentence data in batch and return an input for
    the training or the evaluation function.
    """
    assert len(batch) > 0
    lengths = [len(seq) for seq in batch[0]]
    max_len = max(2, max(lengths))
    ret = []
    for d in batch:
        dd = []
        for seq_id, pos in izip(d, lengths):
            assert len(seq_id) == pos
            pad = [0] * (max_len - pos)
            dd.append(seq_id + pad)
        ret.append(np.array(dd))
    ret.append(np.array(lengths))
    return ret


def data_to_ids(data, mappings):
    """
    Map text data to ids.
    """

    def strQ2B(ustring):
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 12288:
                inside_code = 32
            elif 65281 <= inside_code <= 65374:
                inside_code -= 65248
            rstring += unichr(inside_code)
        return rstring

    def strB2Q(ustring):
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 32:
                inside_code = 12288
            elif 32 <= inside_code <= 126:
                inside_code += 65248
            rstring += unichr(inside_code)
        return rstring

    def map(item, mapping):
        if item in mapping:
            return mapping[item]
        item = strB2Q(item)
        if item in mapping:
            return mapping[item]
        item = strQ2B(item)
        if item in mapping:
            return mapping[item]
        return mapping['<UNK>']

    def map_seq(seqs, mapping):
        return [[map(item, mapping) for item in seq] for seq in seqs]

    ret = []
    for d, m in izip(data, mappings):
        ret.append(map_seq(d, m))
    return tuple(ret)


def data_iterator(inputs, batch_size, shuffle=True, max_length=200):
    """
    A simple iterator for generating dynamic mini batches.
    """
    assert len(inputs) > 0
    assert all([len(item) == len(inputs[0]) for item in inputs])
    inputs = zip(*inputs)
    if shuffle:
        np.random.shuffle(inputs)

    batch = []
    bs = batch_size
    for d in inputs:
        if len(d[0]) > max_length:
            bs = max(1, min(batch_size * max_length / len(d[0]), bs))
        if len(batch) < bs:
            batch.append(d)
        else:
            yield zip(*batch)
            batch = [d]
            if len(d[0]) < max_length:
                bs = batch_size
            else:
                bs = max(1, batch_size * max_length / len(d[0]))
    if batch:
        yield zip(*batch)
