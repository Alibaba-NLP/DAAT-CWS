from __future__ import print_function
import os, codecs
from itertools import izip

def process_train_sentence(sentence):
    sentence = sentence.strip()
    words = sentence.split()
    chars = []
    tags = []
    ret = []
    for w in words:
        chars.extend(list(w))
        if len(w) == 1:
            tags.append('S')
        else:
            tags.extend(['B'] + ['M'] * (len(w) - 2) + ['E'])
    ret.append(chars)
    ret.append(tags)
    return ret

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


def read_train_file(fin):
    """
    Read training data.
    """
    data = []
    for l in fin:
        data.append(process_train_sentence(l))
    return zip(*data)


def create_output(seqs, stags):
    """
    Create final output from characters and BMES tags.
    """
    output = []
    for seq, stag in izip(seqs, stags):
        new_sen = []
        for c, tag in izip(seq, stag):
            new_sen.append(c)
            if tag == 'S' or tag == 'E':
                new_sen.append('  ')
        output.append(''.join(new_sen))
    return output


def evaluator(data, output_dir, output_flag):
    """
    Evaluate presion, recall and F1.
    """
    seqs, gold_stags, pred_stags = data
    assert len(seqs) == len(gold_stags) == len(pred_stags)
    # Create and open temp files.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ref_path = os.path.join(output_dir, '%s.ref' % output_flag)
    pred_path = os.path.join(output_dir, '%s.pred' % output_flag)
    score_path = os.path.join(output_dir, '%s.score' % output_flag)
    # Empty words file.
    temp_path = os.path.join(output_dir, '%s.temp' % output_flag)

    ref_file = codecs.open(ref_path, 'w', 'utf8')
    pred_file = codecs.open(pred_path, 'w', 'utf8')
    for l in create_output(seqs, gold_stags):
        print(l, file=ref_file)
    for i, l in enumerate(create_output(seqs, pred_stags)):
        print(l, file=pred_file)
    ref_file.close()
    pred_file.close()

    os.system('echo > %s' % temp_path)
    os.system('%s  %s %s %s > %s' % ('./score.perl', temp_path, ref_path, pred_path, score_path))
    # Sighan evaluation results
    os.system('tail -n 7 %s > %s' % (score_path, temp_path))
    eval_lines = [l.rstrip() for l in codecs.open(temp_path, 'r', 'utf8')]
    # Remove temp files.
    os.remove(ref_path)
    os.remove(pred_path)
    os.remove(score_path)
    os.remove(temp_path)
    # Precision, Recall and F1 score
    return (float(eval_lines[1].split(':')[1]),
            float(eval_lines[0].split(':')[1]),
            float(eval_lines[2].split(':')[1]))

