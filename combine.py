import numpy
import pickle
from sklearn.metrics import accuracy_score


def read_pred_score(file):
    scores = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            s = float(line)
            scores.append([1 - s, s])
    return numpy.asarray(scores)


def read_truth(file='data/data_emb'):
    with open(file, 'rb') as f:
        all_sets, _, _ = pickle.load(f)
    return all_sets[2]['y']


def combine_score(score_list, weight_list):
    sum_score = numpy.zeros_like(score_list[0])
    for score, weight in zip(score_list, weight_list):
        sum_score += score * weight
    return numpy.argmax(sum_score, axis=-1)


def grid_search():
    nn_scores = read_pred_score('pred/cnn_diff.pred')
    frame_scores = read_pred_score('pred/lgb.pred')
    assert len(nn_scores) == len(frame_scores)
    truth = read_truth()
    alpha = 0.0
    step = 0.01
    best_score = 0
    best_alpha = -1
    while alpha <= 1.0001:
        pred = combine_score([nn_scores, frame_scores], [alpha, 1 - alpha])
        acc = accuracy_score(truth, pred)
        if acc > best_score:
            best_score, best_alpha = acc, alpha
        print(f'nn={alpha:.2f} frame={1-alpha:.2f} acc={acc:.4f}')
        alpha += step
    print(f'Best score:{best_score:.4f} alpha:{best_alpha:.2f}')


if __name__ == '__main__':
    grid_search()
