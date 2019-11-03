import numpy as np
import argparse


def load_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        results = [line.split(' ')[0] for line in lines]
        return results


def accuracy(actual_classes, predicted_classes):
    comparisons = actual_classes == predicted_classes
    correct_predictions = np.sum(comparisons[comparisons == True])
    return correct_predictions / len(actual_classes)


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


def f1_score(prec, rec):
    return 2 * (prec * rec / (prec + rec))


def class_scores(actual_classes, predicted_classes):
    class_precision = []
    class_recall = []
    class_f1_score = []
    for positive_class in range(1, 15):
        # Assume the current class is the positive one and the others are all negative
        positive_class = str(positive_class)
        tp = np.sum((actual_classes == positive_class) & (predicted_classes == positive_class))
        fp = np.sum((actual_classes != positive_class) & (predicted_classes == positive_class))
        fn = np.sum((actual_classes == positive_class) & (predicted_classes != positive_class))
        prec = precision(tp, fp)
        rec = recall(tp, fn)
        f1 = f1_score(prec, rec)

        class_precision.append(prec)
        class_recall.append(rec)
        class_f1_score.append(f1)

    # return sum(f1_scores) / 14
    return class_precision, class_recall, class_f1_score


def write_scores_to_file(acc, macro_f1_score, class_precision, class_recall, class_f1_score, filename):
    with open('./results/' + filename, 'w') as f:
        f.write('Accuracy = %.3f\n' % acc)
        f.write('Macro-F1 = %.3f\n' % macro_f1_score)
        f.write('Results per class:\n')

        for i in range(14):
            f.write('%s: P=%.3f R=%.3f F=%.3f\n' % (str(i+1), class_precision[i], class_recall[i], class_f1_score[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_file', type=str)
    parser.add_argument('out_file', type=str)
    parser.add_argument('eval_file', type=str)
    opts = parser.parse_args()

    actual_classes = np.array(load_file(opts.test_file))
    predicted_classes = np.array(load_file(opts.out_file))

    # Scores
    acc = accuracy(actual_classes, predicted_classes)
    class_precision, class_recall, class_f1_score = class_scores(actual_classes, predicted_classes)
    macro_f1_score = sum(class_f1_score) / 14

    write_scores_to_file(acc, macro_f1_score, class_precision, class_recall, class_f1_score, opts.eval_file)
