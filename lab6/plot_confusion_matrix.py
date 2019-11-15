import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print('cm', cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...

    print(cm.shape)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=(np.arange(cm.shape[0])),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True category',
           xlabel='Predicted category')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt. ylim(13.5, -0.5)  # return the current ylim

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return ax


def map_classes_to_ids():
    with open('./class_ids.txt', 'r') as f:
        lines = f.readlines()
        class_ids = dict()

        for line in lines:
            category, category_id = line.replace('\n', '').split('\t')
            class_ids[category] = category_id

        return class_ids


def get_real_classes(filename):
    with open(filename) as f:
        tweets = f.readlines()
        y_true = [t.split('\t')[2].replace('\n', '') for t in tweets]
        return y_true


def get_pred_classes(filename):
    with open(filename) as f:
        predictions = f.readlines()
        y_pred = [int(t.split(' ')[0]) for t in predictions]
        return y_pred


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
class_ids = map_classes_to_ids()
class_names = list(class_ids.keys())

y_true = get_real_classes('./tweetsclassification/tweets.test')
y_true = [int(class_ids[y]) for y in y_true]

y_pred = get_pred_classes('./outputs/3pred.out')

print(len(y_true))
print(len(y_pred))

plot_confusion_matrix(y_true, y_pred, classes=class_names)

plt.show()
