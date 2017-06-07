import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot(labels, predictions, filename=None):
    conf_arr = tf.confusion_matrix(
        labels,
        predictions,
        num_classes=11
    ).eval()

    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a + 1e-5))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Reds, 
                    interpolation='nearest')

    width, height = conf_arr.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(x, y), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    plt.xticks(range(width), '123')
    plt.yticks(range(height), '123')
    plt.xlabel('Ground truth labels')
    plt.ylabel('Predicted labels')
    # plt.savefig('confusion_matrix.png', format='png')
    plt.show()

