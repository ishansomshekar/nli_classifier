import numpy as np
import matplotlib.pyplot as plt

train_accuracies = [0.234375, 0.34375, 0.546875, 0.65625, 0.796875, 0.84375, 0.9375, 0.96875, 0.96875, 1.0]
dev_accuracies = [0.147272727273, 0.209090909091, 0.307272727273, 0.335454545455, 0.333636363636, 0.332727272727, 0.335454545455, 0.324545454545, 0.324545454545, 0.325454545455]

Epochs = np.arange(1, 11)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(Epochs, train_accuracies, '-b', label='Train accuracy')
plt.plot(Epochs, dev_accuracies, '-g', label='Dev accuracy')
plt.legend(loc='lower left')
plt.show()

# 0.25  - Dev score: 0.187272727273
# Epoch 1  - Train score: 0.453125  - Dev score: 0.355454545455
# Epoch 2  - Train score: 0.671875  - Dev score: 0.398181818182
# Epoch 3  - Train score: 0.8125  - Dev score: 0.414545454545
# Epoch 4  - Train score: 0.875  - Dev score: 0.41
# Epoch 5  - Train score: 0.9375  - Dev score: 0.362727272727
# Epoch 6  - Train score: 0.96875  - Dev score: 0.369090909091
# Epoch 7  - Train score: 0.984375  - Dev score: 0.376363636364
# Epoch 8  - Train score: 1.0  - Dev score: 0.390909090909
# Epoch 9  - Train score: 1.0  - Dev score: 0.38

# Epoch 0  - Train score: 0.234375  - Dev score: 0.147272727273
# Epoch 1  - Train score: 0.34375  - Dev score: 0.209090909091
# Epoch 2  - Train score: 0.546875  - Dev score: 0.307272727273
# Epoch 3  - Train score: 0.65625  - Dev score: 0.335454545455
# Epoch 4  - Train score: 0.796875  - Dev score: 0.333636363636
# Epoch 5  - Train score: 0.84375  - Dev score: 0.332727272727
# Epoch 6  - Train score: 0.9375  - Dev score: 0.335454545455
# Epoch 7  - Train score: 0.96875  - Dev score: 0.334545454545
# Epoch 8  - Train score: 0.96875  - Dev score: 0.324545454545
# Epoch 9  - Train score: 1.0  - Dev score: 0.325454545455