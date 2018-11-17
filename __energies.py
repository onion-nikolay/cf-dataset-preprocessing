from matplotlib import pyplot as plt
import numpy as np
import os
import cv2 as cv
from os.path import join as pjoin


path = pjoin('input', 'Energy')
names = np.sort(os.listdir(path))
photo = cv.imread(pjoin('output', 'result0004', 'grayscale', '_MG_0066.JPG'),
                  0)
mean_energy = np.sum(photo[photo > 0])/np.size(photo[photo > 0])
energies = []
for name in names[1:]:
    img = cv.imread(pjoin(path, name), 0)
    sum_energy = np.sum(img[img > 0])
    sum_size = np.size(img[img > 0])
    energies.append(sum_energy/sum_size)
energies = np.array(energies)
x = np.arange(0.01, 10, 0.01)
plt.plot(x, energies, label='experimental')
plt.plot(x, [mean_energy]*len(1+x), '-', label='nesessery')
coef1 = np.polyfit(np.log(x), energies, 1)
predicted_energies = coef1[1]*np.log(1+x)
plt.plot(x, predicted_energies, label='predicted')
plt.suptitle("Mean energy: {:2.2f}, light level: {:2.2f}".format(mean_energy,
             x[np.argmin(np.abs(energies - mean_energy))]))
plt.legend()
plt.show()
print("MSE: {}".format((np.square(energies - predicted_energies)).mean()))
print("y = {:2.4f}log(x) + {:2.4f}".format(coef1[0], coef1[1]))
