from matplotlib import pyplot as plt
import numpy as np
import os
import cv2 as cv


path = r'E:\Energy gradient'
folders = np.sort(os.listdir(path))
photo = cv.imread(r'output\result0001\step_1\photo_2018-11-04_17-34-56.jpg', 0)
mean_energy = np.sum(photo[photo > 0])/np.size(photo[photo > 0])
energies = []
for fld in folders[:-1]:
    p = path+'\\'+fld
    name = p + '\\' + os.listdir(p)[3]
    img = cv.imread(name, 0)
    energies.append(np.sum(img[img > 0])/np.size(img[img > 0]))
energies = np.array(energies)
x = np.array([float(fld[-4:]) for fld in folders[:-1]])
plt.plot(x, energies, label='experimental')
plt.plot(x, [mean_energy]*len(folders[:-1]), '-', label='nesessery')
coef1 = np.polyfit(x, energies, 1)
predicted_energies = coef1[0]+coef1[1]*x
plt.plot(x, predicted_energies, label='predicted')
plt.suptitle("Mean energy: {:2.2f}, light level: {}".format(mean_energy,
             x[np.argmin(np.abs(energies - mean_energy))]))
plt.legend()
plt.show()
print("MSE: {}".format((np.square(energies - predicted_energies)).mean()))
print("y = {:2.4f}x + {:2.4f}".format(coef1[0], coef1[1]))
