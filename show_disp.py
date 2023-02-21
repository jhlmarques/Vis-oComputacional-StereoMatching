import matplotlib.pyplot as plt
import numpy as np

images = []
images.append('cones_im2.png')
images.append('cones_im6.png')
images.append(np.load('disparities/SSD-3x3-cones.npy'))
images.append(np.load('disparities/robustSSD_5-3x3-cones.npy'))
images.append('teddy_im2.png')
images.append('teddy_im6.png')
images.append(np.load('disparities/SSD-3x3-teddy.npy'))
images.append(np.load('disparities/robustSSD_5-3x3-teddy.npy'))


fig, axs = plt.subplots(nrows=2, ncols=4, layout="constrained")
for row in range(2):
    for col in range(4):
        if isinstance(images[int(4*row + col)], str):
            images[int(4*row + col)] = plt.imread(images[int(4*row + col)])
        
        axs[row, col].matshow(images[int(4*row + col)])
plt.show()
