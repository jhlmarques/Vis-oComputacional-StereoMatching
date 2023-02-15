import cv2 as cv
import numpy as np
from tqdm.contrib.itertools import product
import matplotlib.pyplot as plt
import threading


def SSD(window1: np.array, window2: np.array):
    return np.sum((window1 - window2)**2)

disp_mutex = threading.Lock()

def findDisparity(window, eq_i, eq_j, img, out):
    disp = 0
    min_val = 100000

    im_width = img.shape[0]

    for j in range(im_width): 
        window2 = img[
            max(0, eq_i - WINDOW_SIZE) : min(width, eq_i + WINDOW_SIZE),
            max(0, eq_j - WINDOW_SIZE) : min(width, eq_j + WINDOW_SIZE)
        ]
        val = SSD(window, window2)
        if min_val > val:
            min_val = val
            disp = eq_j - j
    
    disp_mutex.acquire()
    out[eq_i, eq_j] = disp
    disp_mutex.release()
    



WINDOW_SIZE = 1

if __name__ == '__main__':

    # Par de imagens estéreo
    img1 = cv.imread('cones_im2.png')
    img2 = cv.imread('cones_im6.png')
    # Conversão para CIELAB
    cv.cvtColor(img1, cv.COLOR_BGR2LAB, img1)
    cv.cvtColor(img2, cv.COLOR_BGR2LAB, img2)

    width = img1.shape[0]
    height = img1.shape[1]

    disp_matrix = np.zeros_like(img1)



    disp_threads = []

    # Para cada pixel, obtém sua janela de vizinhos e calcula erro
    for i, j in product(range(width), range(height)):
        # Janelas
        window1 = img1[
            max(0, i - WINDOW_SIZE) : min(width, i + WINDOW_SIZE),
            max(0, j - WINDOW_SIZE) : min(width, j + WINDOW_SIZE)
        ]
        th = threading.Thread(target=findDisparity, args=(window1, i, j, img1, disp_matrix))
        th.start()
        disp_threads.append(th)

    for th in disp_threads:
        th.join()


    plt.matshow(disp_matrix)
    plt.show()

    

