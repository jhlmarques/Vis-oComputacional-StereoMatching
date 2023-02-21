import cv2 as cv
import numpy as np

# Com ou sem barra de progresso
from tqdm.contrib.itertools import product
# from itertools import product

import matplotlib.pyplot as plt
import threading
import time


WINDOW_SIZE = 3
pad = max(0, WINDOW_SIZE - 2)

timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
img_name = 'cones'

e = 5
e_sq = e**2

def SSD(window1: np.array, window2: np.array):
    return np.sum((window1 - window2)**2)

def robustSSD(window1: np.array, window2: np.array):
    d_sq = np.sum((window1 - window2)**2)
    return d_sq / (d_sq + e_sq)

error_function = robustSSD
error_function_name  = 'robustSSD_' + str(e) + '-'

# error_function = SSD
# error_function_name  = 'SSD-'


disp_mutex = threading.Lock()

def findDisparity(window, eq_i, eq_j, img, out, cost_function):
    disp = 200
    min_val = 100000

    im_width = img.shape[0] - (2*pad)

    for j in range(pad, im_width + pad - 1): 
        window2 = img[
            eq_i - pad : eq_i + pad,
            j - pad : j + pad
        ]
        val = cost_function(window, window2)
        if min_val > val:
            min_val = val
            disp = eq_j - j
    
    disp_mutex.acquire()
    #print(f'Found disparity for ({eq_i - pad}, {eq_j - pad}) = {disp}')
    out[eq_i - pad, eq_j - pad] = disp
    disp_mutex.release()
    




if __name__ == '__main__':

    # Par de imagens estéreo
    img1 = cv.imread(img_name + '_im2.png')
    img2 = cv.imread(img_name + '_im6.png')
    
    height, width = img1.shape[0], img1.shape[1]

    # Conversão para CIELAB
    cv.cvtColor(img1, cv.COLOR_BGR2LAB, img1)
    cv.cvtColor(img2, cv.COLOR_BGR2LAB, img2)
 
    # Padding
    img1 = cv.copyMakeBorder(img1, pad, pad, pad, pad, cv.BORDER_CONSTANT, (0,0));
    img2 = cv.copyMakeBorder(img2, pad, pad, pad, pad, cv.BORDER_CONSTANT, (0,0));

    disp_matrix = np.zeros((height, width))

    disp_threads = []
    try:
        # Para cada pixel, obtém sua janela de vizinhos e calcula erro
        # for i, j in product(range(pad, height + pad), range(pad, width + pad)):
        for i, j in product(range(pad, height + pad), range(pad, width + pad), miniters=1000):
            # Janelas
            window1 = img1[
                i - pad : i + pad,
                j - pad : j + pad
            ]
            # Execução paralela
            th = threading.Thread(target=findDisparity, args=(window1, i, j, img2, disp_matrix, error_function))
            th.start()
            disp_threads.append(th)
    except Exception as e:
        print(e)
    finally:
        for th in disp_threads:
            th.join()

    plt.matshow(disp_matrix)

    output_filename = 'disparities/'+ error_function_name + f'{WINDOW_SIZE}x{WINDOW_SIZE}-' + img_name + '-' + timestamp + f'-{width}x{height}'
    
    # Salva a matriz resultante e o gráfico
    np.save(output_filename, disp_matrix)

    # Mostra resultado
    plt.show()