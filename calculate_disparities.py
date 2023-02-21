import cv2 as cv
import numpy as np
import argparse
from pathlib import Path

# Com ou sem barra de progresso
from tqdm.contrib.itertools import product
# from itertools import product

import matplotlib.pyplot as plt
import threading
import time



def SSD(window1: np.array, window2: np.array):
    return np.sum((window1 - window2)**2)

def robustSSD(window1: np.array, window2: np.array):
    d_sq = np.sum((window1 - window2)**2)
    return d_sq / (d_sq + e_sq)


disp_mutex = threading.Lock()
def findDisparity(window, pad, eq_i, eq_j, img, out, cost_function):
    disp = 200
    min_val = 100000

    im_width = img.shape[0] - (2*pad)

    for j in range(pad, im_width + pad - 1): 
        window2 = img[
            eq_i - pad : eq_i + pad + 1,
            j - pad : j + pad + 1
        ]
        val = cost_function(window, window2)
        if min_val > val:
            min_val = val
            disp = eq_j - j
    
    out[eq_i - pad, eq_j - pad] = disp
    




if __name__ == '__main__':

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

    parser = argparse.ArgumentParser(
        prog='Calculate Disparities',
        description='Calcula matriz de disparidades entre duas images'
    )
    parser.add_argument('img1')
    parser.add_argument('img2')
    parser.add_argument('-w', '--window', type=int, default=1)
    parser.add_argument('-r', '--robust', action='store_true')
    parser.add_argument('-re', '--robust_value', type=int, default=5)
    parser.add_argument('-ui', '--updateinterval', type=int, default=1000)

    args = parser.parse_args()
    arg_dict = vars(args)

    print('PARAMETROS:')
    for arg in arg_dict:
        print(f'  {arg} = {arg_dict[arg]}')

    # Nomes das imagens
    img_name = Path(args.img1).stem + '-' + Path(args.img2).stem 
    img_name1 = args.img1
    img_name2 = args.img2
    # Tamanho da janela
    WINDOW_SIZE = args.window
    pad = max(0, WINDOW_SIZE - 2)
    # Função de erro
    if args.robust:
        global robust_e 
        global e_sq 
        robust_e = args.robust_value
        e_sq = robust_e**2
        error_function = robustSSD
        error_function_name  = 'robustSSD_' + str(robust_e)  

    else:
        error_function = SSD
        error_function_name  = 'SSD'     

    # Intervalo de atualização da barra de progresso
    updateinterval = args.updateinterval



    # Par de imagens estéreo
    img1 = cv.imread(img_name1)
    img2 = cv.imread(img_name2)
    
    height, width = img1.shape[0], img1.shape[1]

    # Conversão para CIELAB
    cv.cvtColor(img1, cv.COLOR_BGR2LAB, img1)
    cv.cvtColor(img2, cv.COLOR_BGR2LAB, img2)
 
    # Padding
    if pad > 0:
        img1 = cv.copyMakeBorder(img1, pad, pad, pad, pad, cv.BORDER_CONSTANT, (0,0));
        img2 = cv.copyMakeBorder(img2, pad, pad, pad, pad, cv.BORDER_CONSTANT, (0,0));

    disp_matrix = np.zeros((height, width))

    disp_threads = []
    try:
        # Para cada pixel, obtém sua janela de vizinhos e calcula erro
        # for i, j in product(range(pad, height + pad), range(pad, width + pad)):
        for i, j in product(range(pad, height + pad), range(pad, width + pad), miniters=updateinterval):
            # Janelas
            window1 = img1[
                i - pad : i + pad + 1,
                j - pad : j + pad + 1
            ]
            findDisparity(window1, pad, i, j, img2, disp_matrix, error_function)
    except Exception as e:
        print(e)
    plt.matshow(disp_matrix)

    output_filename = 'disparities/'+ error_function_name + '-' + f'{WINDOW_SIZE}x{WINDOW_SIZE}-' + img_name + '-' + timestamp + f'-{width}x{height}'
    output_filename_latest = 'disparities/latest/'+ error_function_name + '-' + f'{WINDOW_SIZE}x{WINDOW_SIZE}-' + img_name

    # Salva a matriz resultante e o gráfico
    np.save(output_filename, disp_matrix)
    np.save(output_filename_latest, disp_matrix)
    
    cv.imwrite(output_filename_latest + '.png', disp_matrix)

    # Mostra resultado
    plt.show()