import cv2 as cv
import numpy as np
import argparse
from pathlib import Path

# Com ou sem barra de progresso
from tqdm.contrib.itertools import product
# from itertools import product

import matplotlib.pyplot as plt
import time



def SSD(window1: np.array, window2: np.array):
    return np.sum((window1 - window2)**2)

def robustSSD(window1: np.array, window2: np.array):
    d_sq = np.sum((window1 - window2)**2, axis=2)
    return np.sum(d_sq / (d_sq + e_sq))


if __name__ == '__main__':

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

    parser = argparse.ArgumentParser(
        prog='Calculate Disparities',
        description='Calcula matriz de disparidades entre duas images'
    )
    parser.add_argument('left_img')
    parser.add_argument('right_img')
    parser.add_argument('-s', '--search_range', type=int, default=64)
    parser.add_argument('-w', '--window', type=int, default=3)
    parser.add_argument('-r', '--robust', action='store_true')
    parser.add_argument('-re', '--robust_value', type=float, default=20)
    parser.add_argument('-ui', '--updateinterval', type=int, default=1000)

    args = parser.parse_args()
    arg_dict = vars(args)

    print('PARAMETROS:')
    for arg in arg_dict:
        print(f'  {arg} = {arg_dict[arg]}')

    # Nomes das imagens
    img_name = Path(args.left_img).stem + '-' + Path(args.right_img).stem 
    img_name1 = args.left_img
    img_name2 = args.right_img
    # Tamanho da janela
    WINDOW_SIZE = args.window
    kernel_half = max(0, WINDOW_SIZE // 2)
    # Intervalo de calculo de disparidade (à esquerda)
    search_range = args.search_range
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
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2LAB)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2LAB)
 
    disp_matrix = np.zeros((height, width))


    # Para cada pixel, obtém sua janela de vizinhos e calcula erro
    # for i, j in product(range(kernel_half, height + kernel_half), range(kernel_half, width + kernel_half)):
    for i, j in product(range(kernel_half, height - kernel_half), range(kernel_half, width - kernel_half), miniters=updateinterval):
        # print('----')
        # print(i, j)
        # Janelas
        window1 = img1[
            i - kernel_half : i + kernel_half + 1,
            j - kernel_half : j + kernel_half + 1
        ]

        min_val = 999999
        disp = 0

        window_y1 = i - kernel_half
        window_y2 = i + kernel_half + 1
        window_x1 = j - kernel_half
        window_x2 = j + kernel_half + 1

        # print(window_y1, window_y2)
        # print(window_x1, window_x2)

        maximum_offset = min(search_range, window_x1)

        # Considerando que comparamos a câmera direita com a esquerda, pixels
        # moveriam-se à esquerda
        for offset in range(maximum_offset):
            window2 = img2[
                window_y1 : window_y2,
                window_x1 - offset : window_x2 - offset
            ]
            val = error_function(window1, window2)
            if min_val > val:
                min_val = val
                disp = offset
        disp_matrix[i, j] = disp

    plt.style.use('grayscale')
    plt.matshow(disp_matrix)

    output_filename = 'disparities/'+ error_function_name + '-' + f'{WINDOW_SIZE}x{WINDOW_SIZE}-' + f'd{search_range}-' + img_name + '-' + timestamp + f'-{width}x{height}'
    output_filename_latest = 'disparities/latest/'+ error_function_name + '-' + f'{WINDOW_SIZE}x{WINDOW_SIZE}-' + f'd{search_range}-' + img_name

    # Salva a matriz resultante e o gráfico
    np.save(output_filename, disp_matrix)
    np.save(output_filename_latest, disp_matrix)
    
    # Normaliza matriz para o maior valor e atribui para grayscale
    cv.normalize(disp_matrix, disp_matrix, 1.0, 0.0, cv.NORM_INF)
    disp_matrix = (disp_matrix * 255).astype(int)

    cv.imwrite(output_filename_latest + '.png', disp_matrix)

    # Mostra resultado
    plt.show()