import argparse
import numpy as np
import cv2 as cv

def RMSE(img1, img2):
    return np.sqrt(np.sum((img1 - img2)**2) / img1.size)

def bad_pixels(img1, img2, threshold=5):
    disp_diffs = np.abs(img2 - img1)
    return ((np.extract(disp_diffs > threshold, disp_diffs).size) / img1.size) * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Compare to Ground Truth',
        description='Compara um mapa de disparidades a uma imagem GT'
    )
    parser.add_argument('window_size', type=int)
    parser.add_argument('file_prefix')
    parser.add_argument('-r', '--robust', action='store_true')
    parser.add_argument('-re', '--robust_value', type=float, default=20)
    parser.add_argument('-d', '--disparity_max', type=int, default=64)

    args = parser.parse_args()

    WINDOW_SIZE = args.window_size
    DISPARITY_MAX = args.disparity_max

    search_dir = 'disparities/latest/'
    
    if args.robust:
        search_dir += 'robustSSD_' + str(args.robust_value)
    else:
        search_dir += 'SSD'
    search_dir += '-'

    search_dir += f'{WINDOW_SIZE}x{WINDOW_SIZE}-d{DISPARITY_MAX}-{args.file_prefix}_im2-{args.file_prefix}_im6.npy'

    img = np.load(search_dir)
    img_gt = cv.imread(f'{args.file_prefix}_disp2.png', cv.IMREAD_GRAYSCALE).astype(np.float64)
    img_gt /= 4



    print(f"Avaliação quantitativa para '{args.file_prefix}' {f'função robusta com e={args.robust_value}' if args.robust else 'SSD'}, janela {WINDOW_SIZE}x{WINDOW_SIZE} e disparidade máxima {DISPARITY_MAX}")
    print("------------------------------------------------------------------------")

    print(f'RMSE: {RMSE(img, img_gt)}')
    print(f'Bad Pixels: {bad_pixels(img, img_gt)}')
    
    half_w = WINDOW_SIZE // 2
    img = img[half_w : img.shape[0] - half_w + 1, half_w : img.shape[1] - half_w + 1]
    img_gt = img_gt[half_w : img_gt.shape[0] - half_w + 1, half_w : img_gt.shape[1] - half_w + 1]
    # print(img.shape)
    # print(img_gt.shape)
    print(f'RMSE (no borders): {RMSE(img, img_gt)}')
    print(f'Bad Pixels (no borders): {bad_pixels(img, img_gt)}')

    img = img[DISPARITY_MAX:]
    img_gt = img_gt[DISPARITY_MAX:]

    # print(img.shape)
    # print(img_gt.shape)
    print(f'RMSE (no borders + disp offset): {RMSE(img, img_gt)}')
    print(f'Bad Pixels (no borders + disp offset): {bad_pixels(img, img_gt)}')