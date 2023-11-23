import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt

IMG_SIZE = (500, 500)

def low_rank_approximation(A : np.ndarray, k : int) -> np.ndarray:

    U, S, V = np.linalg.svd(A)
    S = np.diag(S)
    A_k = U[:, :k] @ S[:k, :k] @ V[:k, :]
    return A_k

def loadImage(path):
    img = Image.open(path)
    img.thumbnail(IMG_SIZE, Image.ANTIALIAS)
    img = img.convert('L')
    return np.array(img)


def main():
    img = loadImage('image.jpg')
    

    minDim = min(img.shape)

    imgs = []
    ks = [5, 10, 15, 20, 30, 50, 70, 100, 150, 200, 250, 300 ]
    
    numImages = len(ks)

    for k in ks:
        img_k = low_rank_approximation(img, k)
        imgs.append(img_k)
        # plt.imshow(img_k, cmap='gray')
        # plt.show()
    
    numRows = math.ceil(numImages / 4)
    fig, axs = plt.subplots(numRows, 4, figsize=(12, numRows * 3.5))
    axs = axs.flatten()

    for i in range(numImages):
        axs[i].imshow(imgs[i], cmap='gray')
        axs[i].set_title(f'num components = {ks[i]}', fontsize=10)
    
    plt.tight_layout()
    plt.show()


    # for k in range(1, minDim, int(minDim/10)):

    #     img_k = low_rank_approximation(img, k)
        
    #     plt.imshow(img_k, cmap='gray')
    #     plt.show()
    

    # print(img.shape)
    
    # U, S, V = np.linalg.svd(img)


if __name__ == '__main__':
    main()
