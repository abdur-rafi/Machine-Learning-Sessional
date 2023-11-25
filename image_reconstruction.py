import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt

IMG_SIZE = (500, 500)

def low_rank_approximation(A : np.ndarray, k : int) -> np.ndarray:

    U, S, V = np.linalg.svd(A)
    S = np.diag(S)
    return U[:, :k] @ S[:k, :k] @ V[:k, :]


def loadImage(path):
    img = Image.open(path)
    img.thumbnail(IMG_SIZE, Image.ANTIALIAS)
    img = img.convert('L')
    return np.array(img)


def main():
    img = loadImage('image.jpg')
    
    imgs = []
    ks = [5, 10, 15, 25, 30, 50, 70, 100, 150, 200, 250, 300 ]
    
    numImages = len(ks)

    for k in ks:
        img_k = low_rank_approximation(img, k)
        imgs.append(img_k)
    
    numRows = math.ceil(numImages / 4)
    fig, axs = plt.subplots(numRows, 4, figsize=(12, numRows * 3.5))
    axs = axs.flatten()

    for i in range(numImages):
        axs[i].imshow(imgs[i], cmap='gray')
        axs[i].set_title(f'num components = {ks[i]}', fontsize=10)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
