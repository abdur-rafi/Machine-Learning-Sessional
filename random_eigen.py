import numpy as np


def generateInvertibleMatrix(n: int) -> np.ndarray:
    A = np.random.randint(5, size=(n, n))
    rowSumSubDiag = np.sum(np.abs(A) , axis=1) - np.abs(np.diag(A))
    # print(rowSumSubDiag) 
    np.fill_diagonal(A, rowSumSubDiag + 1)
    return A


def main():
    n = 3
    mat = generateInvertibleMatrix(n)
    print("Matrix A:")
    print(mat)

    eigenvalues, eigenvectors = np.linalg.eig(mat)
    # print("Eigenvalues:")
    # print(eigenvalues)
    # print("Eigenvectors:")
    # print(eigenvectors)

    V = eigenvectors
    Diaglmbda = np.diag(eigenvalues)

    # matReconstructed = np.matmul(np.matmul(V, Diaglmbda), np.linalg.inv(V))
    matReconstructed = V @ Diaglmbda @ np.linalg.inv(V)


    print("Reconstructed Matrix: ")
    print(matReconstructed)
    print(np.allclose(mat, matReconstructed))


if __name__ == "__main__":
    main()

