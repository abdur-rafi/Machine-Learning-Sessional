import numpy as np


def generateInvertibleSymmetricMatrix(n: int) -> np.ndarray:
    A = np.random.randint(5, size=(n, n))
    A = A + np.transpose(A)
    rowSumSubDiag = np.sum(np.abs(A) , axis=1) - np.abs(np.diag(A))
    # print(rowSumSubDiag) 
    np.fill_diagonal(A, rowSumSubDiag + 1)
    return A


def main():
    # n = 3
    n = int(input("Enter size: "))
    mat = generateInvertibleSymmetricMatrix(n)
    print("Matrix Generated:")
    print(mat)

    eigenvalues, eigenvectors = np.linalg.eig(mat)
    V = eigenvectors
    Diaglmbda = np.diag(eigenvalues)

    # matReconstructed = np.matmul(np.matmul(V, Diaglmbda), np.linalg.inv(V))
    matReconstructed = V @ Diaglmbda @ np.linalg.inv(V)


    print("Reconstructed Matrix: ")
    print(matReconstructed)

    print( "all entries same: " + str(np.allclose(mat, matReconstructed)))



if __name__ == "__main__":
    main()

