import numpy as np
from scipy import linalg


def lu_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform LU decomposition of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            The permutation matrix P, lower triangular matrix L, and upper triangular matrix U.
    """
    P, L, U = linalg.lu(x)
    return P, L, U


def qr_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform QR decomposition of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray]: The orthogonal matrix Q and upper triangular matrix R.
    """
    Q, R = np.linalg.qr(x)
    return Q, R


def determinant(x: np.ndarray) -> np.ndarray:
    """
    Calculate the determinant of a matrix.

    Args:
        x (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The determinant of the matrix.
    """
    return np.linalg.det(x)


def eigen(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and right eigenvectors of a matrix.

    Args:
        x (np.ndarray): The input matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: The eigenvalues and the right eigenvectors of the matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eig(x)
    return eigenvalues, eigenvectors


def svd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Singular Value Decomposition (SVD) of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The matrices U, S, and V.
    """
    U, S, Vt = np.linalg.svd(x)
    return U, S, Vt


# Test functions
def test_lu_decomposition():
    x = np.array([[4, 3, 2, 1], [3, 4, 1, 2], [2, 1, 4, 3], [1, 2, 3, 4]])
    P, L, U = lu_decomposition(x)
    print("LU Decomposition:")
    print("P:\n", P)
    print("L:\n", L)
    print("U:\n", U)


def test_qr_decomposition():
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    Q, R = qr_decomposition(x)
    print("\nQR Decomposition:")
    print("Q:\n", Q)
    print("R:\n", R)


def test_determinant():
    x = np.array([[1, 2], [3, 4]])
    det = determinant(x)
    print("\nDeterminant:")
    print("det(A):", det)


def test_eigen():
    x = np.array([[4, -2], [1, 1]])
    eigenvalues, eigenvectors = eigen(x)
    print("\nEigenvalues and Eigenvectors:")
    print("Eigenvalues:\n", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)


def test_svd():
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=float)
    U, S, Vt = svd(x)
    print("\nSingular Value Decomposition (SVD):")
    print("U:\n", U)
    print("S:\n", S)
    print("Vt:\n", Vt)


if __name__ == "__main__":
    test_lu_decomposition()
    test_qr_decomposition()
    test_determinant()
    test_eigen()
    test_svd()
