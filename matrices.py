import numpy as np
import sympy as sp
import secrets

seed = secrets.randbits(128)
rand_generator = np.random.default_rng(seed)

def get_matrix(n: int, m: int) -> np.ndarray:
    """Create random matrix n * m.

    Args:
        n (int): number of rows.
        m (int): number of columns.

    Returns:
        np.ndarray: matrix n*m.
    """
    return rand_generator.random((n, m))

def test_get_matrix():
    print('test_get_matrix')
    print(get_matrix(3, 3))
test_get_matrix()

def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrix addition.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: matrix sum.
    """
    return x + y

def test_add():
    print('test_add')
    print(add(get_matrix(3, 3), get_matrix(3, 3)))
test_add()


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Matrix multiplication by scalar.

    Args:
        x (np.ndarray): matrix.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied matrix.
    """
    return a * x

def test_scalar_multiplication():
    x = np.array([[1, 2], [3, 4]])
    print('test_scalar_multiplication')
    print(scalar_multiplication(x, 10))
test_scalar_multiplication()


def dot_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrices dot product.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix or vector.

    Returns:
        np.ndarray: dot product.
    """
    return np.dot(x, y)


def test_dot_product():
    x = [[1, 2], [3, 4]]
    y = [[5, 6], [7, 8]]
    print('test_dot_product')
    print(dot_product(x, y))
test_dot_product()


def identity_matrix(dim: int) -> np.ndarray:
    """Create identity matrix with dimension `dim`. 

    Args:
        dim (int): matrix dimension.

    Returns:
        np.ndarray: identity matrix.
    """
    return np.eye(dim)

def test_identity_matrix():
    print('test_identity_matrix')
    print(identity_matrix(3))
test_identity_matrix()

def matrix_inverse(x: np.ndarray) -> np.ndarray:
    """Compute inverse matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: inverse matrix.
    """
    if np.linalg.det(x) == 0:
        raise ValueError("Matrix is not invertible - determinant is 0")
    return np.linalg.inv(x)

def test_matrix_inverse():
    x = [[1, 2], [3, 4]]
    x_inv = matrix_inverse(x)
    print('test_matrix_inverse')
    print(x_inv)
    print(x  @ x_inv)
test_matrix_inverse()


def matrix_transpose(x: np.ndarray) -> np.ndarray:
    """Compute transpose matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: transosed matrix.
    """
    return x.T

def test_matrix_transpose():
    x = np.array([[1, 2], [3, 4]])
    print('test_matrix_transpose')
    print(matrix_transpose(x))
test_matrix_transpose()


def hadamard_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute hadamard product.

    Args:
        x (np.ndarray): 1th matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: hadamard produc
    """
    return x * y

def basis(x: np.ndarray) -> tuple[int]:
    """Compute matrix basis.

    Args:
        x (np.ndarray): matrix.

    Returns:
        tuple[int]: indexes of basis columns.
    """
    m = sp.Matrix(x)
    _, pivot_columns = m.rref()
    return tuple(pivot_columns)

def test_basis():
    x = np.array([[1, 2], [3, 4]])
    print('test_basis')
    print(basis(x))
test_basis()


def norm(x: np.ndarray, order: int | float | str) -> float:
    """Matrix norm: Frobenius, Spectral or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 'fro', 2 or inf.

    Returns:
        float: vector norm
    """
    return np.linalg.norm(x, order)

def test_norm():
    x = np.array([[1, 2], [3, 4]])
    print('test_norm')
    print(norm(x, 2))
test_norm()
