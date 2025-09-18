from typing import Sequence

import numpy as np
import secrets
from scipy import sparse

seed = secrets.randbits(128)
rand_generator = np.random.default_rng(seed)

def get_vector(dim: int) -> np.ndarray:
    """Create random column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        np.ndarray: column vector.
    """
    return rand_generator.random((dim, 1))


def get_sparse_vector(dim: int) -> sparse.coo_matrix:
    """Create random sparse column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        sparse.coo_matrix: sparse column vector.
    """
    density = rand_generator.random()
    S = sparse.random(dim, 1, density=density, random_state=rand_generator)
    return S


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector addition. 

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        np.ndarray: vector sum.
    """
    return x + y


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Vector multiplication by scalar.

    Args:
        x (np.ndarray): vector.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied vector.
    """
    return a * x


def linear_combination(vectors: Sequence[np.ndarray], coeffs: Sequence[float]) -> np.ndarray:
    """Linear combination of vectors.

    Args:
        vectors (Sequence[np.ndarray]): list of vectors of len N.
        coeffs (Sequence[float]): list of coefficients of len N.

    Returns:
        np.ndarray: linear combination of vectors.
    """
    if len(vectors) != len(coeffs):
        raise ValueError("Number of vectors and coefficients must be the same.")
    
    # Alternative solution: sum(c * v for c, v in zip(coeffs, vectors))
    return np.vecdot(vectors, coeffs, axis=0)


def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """Vectors dot product.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: dot product.
    """
    # assuming 1-D vectors
    return np.dot(x.flatten(), y.flatten())


def norm(x: np.ndarray, order: int | float) -> float:
    """Vector norm: Manhattan, Euclidean or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 1, 2 or inf.

    Returns:
        float: vector norm
    """
    return np.linalg.norm(x, order)


def distance(x: np.ndarray, y: np.ndarray) -> float:
    """L2 distance between vectors.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: distance.
    """
    return np.linalg.norm(x - y)


def cos_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine between vectors in deg.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.


    Returns:
        np.ndarray: angle in deg.
    """
    norm_x = norm(x, 2)
    norm_y = norm(y, 2)
    if norm_x == 0 or norm_y == 0:
        raise ValueError("Cannot compute angle between zero vectors")
    
    cos_angle = dot_product(x, y) / (norm_x * norm_y)
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


def is_orthogonal(x: np.ndarray, y: np.ndarray) -> bool:
    """Check is vectors orthogonal.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        bool: are vectors orthogonal.
    """
    return dot_product(x, y) == 0


def solves_linear_systems(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve system of linear equations.

    Args:
        a (np.ndarray): coefficient matrix.
        b (np.ndarray): ordinate values.

    Returns:
        np.ndarray: sytems solution
    """
    return np.linalg.solve(a, b)


# Test functions
def test_get_vector():
    print('test_get_vector')
    print(get_vector(10))


def test_get_sparse_vector():
    s = get_sparse_vector(10)
    print('test_get_sparse_vector')
    print(s, s.shape)


def test_add():
    x = get_vector(3)
    y = get_vector(3)
    print('test_add')
    print(f'add {x} and {y} is {add(x, y)}')


def test_scalar_multiplication():
    x = get_vector(3)
    print('test_scalar_multiplication')
    print(f'scalar_multiplication {x} and 10 is {scalar_multiplication(x, 10)}')


def test_linear_combination():
    vectors = [get_vector(3), get_vector(3)]
    coeffs = [2, 2]
    print('test_linear_combination')
    print(f'linear_combination {vectors} and {coeffs} is {linear_combination(vectors, coeffs)}')


def test_dot_product():
    x = get_vector(3)
    y = get_vector(3)
    print('test_dot_product')
    print(f'dot_product {x} and {y} is {dot_product(x, y)}')


def test_norm():
    x = get_vector(3)
    print('test_norm')
    print(f'norm {x} is {norm(x, 1)}')


def test_distance():
    x = get_vector(2)
    y = get_vector(2)
    print('test_distance')
    print(f'distance {x} and {y} is {distance(x, y)}')


def test_cos_between_vectors():
    x = np.array([[0], [1]])
    y = np.array([[1], [0]])
    print('test_cos_between_vectors')
    print(f'cos_between_vectors {x} and {y} is {cos_between_vectors(x, y)} degrees')


def test_is_orthogonal():
    x = np.array([[0], [1]])
    y = np.array([[1], [0]])
    print('test_is_orthogonal')
    print(f'is_orthogonal {x} and {y} is {is_orthogonal(x, y)}')


def test_solves_linear_systems():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    print('test_solves_linear_systems')
    print(f'solves_linear_systems {a} and {b} is {solves_linear_systems(a, b)}')


if __name__ == "__main__":
    test_get_vector()
    test_get_sparse_vector()
    test_add()
    test_scalar_multiplication()
    test_linear_combination()
    test_dot_product()
    test_norm()
    test_distance()
    test_cos_between_vectors()
    test_is_orthogonal()
    test_solves_linear_systems()