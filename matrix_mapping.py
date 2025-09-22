import numpy as np

def negative_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the negation of each element in the input vector or matrix.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with each element negated.
    """
    return -x


def reverse_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the input vector or matrix with the order of elements reversed.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with the order of elements reversed.
    """
    return np.flip(x)


def affine_transform(
    x: np.ndarray, alpha_deg: float, scale: tuple[float, float], shear: tuple[float, float],
    translate: tuple[float, float],
) -> np.ndarray:
    """Compute affine transformation

    Args:
        x (np.ndarray): vector n*1 or matrix n*n.
        alpha_deg (float): rotation angle in deg.
        scale (tuple[float, float]): x, y scale factor.
        shear (tuple[float, float]): x, y shear factor.
        translate (tuple[float, float]): x, y translation factor.

    Returns:
        np.ndarray: transformed matrix.
    """
    alpha_rad = np.radians(alpha_deg)
    cos_alpha = np.cos(alpha_rad)
    sin_alpha = np.sin(alpha_rad)

    rotation_matrix = np.array([
        [cos_alpha, -sin_alpha],
        [sin_alpha,  cos_alpha]
    ])
    
    scale_matrix = np.array([
        [scale[0], 0],
        [0, scale[1]]
    ])

    shear_matrix = np.array([
        [1, shear[0]],
        [shear[1], 1]
    ])
    
    affine_matrix = rotation_matrix @ shear_matrix @ scale_matrix
    transformed = x @ affine_matrix.T
    transformed += np.array(translate)

    return transformed


# Test functions
def test_negative_matrix():
    print('test_negative_matrix')
    # Test with vector
    vector = np.array([[1], [2], [3]])
    print(f'Original vector: {vector.flatten()}')
    print(f'Negative vector: {negative_matrix(vector).flatten()}')
    
    # Test with matrix
    matrix = np.array([[1, 2], [3, 4]])
    print(f'Original matrix:\n{matrix}')
    print(f'Negative matrix:\n{negative_matrix(matrix)}')


def test_reverse_matrix():
    print('test_reverse_matrix')
    # Test with vector
    vector = np.array([[1], [2], [3]])
    print(f'Original vector: {vector.flatten()}')
    print(f'Reversed vector: {reverse_matrix(vector).flatten()}')
    
    # Test with matrix
    matrix = np.array([[1, 2], [3, 4]])
    print(f'Original matrix:\n{matrix}')
    print(f'Reversed matrix:\n{reverse_matrix(matrix)}')


def test_affine_transform():
    print('test_affine_transform')
    # Test with simple 2D points
    points = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    print(f'Original points:\n{points}')
    
    # Apply rotation by 90 degrees
    rotated = affine_transform(points, alpha_deg=90, scale=(1, 1), 
                              shear=(0, 0), translate=(0, 0))
    print(f'Rotated 90° points:\n{rotated}')
    
    # Apply scaling
    scaled = affine_transform(points, alpha_deg=0, scale=(2, 0.5), 
                             shear=(0, 0), translate=(0, 0))
    print(f'Scaled (2x, 0.5x) points:\n{scaled}')
    
    # Apply translation
    translated = affine_transform(points, alpha_deg=0, scale=(1, 1), 
                                 shear=(0, 0), translate=(5, 3))
    print(f'Translated (+5, +3) points:\n{translated}')
    
    # Complex transformation
    complex_transform = affine_transform(points, alpha_deg=45, scale=(1.5, 1.5), 
                                       shear=(0.2, 0), translate=(2, 1))
    print(f'Complex transformation points:\n{complex_transform}')


if __name__ == "__main__":
    test_negative_matrix()
    print()
    test_reverse_matrix()
    print()
    test_affine_transform()