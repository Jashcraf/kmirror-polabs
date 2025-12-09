# Methods for generally elliptical diattenuator / retarder
# decomposition using Pauli spin matrices
import numpy as np

def _empty_jones(shape=None):
    """Returns an empty array to populate with jones matrix elements.

    Parameters
    ----------
    shape : list
        shape to prepend to the jones matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.

    Returns
    -------
    ndarray
        The empty array of specified shape
    """

    if shape is None:

        shape = (2, 2)

    else:

        shape = (*shape, 2, 2)

    return np.zeros(shape, dtype=np.complex128)

def pauli_spin_matrix(index, shape=None):
    """Generates a pauli spin matrix used for Jones matrix data reduction. From CLY Eq 6.108.

    Parameters
    ----------
    index : int
        0 - the identity matrix
        1 - a linear half-wave retarder oriented horizontally
        2 - a linear half-wave retarder oriented 45 degrees
        3 - a circular half-wave retarder
    shape : list, optional
        shape to prepend to the jones matrix array.
        shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices.
        Default returns a 2x2 array

    Returns
    -------
    jones
        pauli spin matrix of index specified
    """

    jones = _empty_jones(shape=shape)

    assert index in (0, 1, 2, 3), f"index should be 0,1,2, or 3. Got {index}"

    if index == 0:
        jones[..., 0, 0] = 1
        jones[..., 1, 1] = 1

    elif index == 1:
        jones[..., 0, 0] = 1
        jones[..., 1, 1] = -1

    elif index == 2:
        jones[..., 0, 1] = 1
        jones[..., 1, 0] = 1

    elif index == 3:
        jones[..., 0, 1] = -1j
        jones[..., 1, 0] = 1j

    return jones


def pauli_coefficients(jones):
    """Compute the pauli coefficients of a jones matrix.

    Parameters
    ----------
    jones : ndarray
        complex jones matrix to decompose


    Returns
    -------
    c0,c1,c2,c3
        complex coefficients of pauli matrices
    """

    c0 = (jones[..., 0, 0] + jones[..., 1, 1]) / 2
    c1 = (jones[..., 0, 0] - jones[..., 1, 1]) / 2
    c2 = (jones[..., 0, 1] + jones[..., 1, 0]) / 2
    c3 = 1j*(jones[..., 0, 1] - jones[..., 1, 0]) / 2

    return c0, c1, c2, c3


def vectorized_polar(matrices):
    """
    Vectorized polar decomposition for N×2×2 array of matrices.
    Here N is an arbitrary number of dimensions
    
    Returns J = U @ P where:
    - U is unitary (retarder)
    - P is Hermitian positive semi-definite (diattenuator)
    
    Parameters:
    -----------
    matrices : ndarray of shape (..., 2, 2)
        Array of 2×2 matrices to decompose
    
    Returns:
    --------
    U : ndarray of shape (..., 2, 2)
        Unitary matrices (retarders)
    P : ndarray of shape (..., 2, 2)
        Hermitian positive semi-definite matrices (diattenuators)
    """
    # Perform SVD on all matrices at once
    # matrices = U_svd @ diag(S) @ Vh
    U_svd, S, Vh = np.linalg.svd(matrices)
    
    # Construct the unitary part: U = U_svd @ Vh
    # This uses batched matrix multiplication
    U = U_svd @ Vh
    
    # Construct the Hermitian positive semi-definite part: P = V @ diag(S) @ Vh
    # where V = conj(Vh.T)
    # First, reconstruct the diagonal matrix from singular values
    S_diag = np.zeros_like(matrices)
    S_diag[..., 0, 0] = S[..., 0]
    S_diag[..., 1, 1] = S[..., 1]
    
    # P = V @ S_diag @ Vh = conj(Vh).T @ S_diag @ Vh
    V = np.conjugate(np.swapaxes(Vh, -2, -1))
    P = V @ S_diag @ Vh
    
    return U, P


def vectorized_logm_2x2(matrices):
    """
    Vectorized matrix logarithm for arrays of 2×2 matrices.
    Uses eigendecomposition: log(A) = V @ log(D) @ V^(-1)
    
    Parameters:
    -----------
    matrices : ndarray of shape (..., 2, 2)
        Array of 2×2 matrices
    
    Returns:
    --------
    logm : ndarray of shape (..., 2, 2)
        Matrix logarithm of each input matrix
    """
    # Compute eigenvalues and eigenvectors for all matrices at once
    eigenvalues, eigenvectors = np.linalg.eig(matrices)
    
    # Take logarithm of eigenvalues
    # Add small epsilon to avoid log(0) issues
    log_eigenvalues = np.log(eigenvalues + 1e-15)
    
    # Create diagonal matrices from log(eigenvalues)
    log_D = np.zeros_like(matrices)
    log_D[..., 0, 0] = log_eigenvalues[..., 0]
    log_D[..., 1, 1] = log_eigenvalues[..., 1]
    
    # Compute V^(-1) for all matrices at once
    eigenvectors_inv = np.linalg.inv(eigenvectors)
    
    # Reconstruct: log(A) = V @ log(D) @ V^(-1)
    logm = eigenvectors @ log_D @ eigenvectors_inv
    
    return logm


def compute_retardance_parameters(jones):
    """
    Eq 5.60 from CLY, computes general retarder and grabs it's Pauli matrix elements
    
    here Exp is the _MATRIX_ Exponential, not the regular exponential

    ED = Exp(-i * (r_H * sigma1 + r_45 * sigma2 + r_C * sigma3)/2)
    """
    U, P = vectorized_polar(jones)
    log_jones = vectorized_logm_2x2(U)
    log_jones *= -2/1j

    return pauli_coefficients(U)


