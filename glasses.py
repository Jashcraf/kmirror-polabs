import numpy as np

# Use Sellmeier Dispersion Formula for Glass Models
S_LAH60 = {
    "A": [1.95243489, 3.07100210e-1, 1.56578094],
    "B": [1.06442437e-2, 4.56735302e-2, 1.10281410e2]
}

S_BAH27 = {
    "A": [1.68939052, 1.33081013e-1, 1.41165515],
    "B": [1.03598193e-2, 5.33982239e-2, 1.26515503e2]
}

def sellmeier(wvl, A, B):
    """Sellmeier glass equation.
    This is from github.com/brandondube/prysm, I just haven't installed
    prysm on this Windows parallel yet

    Parameters
    ----------
    wvl : ndarray
        wavelengths, microns
    A : Iterable
        sequence of "A" coefficients
    B : Iterable
        sequence of "B" coefficients

    Returns
    -------
    ndarray
        refractive index

    """
    wvlsq = wvl ** 2
    seed = np.ones_like(wvl)
    for a, b, in zip(A, B):
        num = a * wvlsq
        den = wvlsq - b
        seed += (num/den)

    return np.sqrt(seed)
