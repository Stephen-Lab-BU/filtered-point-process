import numpy as np
import warnings


def nextpow2(n):
    """Compute the next highest power of 2 of 32-bit integer n.

    :param n:

    """
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return int(2**m_i)
