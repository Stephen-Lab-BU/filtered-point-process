import math

def nextpow2(N):
    """
    The nextpow2 function calculates the next power of two greater than or equal to a given integer.

    The function takes two arguments:
    N: An integer representing the number to be rounded up to the next power of two.
    This is useful if the caller needs the result to be larger than the next power of two by a certain amount.

    Reference:
        This function is adapted by the method described on GeeksforGeeks: 
        https://www.geeksforgeeks.org/smallest-power-of-2-greater-than-or-equal-to-n/ (Updated Feb 13th, 2023; Accessed May 26th, 2024)
    """

    a = int(math.log2(N))
 
    # If 2^a is equal to N, return N
    if 2**a == N:
        return N
     
    # Return 2^(a + 1)
    return 2**(a + 1)


