#An ugly number is a positive integer whose prime factors are limited to 2, 3, and 5.
#Given an integer n, return true if n is an ugly number.

def isUgly( n: int) -> bool:
    if n == 1:
        return True
    if n == 0:
        return False
        uglyPrimeSet = [2, 3, 5]

        for f in uglyPrimeSet:
            while (n % f) == 0:
                n = n / f
        if n == 1:
            return True
        else:
            return False
print(isUgly(905391974))