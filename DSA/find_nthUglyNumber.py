# An ugly number is a positive integer whose prime factors are limited to 2, 3, and 5.
#  
# Given an integer n, return the nth ugly number.

import math
n =  15
ugly = [0] * n  # To store ugly numbers

# 1 is the first ugly number
ugly[0] = 1

# i2, i3, i5 will indicate indices for
# 2,3,5 respectively
i2 = i3 = i5 = 0

# Set initial multiple value
next_multiple_of_2 = 2
next_multiple_of_3 = 3
next_multiple_of_5 = 5

# Start loop to find value from
# ugly[1] to ugly[n]
for l in range(1, n):

    # Choose the min value of all
    # available multiples
    ugly[l] = min(next_multiple_of_2,
                  next_multiple_of_3,
                  next_multiple_of_5)

    # Increment the value of index accordingly
    if ugly[l] == next_multiple_of_2:
        i2 += 1
        next_multiple_of_2 = ugly[i2] * 2
        print ( "2s: " ,i2, ":",next_multiple_of_2 )

    if ugly[l] == next_multiple_of_3:
        i3 += 1
        next_multiple_of_3 = ugly[i3] * 3
        print("3s: ", i3, ":", next_multiple_of_3)

    if ugly[l] == next_multiple_of_5:
        i5 += 1
        next_multiple_of_5 = ugly[i5] * 5
        print("5s: ", i5, ":", next_multiple_of_5)

# Return ugly[n] value
print  (ugly[-1])