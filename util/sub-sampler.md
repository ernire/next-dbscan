# Use AWK to select and print the n-th line of input dataset
# Used for simple sub-sampling
# n == sampling ratio divisor

awk 'NR % n == 0' input > output
