import numpy as np
import os
from collections import Counter

# Fetch the individual position vectors from the log file
os.system("grep 'indiv_position' log_20170813_001048.txt | sed -r 's/.{42}//' | sed 's/[\[]//g'| sed 's/\]//g' > positions.txt")

# Read into an np array
with open('positions.txt') as file:
	array2d = [[float(digit) for digit in line.split(',')] for line in file]
array2d = np.array(array2d)

# Read the first 50 rows
array2d_1 = array2d[0:50,:]

# Count the frequency of unique values in each column over each iteration
start=0
end=50
for i in range(200):
    if i%50==0:
        array2d_1 = array2d[start:end,:]
        for j in range(array2d_1.shape[1]):
            count = Counter(array2d_1[:, j])
            print 'Frequency of column {0} over iteration {1}'.format(j,i+1)
            print count.most_common(3)
        print '========================'
    start += 50
    end += 50
    
print''

# Count the frequency of unique values in each column over all iterations together
print 'Frequency over all iterations'
for i in range(array2d.shape[1]):
    count = Counter(array2d[:, i])
    print count.most_common(3)


