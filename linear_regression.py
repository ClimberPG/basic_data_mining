import csv
import time
import numpy as np

start = time.time()

with open('save_train.csv', 'rb') as f:
    my_matrix = np.loadtxt(f, delimiter=',', skiprows=1)

N = 385
M = 25000
COUNT = 100000

theta = np.zeros(N + 1)
theta[N] = -1

my_matrix[:, 0] = 1
my_matrix_t = my_matrix.T[:N]

alpha = 0.0993

for t in range(COUNT):
    value = my_matrix.dot(theta)
    theta[:N] -= my_matrix_t.dot(value) / M * alpha
    print(t,value.dot(value) / (2 * M))

end = time.time()
print(end - start)

matrix_test = np.loadtxt(open('save_test.csv', 'rb'), delimiter=',', skiprows=1)

matrix_test[:, 0] = 1
theta = np.delete(theta, N, axis=0)

answer = matrix_test.dot(theta)

csvfile = file('csv_test.csv', 'wb')
writer = csv.writer(csvfile)
writer.writerow(['Id', 'reference'])

for my_id in range(M):
	writer.writerow([my_id, answer[my_id]])

csvfile.close()
