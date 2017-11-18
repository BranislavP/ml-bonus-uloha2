import numpy as np
from functools import reduce


data = np.genfromtxt('data.csv', skip_header=1, usecols=(1, 2, 4, 7, 8), delimiter=',',
                     converters={1: lambda s: float(reduce((lambda x, y: int(x) * 60 + int(y)), s.decode().split(' ')[1]
                                                           .split(':'))),
                                 2: lambda s: float(reduce((lambda x, y: int(x) * 60 + int(y)), s.decode().split(' ')[1]
                                                           .split(':')))})

# only for airport
selected = data[:, 4] == 1
y = data[selected, 0]
X = data[selected, 1:]
X = np.insert(X, 0, 1, axis=1)

print(y.shape)
print(y)
print(X.shape)
print(X)
theta = np.linalg.pinv(X.T @ X) @ X.T @ y
print(theta.shape)
print(theta)
predict = np.array([1.0, 43200.0, 15.0, 87.0, 1.0])
result = predict @ theta
print(result.shape)
print(result)
hours = int(result/3600)
minutes = int((result - hours * 3600)/60)
seconds = int(((result - hours * 3600) - minutes * 60))
print(str(hours) + ':' + str(minutes) + ':' + str(seconds))

# for any location --- may need additional data
#y = data[:, 0]
#X = data[:, 1:]
#X = np.insert(X, 0, 1, axis=1)

#theta = np.linalg.pinv(X.T @ X) @ X.T @ y
#predict = np.array([1.0, 43200.0, 15.0, 87.0, 1.0])
#result = predict @ theta
#hours = int(result/3600)
#minutes = int((result - hours * 3600)/60)
#seconds = int(((result - hours * 3600) - minutes * 60)/60)
#print('%d:%d:%d', hours, minutes, seconds)

