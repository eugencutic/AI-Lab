from sklearn import preprocessing
import numpy as np

# normalizare ( (x-medie)/deviatie )
x_train = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]], dtype=np.float64)
x_test = np.array([[-1, 1, 0]], dtype=np.float64)
# facem statisticile pe datele de antrenare
scaler = preprocessing.StandardScaler()
scaler.fit(x_train)
# afisam media
print(scaler.mean_) # => [1. 0. 0.33333333]
# afisam deviatia standard
print(scaler.scale_) # => [0.81649658 0.81649658 1.24721913]
# scalam datele de antrenare
scaled_x_train = scaler.transform(x_train)
print(scaled_x_train) # => [[0. -1.22474487 1.33630621]
 # [1.22474487 0. -0.26726124]
# [-1.22474487 1.22474487 -1.06904497]]
# scalam datele de test
scaled_x_test = scaler.transform(x_test)
print(scaled_x_test) # => [[-2.44948974 1.22474487 -0.26726124]]

#########
# aducerea datelor in intervalul [0, 1]
x_train = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]], dtype=np.float64)
x_test = np.array([[-1, 1, 0]], dtype=np.float64)
# facem statisticile pe datele de antrenare
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)) # (0, 1) default
min_max_scaler.fit(x_train)
# scalam datele de antrenare
scaled_x_train = min_max_scaler.transform(x_train)
print(scaled_x_train) # => [[0.5 0. 1. ]
 # [1. 0.5 0.33333333]
 # [0. 1. 0. ]]
# scalam datele de test
scaled_x_test = min_max_scaler.transform(x_test)
print(scaled_x_test) # => [[-0.5 1. 0.33333333]]

########
# L1
scaler = preprocessing.Normalizer(norm='l1')
scaler.fit(x_train)
print('-------')
print(scaler.transform(x_train))
print()
print(scaler.transform(x_test))

# L2

scaler = preprocessing.Normalizer(norm='l2')
scaler.fit(x_train)
print('-------')
print(scaler.transform(x_train))
print()
print(scaler.transform(x_test))