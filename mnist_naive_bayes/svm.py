from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# valori mari ale lui C determina margine mica => nu permite misclasare ~ posibil suprainvatare
# valori mici ale lui C => margine mare ~ posibil subinvatare

train_images = np.loadtxt('train_images.txt') # incarcam imaginile
train_labels = np.loadtxt('train_labels.txt', 'int') # incarcam etichetele avand tipul de date int
test_images = np.loadtxt('test_images.txt')
test_labels = np.loadtxt('test_labels.txt', 'int')


##############
def normalize(norm_type, train_data, test_data):
    if norm_type == 'standard':
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        return train_data, test_data
    elif norm_type == 'minmax':
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        return train_data, test_data
    elif norm_type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        return train_data, test_data
    elif norm_type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        return train_data, test_data
    else:
        print('Not a normalization method')
        return


def evaluate(pred_labels, real_labels):
    aux = 0
    for i in range(len(pred_labels)):
        if pred_labels[i] == real_labels[i]:
            aux += 1
    return aux / len(real_labels)
##############


norm_type = ['standard', 'minmax', 'l1', 'l2']
Cs = [0.1, 0.3, 0.5, 0.7, 1]
kerns = ['rbf', 'linear']
accuracies_train = np.zeros((len(norm_type), len(Cs), len(kerns)))
accuracies_test = np.zeros((len(norm_type), len(Cs), len(kerns)))

train_images_copy, test_images_copy = train_images.copy(), test_images.copy()

for norm_idx, norm_val in enumerate(norm_type):
    for Cs_idx, Cs_val in enumerate(Cs):
        for kern_idx, kern_val in enumerate(kerns):
            train_images, test_images = normalize(norm_val, train_images_copy, test_images_copy)

            # fac un obiect de tip SVC
            clasificator = SVC(C=Cs_val, kernel=kern_val, degree=3, gamma='auto_deprecated',
                               coef0=0.0, shrinking=True, probability=False,
                               tol=1e-3, cache_size=200, class_weight=None,
                               verbose=False, max_iter=-1, decision_function_shape='ovr',
                               random_state=None)
            clasificator.fit(train_images, train_labels)
            etichete_pred_train = clasificator.predict(train_images)
            etichete_pred_test = clasificator.predict(test_images)

            accuracies_test[norm_idx, Cs_idx, kern_idx] = evaluate(etichete_pred_test, test_labels)
            accuracies_train[norm_idx, Cs_idx, kern_idx] = evaluate(etichete_pred_train, train_labels)


for norm_idx, norm_val in enumerate(norm_type):
    for Cs_idx, Cs_val in enumerate(Cs):
        for kern_idx, kern_val in enumerate(kerns):
            print('norm_val, Cs_val, kern_val')
            print(norm_val, Cs_val, kern_val)
            print(accuracies_train[norm_idx, Cs_idx, kern_idx])
            print(accuracies_test[norm_idx, Cs_idx, kern_idx])
