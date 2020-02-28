from skimage import io # pentru afisarea imaginii
import numpy as np
import matplotlib.pyplot as plt


train_images = np.loadtxt('train_images.txt') # incarcam imaginile
train_labels = np.loadtxt('train_labels.txt', 'int') # incarcam etichetele avand tipul de date int
test_images = np.loadtxt('test_images.txt')
test_labels = np.loadtxt('test_labels.txt', 'int')

class KnnClassifier:
    def __init__(self, train_images, train_labels):
        #putem considera ca am facut fit
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors = 3, metric ='l2'):
        distante = np.zeros((self.train_images.shape[0]))
        if metric == 'l1':
            for i in range(self.train_images.shape[0]):
                distante[i] = np.sum(self.train_images[i, :] - test_image)
        else:
            for i in range(self.train_images.shape[0]):
                distante[i] = np.sqrt(np.sum((self.train_images[i, :] - test_image) ** 2))
        # determinam cei mai apropiati num_neighbors vecini si eticheta lor
        # eticheta cu nr maxim de aparitii da eticheta lui test image
        indici_sortare = np.argsort(distante)
        etichete = self.train_labels[indici_sortare[0 : num_neighbors]]
        aparitii = np.bincount(etichete)  # da un vector de frecventa de la 0 la cel mai mare nr din array
        return np.argmax(aparitii)

    def evaluate(self, pred_labels, real_labels):
        aux = 0
        for i in range(len(pred_labels)):
            if pred_labels[i] == real_labels[i]:
                aux += 1
        return aux / len(real_labels)

    def predict(self, test_images, num_neighbors = 3, metric = 'l2'):
        predicted_labels = np.zeros(test_images.shape[0])
        for i in range(test_images.shape[0]):
            predicted_labels[i] = self.classify_image(test_images[i, :], num_neighbors, metric)
        return predicted_labels
    def confusion(self, pred_labels, real_labels):
        #formati o matrice de confuzie
        confusion_matrix = np.zeros((np.unique(real_labels).shape[0], (np.unique(real_labels).shape[0])))
        #TODO: popularea matricei

        return confusion_matrix



# ob = KnnClassifier(train_images, train_labels)
# predicted = ob.predict(test_images)
# print(ob.evaluate(predicted, test_labels))
# ##############
# #grid search
# num_neighbors = [1, 3, 5, 7, 9]
# acuratete = []
# for vecin_val in num_neighbors:
#     etichete = ob.predict(test_images, vecin_val)
#     acuratete.append(ob.evaluate(etichete, test_labels))
# #plotam acuratetea pt fiecare nr de vecini
# plt.plot(acuratete)
# plt.show()

print(train_images.shape[0])
print(train_images.shape[1])



