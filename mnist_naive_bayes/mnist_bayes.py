from skimage import io # pentru afisarea imaginii
import numpy as np
import matplotlib.pyplot as plt


train_images = np.loadtxt('train_images.txt') # incarcam imaginile
train_labels = np.loadtxt('train_labels.txt', 'int') # incarcam etichetele avand tipul de date int
test_images = np.loadtxt('test_images.txt')
test_labels = np.loadtxt('test_labels.txt', 'int')
#
# image = train_images[11, :]
# image = np.reshape(image, (28, 28))
# #--------
# num_bins = 5
# x = image
# #--------
# bins = np.linspace(start=0, stop=255, num=num_bins) # returneaza intervalele
# x_to_bins = np.digitize(x, bins) # returneaza pentru fiecare element intervalul
#                                  # corespunzator
#                                  # Atentie! In cazul nostru indexarea elementelor va
#                                  # incepe de la 1, intrucat nu avem valori < 0
# print(x_to_bins)
#
# # for i in range(16):
# #     image = train_images[i, :]
# #     image = np.reshape(image, (28, 28))
# #     plt.subplot(4, 4, i + 1);
# #     io.imshow(image.astype(np.uint8))
# # io.show()
# #
# # etichete = []
# # for i in range(16):
# #     etichete.append(train_labels[i])
# # for i in range(16):
# #     if i % 4 == 0:
# #         print()
# #     print(etichete[i])

class Naive_Bayes:
    def __init__(self, num_bins, max_values):
        self.bins = np.linspace(start=0, stop=255, num=num_bins)
        self.num_bins = num_bins

    def values_to_bins(self, matrice_imagini):
        return np.digitize(matrice_imagini, self.bins) - 1

    def fit(self, train_images, train_labels):
        train_images = self.values_to_bins(train_images) # aplicam histograma
        # calculam p(c) so p(x|c)
        # dimenisunea lui p(c) este 1 x num_classes
        # dimensiunea lui p(x|c) este num_features x num_samples x num_classes
        pc = np.zeros((np.unique(train_labels).shape[0])) # initializam cu val 0 si dim num_classes
        for class_val in np.unique(train_labels):
            pc[class_val] = sum(train_labels == class_val) / train_labels.shape[0]
        # p(x|c#)
        pxc = np.zeros((train_images.shape[1], self.num_bins, np.unique(train_labels).shape[0]))
        for i in range(train_images.shape[1]):
            for class_val in np.unique(train_labels):
                imgs_in_class_val = train_images[train_labels == class_val, :]
                # for k in range(16):
                #     plt.subplot(4, 4, k + 1)
                #     image = imgs_in_class_val[k, :]
                #     image = np.reshape(image, (28, 28))
                #     io.imshow(image.astype(np.uint8))
                # io.show()
                for j in range(self.num_bins):
                    numar_bins_pe_feature = sum(imgs_in_class_val[:, i] == j)

                    pxc[i, j, class_val] = numar_bins_pe_feature / imgs_in_class_val.shape[0]

        self.pc = pc
        self.pxc = pxc + 1e-10
        return pc, pxc

    def predict(self, test_images):
        # dimensiunea lui p(x|c) este num_features x num_samples x num_classes
        prediction_classes = np.zeros((test_images.shape[0], len(np.unique(train_labels))))
        test_images = self.values_to_bins(test_images)
        for img_idx in range(test_images.shape[0]):
            for class_val in np.unique(train_labels):
                prb = np.log(self.pc[class_val])
                for i in range(train_images.shape[1]):
                    prb += np.log(self.pxc[i, test_images[img_idx, i], class_val])
                    prediction_classes[img_idx, class_val] = prb

        predicted_labels = np.zeros((test_images.shape[0]))
        for img_idx in range(prediction_classes.shape[0]):
            predicted_labels[img_idx] = np.argmax(prediction_classes[img_idx, :])
        return predicted_labels

    def evaluate(self, pred_labels, real_labels):
        aux = 0
        for i in range(len(pred_labels)):
            if pred_labels[i] == real_labels[i]:
                aux += 1
        return aux / len(real_labels)
        #alta metoda:
        # pred_labels = np.array(pred_labels)
        # real_labels = np.array(real_labels)
        # return sum(real_labels == pred_labels) / pred_labels.shape[0]


ob = Naive_Bayes(5, 255)
pc, pxc = ob.fit(train_images, train_labels)
predicted_labels = ob.predict(test_images)
print(ob.evaluate(predicted_labels, test_labels))


