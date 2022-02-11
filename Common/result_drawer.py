import matplotlib.pyplot as plt
import numpy as np


class ResultDrawer:
    def __init__(self, row=6, col=4):
        self.num_rows = row
        self.num_cols = col
        self.class_names = ['male', 'female']

    def plot_image(self, i, predictions_array, true_label, img):
        true_label, img = true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap='gray')

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(self.class_names[predicted_label],
                                             100*np.max(predictions_array),
                                             self.class_names[true_label]),
                   color=color)

    def plot_value_array(self, i, predictions_array, true_label):
        true_label = true_label[i]
        plt.grid(False)
        plt.xticks(range(2))
        plt.yticks([])
        thisplot = plt.bar(range(2), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    def plot(self, model, test_images, test_labels):
        predictions = model.predict(test_images)
        num_images = self.num_rows * self.num_cols
        plt.figure(figsize=(2 * 2 * self.num_cols, 2 * self.num_rows))

        for i in range(num_images):
            plt.subplot(self.num_rows, 2 * self.num_cols, 2 * i + 1)
            self.plot_image(i, predictions[i], test_labels, test_images)
            plt.subplot(self.num_rows, 2 * self.num_cols, 2 * i + 2)
            self.plot_value_array(i, predictions[i], test_labels)

        plt.tight_layout()
        plt.show()
