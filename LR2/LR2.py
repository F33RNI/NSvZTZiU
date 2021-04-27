"""
 Licensed under the Unlicense License;
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://unlicense.org

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import csv
import threading

import numpy as np
import cv2
import sys
from PyQt5 import QtWidgets
import random
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QPalette, QColor
import qimage2ndarray
import pickle
import json

import gui_2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dot_0_layer(input_layer, synaptic_weights):
    return layer_0_activator(np.dot(input_layer, synaptic_weights.T))


def valmap(value, istart, istop, ostart, ostop):
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))


def layer_0_activator(weights_sum):
    # THE ASS IS COMING
    # return max(0, weights_sum)
    result = [[0] * weights_sum[0]] * weights_sum
    for i in range(len(weights_sum)):
        sample_result = [0] * weights_sum[0]
        for k in range(len(weights_sum[i])):

            threshold = 1.8  # 1.79
            if weights_sum[i][k] >= threshold:
                sample_result[k] = weights_sum[i][k] - 2.2  # 0.79 #  weights_sum[i][k] - 1.6 # weights_sum[i][k] - 2.1
            else:
                sample_result[k] = 0

            # sample_result[k] = max(2, weights_sum[i][k] - 2)
        result[i] = sample_result
    result = np.array(result)
    # print(result)
    # exit(0)
    return np.array(result)


class CollectorApp(QtWidgets.QMainWindow, gui_2_2.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btn_load_train_array.clicked.connect(self.load_train_array)
        self.btn_start_training.clicked.connect(self.start_training)
        self.btn_load_test_array.clicked.connect(self.load_test_array)
        self.btn_predict.clicked.connect(self.predict_test_image)
        self.btn_save_to_file.clicked.connect(self.save_to_file)
        self.btn_load_from_file.clicked.connect(self.load_from_file)
        self.test_values = False

        self.synaptic_weights_0 = np.array([])
        self.synaptic_weights_1 = np.array([])
        self.train_images = np.array([])
        self.train_labels = np.array([])
        self.test_array = np.array([])
        self.test_labels = np.array([])
        self.test_image = None

    def load_train_data(self, debug, folder):
        train_images = []
        train_labels = []
        for i in range(4):
            train_data_temp = cv2.cvtColor(cv2.imread(folder + str(i) + '.bmp'), cv2.COLOR_BGR2GRAY) / 255.0
            for x in range(32):  # 32
                for y in range(32):  # 32
                    temp_data = train_data_temp[y: y + 16, x: x + 16].flatten()
                    train_images.append(temp_data)
                    train_labels.append(i)

        combined_lists = list(zip(train_images, train_labels))
        random.shuffle(combined_lists)
        train_images, train_labels = zip(*combined_lists)
        self.train_images = np.array(train_images)
        self.train_labels = np.array([train_labels])

        # TEST VALUES
        if self.test_values:
            self.train_images = []
            self.train_labels = []
            for _ in range(50):
                a = []
                b = []
                for _ in range(8):
                    a.append(random.randrange(0, 10) / 10)
                b.append(a[0] + a[1])
                b.append(a[2] + a[3])
                b.append(a[4] + a[5])
                b.append(a[6] + a[7])
                self.train_labels.append(np.argmax(b))
                self.train_images.append(a)
            self.train_images = np.array(self.train_images)
            self.train_labels = np.array([self.train_labels])

        if debug:
            print('-------------------- TRAIN DATA --------------------')
            print('Shape of train_images: ' + str(self.train_images.shape))
            print('Shape of train_labels: ' + str(self.train_labels.shape))
            print('Arrays:')
            print(self.train_images)
            print()
            print(self.train_labels)
            print('----------------------------------------------------')

    def load_test_array(self):
        self.test_image = cv2.cvtColor(cv2.imread(self.line_folder.text() + 'test/' +
                                                  str(self.spin_test_array_id.value() - 1) + '.bmp'),
                                       cv2.COLOR_BGR2GRAY)
        self.ocl_test_array.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
            cv2.cvtColor(cv2.resize(self.test_image, (256, 256)), cv2.COLOR_GRAY2RGB))))

    def predict_test_image(self):
        # noinspection PyBroadException
        try:
            x = random.randrange(0, 32)
            y = random.randrange(0, 32)
            temp_image = self.test_image[y * 16: y * 16 + 16, x * 16: x * 16 + 16]
            self.ocl_test_image.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
                cv2.cvtColor(cv2.resize(temp_image, (128, 128), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2RGB))))
            self.test_array = np.array([(temp_image / 255.0).flatten()])

            self.test_labels = np.array([[self.spin_test_array_id.value() - 1]])
            output_l0 = self.test_array
            output_l1 = dot_0_layer(output_l0, self.synaptic_weights_0)
            output_l2 = sigmoid(np.dot(output_l1, self.synaptic_weights_1.T))
            if np.argmax(output_l2[0]) == self.test_labels[0][0]:
                self.label_predicted.setText(str(int(np.argmax(output_l2[0]) + 1)) + ' YEAH')
            else:
                self.label_predicted.setText(str(int(np.argmax(output_l2[0]) + 1)) + ' NOPE')
            self.progressBar_2.setValue(output_l2[0][0] * 100)
            self.progressBar_3.setValue(output_l2[0][1] * 100)
            self.progressBar_4.setValue(output_l2[0][2] * 100)
            self.progressBar_5.setValue(output_l2[0][3] * 100)
        except:
            print(sys.exc_info())

    def save_to_file(self):
        compressed_data = [self.synaptic_weights_0, self.synaptic_weights_1]
        with open(self.line_folder.text() + 'model.txt', 'wb') as filehandle:
            pickle.dump(compressed_data, filehandle)

    def load_from_file(self):
        with open(self.line_folder.text() + 'model.txt', 'rb') as filehandle:
            compressed_data = pickle.load(filehandle)
            self.synaptic_weights_0 = np.array(compressed_data[0])
            self.synaptic_weights_1 = np.array(compressed_data[1])

        print('-------------------- WEIGHTS --------------------')
        print('Shape of synaptic_weights_0: ' + str(self.synaptic_weights_0.shape))
        print('Shape of synaptic_weights_1: ' + str(self.synaptic_weights_1.shape))
        print('Arrays:')
        print(self.synaptic_weights_0)
        print()
        print(self.synaptic_weights_1)
        print('-------------------------------------------------')

    def load_train_array(self):
        random.seed = 1
        np.random.seed(1)
        self.load_train_data(True, self.line_folder.text())

        # Synaptic weights arrays
        self.synaptic_weights_0 = []
        for i in range(512):
            string_array = [int(random.randrange(-1, 2)) for _ in range(3)] + [0 for _ in range(253)]
            random.shuffle(string_array)
            self.synaptic_weights_0.append(string_array)
        self.synaptic_weights_0 = np.array(self.synaptic_weights_0)
        self.synaptic_weights_1 = np.array(2 * np.random.random((4, 512)) - 1)

        # TEST VALUES
        if self.test_values:
            self.synaptic_weights_0 = []
            for i in range(16):
                string_array = [int(random.randrange(-1, 2)) for _ in range(3)] + [0 for _ in range(5)]
                random.shuffle(string_array)
                self.synaptic_weights_0.append(string_array)
            self.synaptic_weights_0 = np.array(self.synaptic_weights_0)
            self.synaptic_weights_1 = np.array(2 * np.random.random((4, 16)) - 1)
        if True:
            print('-------------------- WEIGHTS --------------------')
            print('Shape of synaptic_weights_0: ' + str(self.synaptic_weights_0.shape))
            print('Shape of synaptic_weights_1: ' + str(self.synaptic_weights_1.shape))
            print('Arrays:')
            print(self.synaptic_weights_0)
            print()
            print(self.synaptic_weights_1)
            print('-------------------------------------------------')

    def start_training(self):
        thread = threading.Thread(target=self.training)
        thread.start()

    def training(self):
        # noinspection PyBroadException
        try:
            i = 0
            while i < int(self.spin_iterations.value()):
                output_l0 = self.train_images

                output_l1 = dot_0_layer(output_l0, self.synaptic_weights_0)
                output_l2 = sigmoid(np.dot(output_l1, self.synaptic_weights_1.T))

                # Layer 2 error calculations
                error_l2 = []
                for k in range(len(output_l2)):
                    a = []
                    for m in range(4):
                        if m == self.train_labels[0][k]:
                            a.append(1 - output_l2[k][m])
                        else:
                            a.append(0 - output_l2[k][m])
                    error_l2.append(a)
                error_l2 = np.array(error_l2)

                adjustments_l2 = output_l1.T.dot(error_l2 * (output_l2 * (1 - output_l2)))
                self.synaptic_weights_1 += adjustments_l2.T

                # Occuracy calculations
                predicted = []
                occuracy = 0
                for k in range(len(output_l2)):
                    predicted.append(np.argmax(output_l2[k]))
                    if np.argmax(output_l2[k]) == self.train_labels[0][k]:
                        occuracy += 1
                occuracy /= len(output_l2)
                predicted = np.array(predicted)

                if i % 1 == 0:
                    print('-------------------- I: ' + str(i) + ' --------------------')
                    # print('output_l2: ' + str(output_l2))
                    # print('error_l2: ' + str(error_l2))
                    print('predicted: ' + str(predicted))
                    print('occuracy: ' + str(occuracy))
                    # print('adjustments_l2: ' + str(adjustments_l2))
                    # print('----------------------------------------------')
                i += 1
                self.progressBar.setValue(valmap(i, 0, self.spin_iterations.value(), 0, 100))

            self.progressBar.setValue(0)

        except:
            print(sys.exc_info())


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    app.setStyle("Fusion")

    """
    dark_palette = QPalette()
    WHITE = QColor(255, 255, 255)
    BLACK = QColor(0, 0, 0)
    RED = QColor(255, 0, 0)
    PRIMARY = QColor(53, 53, 53)
    SECONDARY = QColor(25, 25, 25)
    LIGHT_PRIMARY = QColor(100, 100, 100)
    TERTIARY = QColor(42, 130, 218)
    dark_palette.setColor(QPalette.Window, PRIMARY)
    dark_palette.setColor(QPalette.WindowText, WHITE)
    dark_palette.setColor(QPalette.Base, SECONDARY)
    dark_palette.setColor(QPalette.AlternateBase, PRIMARY)
    dark_palette.setColor(QPalette.ToolTipBase, WHITE)
    dark_palette.setColor(QPalette.ToolTipText, WHITE)
    dark_palette.setColor(QPalette.Text, WHITE)
    dark_palette.setColor(QPalette.Button, LIGHT_PRIMARY)
    dark_palette.setColor(QPalette.ButtonText, WHITE)
    dark_palette.setColor(QPalette.BrightText, RED)
    dark_palette.setColor(QPalette.Link, TERTIARY)
    dark_palette.setColor(QPalette.Highlight, TERTIARY)
    dark_palette.setColor(QPalette.HighlightedText, BLACK)

    app.setPalette(dark_palette)

    app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")
    """

    window = CollectorApp()
    window.show()
    app.exec_()
