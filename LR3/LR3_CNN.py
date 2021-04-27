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

import threading

import numpy as np
import cv2
import sys
from PyQt5 import QtWidgets
import random
from PyQt5.QtGui import QPixmap
import qimage2ndarray
import pickle

import gui_3_CNN


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
                sample_result[k] = weights_sum[i][k] - 2.2  # 0.79
            else:
                sample_result[k] = 0

        result[i] = sample_result
    result = np.array(result)
    # exit(0)
    return np.array(result)


class CNNApp(QtWidgets.QMainWindow, gui_3_CNN.Ui_MainWindow):
    def __init__(self):
        self.DEBUG = True

        super().__init__()
        self.setupUi(self)
        self.btn_load_images.clicked.connect(self.load_images)
        self.btn_filter_1_generate.clicked.connect(self.filter_1_generate)
        self.btn_filter_1_load.clicked.connect(self.filter_1_load)
        self.btn_filter_1_save.clicked.connect(self.filter_1_save)
        self.btn_filter_2_generate.clicked.connect(self.filter_2_generate)
        self.btn_filter_2_load.clicked.connect(self.filter_2_load)
        self.btn_filter_2_save.clicked.connect(self.filter_2_save)
        self.btn_filters_load.clicked.connect(self.filters_load)
        self.btn_filters_save.clicked.connect(self.filters_save)

        self.btn_apply_filters.clicked.connect(self.apply_filters)
        self.btn_preview_filters.clicked.connect(self.preview_filters)

        self.btn_start_training.clicked.connect(self.start_training)
        self.btn_predict.clicked.connect(self.predict_test_image)
        self.btn_save_to_file.clicked.connect(self.save_model_to_file)
        self.btn_load_from_file.clicked.connect(self.load_model_from_file)
        self.test_values = False

        self.synaptic_weights_0 = np.array([])
        self.synaptic_weights_1 = np.array([])

        self.loaded_images = []
        self.loaded_labels = []
        self.cnn_filters_1 = []
        self.cnn_convoluted_1 = []
        self.cnn_filters_2 = []
        self.cnn_convoluted_2 = []

    def load_images(self):
        self.loaded_images = []
        self.loaded_labels = []

        for i in range(4):
            loaded_images_temp = []
            loaded_labels_temp = []
            for k in range(self.spin_images_n.value()):
                loaded_images_temp.append(1.0 - cv2.cvtColor(cv2.imread(
                    str(self.line_folder.text()) + str(i) + '_' + str(k) + '.bmp'), cv2.COLOR_BGR2GRAY) / 255.0)
                loaded_labels_temp.append([i])
            self.loaded_labels.append(loaded_labels_temp)
            self.loaded_images.append(loaded_images_temp)

        self.loaded_images = np.array(self.loaded_images)
        self.loaded_labels = np.array(self.loaded_labels)

        if self.DEBUG:
            print('-------------------- INPUT DATA --------------------')
            print('Shape of loaded_imagess: ' + str(self.loaded_images.shape))
            print('Shape of loaded_labels: ' + str(self.loaded_labels.shape))
            print('Arrays:')
            print(self.loaded_images)
            print()
            print(self.loaded_labels)
            print('----------------------------------------------------')

    def filter_1_generate(self):
        self.cnn_filters_1 = []
        for i in range(4):
            x_temp = []
            for x in range(5):
                y_temp = []
                for y in range(5):
                    y_temp.append(random.randrange(0, 2))  # {0; 1}
                x_temp.append(y_temp)
            self.cnn_filters_1.append(x_temp)
        self.cnn_filters_1 = np.array(self.cnn_filters_1)
        self.filter_1_show()

    def filter_1_load(self):
        with open(self.line_folder.text() + 'cnn_filters_1.dat', 'rb') as filehandle:
            self.cnn_filters_1 = np.array(pickle.load(filehandle))
        self.filter_1_show()

    def filter_1_save(self):
        with open(self.line_folder.text() + 'cnn_filters_1.dat', 'wb') as filehandle:
            pickle.dump(self.cnn_filters_1, filehandle)

    def filter_1_show(self):
        # filter 1
        image_temp = (self.cnn_filters_1[0]) * 255.0
        image_temp = cv2.resize(image_temp.astype(int), (60, 60), interpolation=cv2.INTER_NEAREST)
        self.cvl_filter_1_1.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(image_temp)))

        # filter 2
        image_temp = (self.cnn_filters_1[1]) * 255.0
        image_temp = cv2.resize(image_temp.astype(int), (60, 60), interpolation=cv2.INTER_NEAREST)
        self.cvl_filter_1_2.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(image_temp)))

        # filter 3
        image_temp = (self.cnn_filters_1[2]) * 255.0
        image_temp = cv2.resize(image_temp.astype(int), (60, 60), interpolation=cv2.INTER_NEAREST)
        self.cvl_filter_1_3.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(image_temp)))

        # filter 4
        image_temp = (self.cnn_filters_1[3]) * 255.0
        image_temp = cv2.resize(image_temp.astype(int), (60, 60), interpolation=cv2.INTER_NEAREST)
        self.cvl_filter_1_4.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(image_temp)))

    def filter_2_generate(self):
        self.cnn_filters_2 = []
        for i in range(4):
            x_temp = []
            for x in range(2):
                y_temp = []
                for y in range(2):
                    y_temp.append(random.randrange(0, 2))  # {0; 1}
                x_temp.append(y_temp)
            self.cnn_filters_2.append(x_temp)
        self.cnn_filters_2 = np.array(self.cnn_filters_2)
        self.filter_2_show()

    def filter_2_load(self):
        with open(self.line_folder.text() + 'cnn_filters_2.dat', 'rb') as filehandle:
            self.cnn_filters_2 = np.array(pickle.load(filehandle))
        self.filter_2_show()

    def filter_2_save(self):
        with open(self.line_folder.text() + 'cnn_filters_2.dat', 'wb') as filehandle:
            pickle.dump(self.cnn_filters_2, filehandle)

    def filter_2_show(self):
        # filter 1
        image_temp = (self.cnn_filters_2[0]) * 255.0
        image_temp = cv2.resize(image_temp.astype(int), (60, 60), interpolation=cv2.INTER_NEAREST)
        self.cvl_filter_2_1.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(image_temp)))

        # filter 2
        image_temp = (self.cnn_filters_2[1]) * 255.0
        image_temp = cv2.resize(image_temp.astype(int), (60, 60), interpolation=cv2.INTER_NEAREST)
        self.cvl_filter_2_2.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(image_temp)))

        # filter 3
        image_temp = (self.cnn_filters_2[2]) * 255.0
        image_temp = cv2.resize(image_temp.astype(int), (60, 60), interpolation=cv2.INTER_NEAREST)
        self.cvl_filter_2_3.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(image_temp)))

        # filter 4
        image_temp = (self.cnn_filters_2[3]) * 255.0
        image_temp = cv2.resize(image_temp.astype(int), (60, 60), interpolation=cv2.INTER_NEAREST)
        self.cvl_filter_2_4.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(image_temp)))

    def apply_filters(self):
        # CONVOLUTION 1
        self.cnn_convoluted_1 = []
        for i in range(4):
            image_array = []
            for k in range(self.spin_images_n.value()):
                single_image = []
                for x in range(16):
                    convoluted_row = []
                    for y in range(16):
                        convoluted_pixel = 0
                        for x_k in range(5):
                            for y_k in range(5):
                                convoluted_pixel += self.loaded_images[i][k][x + x_k][y + y_k] \
                                                    * self.cnn_filters_1[i][x_k][y_k]
                        convoluted_row.append(convoluted_pixel)
                    single_image.append(convoluted_row)
                image_array.append(single_image)
            self.cnn_convoluted_1.append(image_array)

        self.cnn_convoluted_1 = np.array(self.cnn_convoluted_1)
        self.cnn_convoluted_1 /= 25.0

        # Normalization
        for i in range(4):
            for k in range(self.spin_images_n.value()):
                self.cnn_convoluted_1[i][k] = self.cnn_convoluted_1[i][k] - self.cnn_convoluted_1[i][k].min()
                if self.cnn_convoluted_1[i][k].max() > 0:
                    self.cnn_convoluted_1[i][k] = self.cnn_convoluted_1[i][k] *\
                                                  (1.0 / self.cnn_convoluted_1[i][k].max())

        # CONVOLUTION 2
        self.cnn_convoluted_2 = []
        for i in range(4):
            image_array = []
            for k in range(self.spin_images_n.value()):
                single_image = []
                x = 0
                while x <= 14:
                    convoluted_row = []
                    y = 0
                    while y <= 14:
                        convoluted_pixel = 0
                        for x_k in range(2):
                            for y_k in range(2):
                                convoluted_pixel += self.cnn_convoluted_1[i][k][x + x_k][y + y_k] \
                                                    * self.cnn_filters_2[i][x_k][y_k]
                        convoluted_row.append(convoluted_pixel)
                        y += 2
                    single_image.append(convoluted_row)
                    x += 2
                image_array.append(single_image)
            self.cnn_convoluted_2.append(image_array)

        self.cnn_convoluted_2 = np.array(self.cnn_convoluted_2)
        self.cnn_convoluted_2 /= 4.0

        # Normalization
        for i in range(4):
            for k in range(self.spin_images_n.value()):
                self.cnn_convoluted_2[i][k] = self.cnn_convoluted_2[i][k] - self.cnn_convoluted_2[i][k].min()
                if self.cnn_convoluted_2[i][k].max() > 0:
                    self.cnn_convoluted_2[i][k] = self.cnn_convoluted_2[i][k] * \
                                                  (1.0 / self.cnn_convoluted_2[i][k].max())
        self.preview_filters()

    def filters_load(self):
        # noinspection PyBroadException
        try:
            print('Loading filters and convolutions from file...')
            self.filter_1_load()
            self.filter_2_load()
            with open(self.line_folder.text() + 'cnn_convolution_1.dat', 'rb') as filehandle:
                self.cnn_convoluted_1 = np.array(pickle.load(filehandle))
            with open(self.line_folder.text() + 'cnn_convolution_2.dat', 'rb') as filehandle:
                self.cnn_convoluted_2 = np.array(pickle.load(filehandle))
            self.preview_filters()
            print('Done.')
        except:
            print(sys.exc_info())

    def filters_save(self):
        print('Saving filters and convolutions to file...')
        self.filter_1_save()
        self.filter_2_save()
        with open(self.line_folder.text() + 'cnn_convolution_1.dat', 'wb') as filehandle:
            pickle.dump(self.cnn_convoluted_1, filehandle)
        with open(self.line_folder.text() + 'cnn_convolution_2.dat', 'wb') as filehandle:
            pickle.dump(self.cnn_convoluted_2, filehandle)
        print('Done.')

    def preview_filters(self):
        src_data = self.loaded_images[self.spin_preview_from.value()][self.spin_preview_n.value()]
        filter_data_1 = self.cnn_filters_1[self.spin_preview_from.value()]
        result_data_1 = self.cnn_convoluted_1[self.spin_preview_from.value()][self.spin_preview_n.value()]
        filter_data_2 = self.cnn_filters_2[self.spin_preview_from.value()]
        result_data_2 = self.cnn_convoluted_2[self.spin_preview_from.value()][self.spin_preview_n.value()]

        src_image = (src_data * 255.0).astype(int)
        filter_image_1 = (filter_data_1 * 255.0).astype(int)
        result_image_1 = (result_data_1 * 255.0).astype(int)
        filter_image_2 = (filter_data_2 * 255.0).astype(int)
        result_image_2 = (result_data_2 * 255.0).astype(int)

        self.ocl_preview_src.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
            cv2.resize(src_image, (120, 120), interpolation=cv2.INTER_NEAREST))))
        self.ocl_preview_filter.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
            cv2.resize(filter_image_1, (30, 30), interpolation=cv2.INTER_NEAREST))))
        self.ocl_preview_result.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
            cv2.resize(result_image_1, (96, 96), interpolation=cv2.INTER_NEAREST))))
        self.ocl_preview_src_2.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
            cv2.resize(result_image_1, (96, 96), interpolation=cv2.INTER_NEAREST))))
        self.ocl_preview_filter_2.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
            cv2.resize(filter_image_2, (12, 12), interpolation=cv2.INTER_NEAREST))))
        self.ocl_preview_result_2.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
            cv2.resize(result_image_2, (48, 48), interpolation=cv2.INTER_NEAREST))))

    def predict_test_image(self):
        # noinspection PyBroadException
        try:
            # self.load_images()
            # self.filters_load()
            # self.load_model_from_file()

            random_index = random.randrange(0, self.spin_images_n.value())
            temp_data = self.cnn_convoluted_2[self.spin_test_array_id.value()][random_index]

            temp_image = (temp_data * 255.0).astype(int)
            self.ocl_test_image.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
                cv2.resize(temp_image, (128, 128), interpolation=cv2.INTER_NEAREST))))

            test_array = np.array([temp_data.flatten()])

            test_labels = np.array([[self.spin_test_array_id.value()]])
            output_l0 = test_array
            output_l1 = dot_0_layer(output_l0, self.synaptic_weights_0)
            output_l2 = sigmoid(np.dot(output_l1, self.synaptic_weights_1.T))
            if np.argmax(output_l2[0]) == test_labels[0][0]:
                self.label_predicted.setText(str(int(np.argmax(output_l2[0])) + 1) + ' YEAH')
            else:
                self.label_predicted.setText(str(int(np.argmax(output_l2[0])) + 1) + ' NOPE')
            self.progressBar_2.setValue(output_l2[0][0] * 100)
            self.progressBar_3.setValue(output_l2[0][1] * 100)
            self.progressBar_4.setValue(output_l2[0][2] * 100)
            self.progressBar_5.setValue(output_l2[0][3] * 100)

            # Preview CNN
            cnn_preview_image = 255 * np.ones((512, 512, 3), dtype=np.uint8)
            y = int((512 - 256) / 2)
            for i in range(64):
                color = int((1 - test_array[0][i]) * 255)
                cv2.circle(cnn_preview_image, (10, y + 4 * i), 2, (color, color, color), -1)

            y = int((512 - 256) / 2)
            for i in range(512):
                for k in range(64):
                    calculated = self.synaptic_weights_0[i][k] * output_l0[0][k]
                    below_zero = False if calculated > 0 else True
                    saturation = calculated * 255 if not below_zero else 255
                    value = calculated * -255 if below_zero else 255

                    color_hsv = np.uint8([[[k * 2, saturation, value]]])
                    color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0][0]
                    if calculated != 0:
                        cv2.line(cnn_preview_image, (254, i), (11, y + 4 * k),
                                 (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])))

            y = int((512 - 80) / 2)
            for i in range(512):
                for k in range(4):
                    calculated = self.synaptic_weights_1[k][i] * output_l1[0][i]
                    below_zero = False if calculated > 0 else True
                    saturation = calculated * 127 if not below_zero else 255
                    value = calculated * -127 if below_zero else 255

                    color_hsv = np.uint8([[[k * 40, saturation, value]]])
                    color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0][0]
                    if calculated != 0:
                        cv2.line(cnn_preview_image, (256, i), (500, y + 20 * k),
                                 (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])))

            y = int((512 - 80) / 2)
            for i in range(4):
                saturation = output_l2[0][i] * 255
                color_hsv = np.uint8([[[i * 40, saturation, 255]]])
                color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0][0]

                cv2.circle(cnn_preview_image, (500, y + 20 * i + 1), 5,
                           (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])), -1)

            for i in range(512):
                calculated = output_l1[0][i]
                below_zero = False if calculated > 0 else True
                saturation = calculated * 255 if not below_zero else 255
                value = calculated * -255 if below_zero else 255
                color_hsv = np.uint8([[[255, saturation, value]]])
                color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0][0]
                if calculated != 0:
                    cv2.circle(cnn_preview_image, (255, i), 4,
                               (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])), -1)

            self.ocl_preview_cnn.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
                cv2.resize(cnn_preview_image, (512, 512), interpolation=cv2.INTER_NEAREST))))

        except:
            print(sys.exc_info())

    def save_model_to_file(self):
        compressed_data = [self.synaptic_weights_0, self.synaptic_weights_1]
        with open(self.line_folder.text() + 'model.dat', 'wb') as filehandle:
            pickle.dump(compressed_data, filehandle)

    def load_model_from_file(self):
        with open(self.line_folder.text() + 'model.dat', 'rb') as filehandle:
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

    def start_training(self):
        if len(self.synaptic_weights_0) == 0 or len(self.synaptic_weights_1) == 0:
            # Synaptic weights arrays
            self.synaptic_weights_0 = []
            for i in range(512):  # 4096
                string_array = [int(random.randrange(-1, 2)) for _ in range(3)] + [0 for _ in range(61)]
                random.shuffle(string_array)
                self.synaptic_weights_0.append(string_array)
            self.synaptic_weights_0 = np.array(self.synaptic_weights_0)
            self.synaptic_weights_1 = np.array(2 * np.random.random((4, 512)) - 1)  # 4096
            if self.DEBUG:
                print('-------------------- WEIGHTS --------------------')
                print('Shape of synaptic_weights_0: ' + str(self.synaptic_weights_0.shape))
                print('Shape of synaptic_weights_1: ' + str(self.synaptic_weights_1.shape))
                print('Arrays:')
                print(self.synaptic_weights_0)
                print()
                print(self.synaptic_weights_1)
                print('-------------------------------------------------')

        thread = threading.Thread(target=self.training)
        thread.start()

    def training(self):
        self.cnn_convoluted_2 = np.array(self.cnn_convoluted_2)
        self.loaded_labels = np.array(self.loaded_labels)
        train_data = []
        train_labels = []
        for i in range(4):
            for k in range(self.spin_images_n.value()):
                train_data.append(self.cnn_convoluted_2[i][k].flatten())
        for i in range(4):
            for k in range(self.spin_images_n.value()):
                train_labels.append(self.loaded_labels[i][k][0])
        train_data = np.array(train_data)
        train_labels = np.array([train_labels])

        if self.DEBUG:
            print('-------------------- TRAIN DATA --------------------')
            print('Shape of train_data: ' + str(train_data.shape))
            print('Shape of train_labels: ' + str(train_labels.shape))
            print('Arrays:')
            print(train_data)
            print()
            print(train_labels)
            print('----------------------------------------------------')

        # noinspection PyBroadException
        try:
            i = 0
            while i < int(self.spin_iterations.value()):
                output_l0 = train_data
                output_l1 = dot_0_layer(output_l0, self.synaptic_weights_0)
                output_l2 = sigmoid(np.dot(output_l1, self.synaptic_weights_1.T))

                # Layer 2 error calculations
                error_l2 = []
                for k in range(len(output_l2)):
                    a = []
                    for m in range(4):
                        if m == train_labels[0][k]:
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
                    if np.argmax(output_l2[k]) == train_labels[0][k]:
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

    window = CNNApp()
    window.show()
    app.exec_()
