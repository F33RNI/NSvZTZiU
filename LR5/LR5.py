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

import os
import threading

import numpy as np
import cv2
import sys
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
import qimage2ndarray
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import random
from tkinter import filedialog
import tensorflow as tf
import keras

import gui_5

INPUT_SHAPE = (64, 64, 3)
PREDICT_SHAPE = (-1, 64, 64, 3)
KERNEL_ACTIVATION = 'relu'
MLP_ACTIVATION = 'relu'
OPTIMIZER = 'adam'


class LR5(QtWidgets.QMainWindow, gui_5.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btn_train_browse.clicked.connect(self.train_browse)
        self.btn_train_load.clicked.connect(self.train_load)
        self.btn_test_browse.clicked.connect(self.test_browse)
        self.btn_test_load.clicked.connect(self.test_load)
        self.btn_tfn_train.clicked.connect(self.tfn_train)
        self.btn_tfn_save.clicked.connect(self.tfn_save)
        self.btn_tfn_load.clicked.connect(self.tfn_load)
        self.btn_tfn_predict.clicked.connect(self.tfn_predict)
        self.btn_tfn_predict_camera.clicked.connect(self.predict_camera)
        self.btn_tfn_predict_camera_stop.clicked.connect(self.predict_camera_stop)

        self.train_data = None
        self.test_data = None
        self.model = None

        self.camera_running = False
        self.cv_cap = None

    def train_browse(self):
        file_path = filedialog.askdirectory()
        if file_path is not None:
            self.line_train_folder.setText(file_path)

    def train_load(self):
        datagen = ImageDataGenerator(rescale=1 / 255.0)
        self.train_data = datagen.flow_from_directory(
            self.line_train_folder.text(),
            target_size=INPUT_SHAPE[:2],
            batch_size=self.spin_train_n.value(),
            class_mode='categorical',
            shuffle=True
        )

    def test_browse(self):
        file_path = filedialog.askdirectory()
        if file_path is not None:
            self.line_test_folder.setText(file_path)

    def test_load(self):
        datagen = ImageDataGenerator(rescale=1 / 255.0)
        self.test_data = datagen.flow_from_directory(
            self.line_test_folder.text(),
            target_size=INPUT_SHAPE[:2],
            batch_size=self.spin_test_n.value(),
            class_mode='categorical'
        )

    def tfn_train(self):
        # Create model
        self.model = tf.keras.models.Sequential([
            layers.Conv2D(64, (6, 6), padding='same', activation=KERNEL_ACTIVATION, input_shape=INPUT_SHAPE),
            layers.Conv2D(64, (6, 6), padding='same', activation=KERNEL_ACTIVATION),
            layers.MaxPooling2D((2, 2)),
            #
            layers.Conv2D(32, (2, 2), padding='same', activation=KERNEL_ACTIVATION),
            layers.Conv2D(32, (2, 2), padding='same', activation=KERNEL_ACTIVATION),
            layers.MaxPooling2D((2, 2)),
            #
            layers.Flatten(),
            layers.Dense(128, activation=MLP_ACTIVATION),
            layers.Dense(self.train_data.num_classes, activation='softmax')
        ])

        # Compile model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=OPTIMIZER,
                           metrics=['accuracy'])

        # Train model
        self.model.fit(self.train_data,
                       epochs=self.spin_tfn_epochs.value(),
                       validation_data=self.test_data)

        print('Training done.')

        loss, accuracy = self.model.evaluate(self.test_data, verbose=2)
        print('Accuracy: ' + str(accuracy))

    def tfn_save(self):
        self.model.save('LR5_data/model.h5')

    def tfn_load(self):
        self.model = keras.models.load_model('LR5_data/model.h5')

    def tfn_predict(self):
        class_names = []
        for folder, dirs, files in os.walk(self.line_test_folder.text()):
            for directory in dirs:
                class_names.append(directory)

        random_class = random.randrange(0, self.test_data.num_classes)
        test_path = self.line_test_folder.text() + '/' + class_names[random_class]
        random_image = random.randrange(0, len(os.listdir(test_path)))

        img_array = cv2.cvtColor(cv2.imread(
            os.path.join(test_path, os.listdir(test_path)[random_image])), cv2.COLOR_BGR2RGB)

        new_array = cv2.resize(img_array, INPUT_SHAPE[:2])

        self.cvl_image.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
            cv2.resize(img_array, (320, 240), interpolation=cv2.INTER_NEAREST))))

        new_array = np.expand_dims(new_array, axis=0)
        new_array = np.array(new_array).reshape(PREDICT_SHAPE)
        new_array = new_array / 255.0

        result_1 = self.model.predict([new_array])
        result = int(np.argmax(result_1[0]))

        print(class_names[result] + ' / ' + class_names[random_class])
        self.label_tfn_result.setText(class_names[result] + ' on photo. ' + class_names[random_class] + ' in reality.')

    def predict_camera(self):
        self.camera_running = True
        self.cv_cap = cv2.VideoCapture(self.camera_id.value(), cv2.CAP_DSHOW)
        thread = threading.Thread(target=self.camera_prediction)
        thread.start()

    def predict_camera_stop(self):
        self.camera_running = False

    def camera_prediction(self):
        class_names = []
        for folder, dirs, files in os.walk(self.line_test_folder.text()):
            for directory in dirs:
                class_names.append(directory)

        while self.camera_running:
            ret, img = self.cv_cap.read()
            img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            new_array = cv2.resize(img_array, INPUT_SHAPE[:2])

            self.cvl_image.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
                cv2.resize(img_array, (320, 240), interpolation=cv2.INTER_NEAREST))))

            new_array = np.expand_dims(new_array, axis=0)
            new_array = np.array(new_array).reshape(PREDICT_SHAPE)
            new_array = new_array / 255.0

            result_1 = self.model.predict([new_array])
            result = int(np.argmax(result_1[0]))

            self.label_tfn_result.setText(class_names[result] + ' on photo.')
        self.cv_cap.release()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('fusion')
    window = LR5()
    window.show()
    app.exec_()
