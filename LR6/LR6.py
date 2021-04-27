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
import random
import threading
import numpy as np
import cv2
import sys
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
import qimage2ndarray
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
import Model

import gui_6


class LR6(QtWidgets.QMainWindow, gui_6.Ui_MainWindow):
    def __init__(self):
        super().__init__()

        self.setupUi(self)
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_single.clicked.connect(self.single)
        self.btn_camera_start.clicked.connect(self.camera_start)
        self.btn_camera_pause.clicked.connect(self.camera_pause)
        self.btn_camera_stop.clicked.connect(self.camera_stop)
        self.btn_train_start.clicked.connect(self.train_start)
        self.btn_train_save.clicked.connect(self.train_save)

        self.camera_running = False
        self.camera_paused = False
        self.cv_cap = None
        self.model = None
        self.image_shape = None

    def single(self):
        random_file = self.test_image.text() + '/' + random.choice(os.listdir(self.test_image.text()))
        self.proceed_frame(cv2.imread(random_file))

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_file.text())

    def proceed_frame(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.model_width.value(), self.model_height.value()),
                           interpolation=cv2.INTER_NEAREST)

        mask = self.model.predict(np.array([image / 255]))[0]
        mask = np.array(mask * 255).astype('uint8')
        mask_r = mask[:, :, 0]

        if self.checkbox_binarize.isChecked():
            mask_r[mask_r >= 127] = 255
            mask_r[mask_r < 127] = 0

        self.main_image.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
            cv2.cvtColor(cv2.resize(image, (320, 240)), cv2.COLOR_GRAY2RGB))))

        self.mask_image.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
            cv2.cvtColor(cv2.resize(mask_r, (320, 240)), cv2.COLOR_GRAY2RGB))))

    def camera_start(self):
        if not self.camera_paused:
            self.camera_running = True
            self.cv_cap = cv2.VideoCapture(self.camera_id.value(), cv2.CAP_DSHOW)
            thread = threading.Thread(target=self.camera_process)
            thread.start()
        self.camera_paused = False
        self.camera_id.setEnabled(False)
        self.btn_camera_start.setEnabled(False)
        self.btn_camera_pause.setEnabled(True)

    def camera_pause(self):
        self.camera_paused = True
        self.btn_camera_start.setEnabled(True)
        self.btn_camera_pause.setEnabled(False)

    def camera_process(self):
        while self.camera_running:
            if not self.camera_paused:
                ret, img = self.cv_cap.read()
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if img is not None:
                    self.proceed_frame(img)
        self.cv_cap.release()

    def camera_stop(self):
        self.camera_paused = False
        self.camera_running = False
        self.camera_id.setEnabled(True)
        self.btn_camera_start.setEnabled(True)
        self.btn_camera_pause.setEnabled(False)

    def train_start(self):
        self.image_shape = (self.cnn_model_height.value(), self.cnn_model_width.value(), 1)

        thread = threading.Thread(target=self.training_thread)
        thread.start()

    def training_thread(self):
        # noinspection PyBroadException
        try:
            self.btn_train_start.setEnabled(False)
            self.btn_train_save.setEnabled(False)

            self.model = Model.unet(self.image_shape)
            data_gen_args = dict(rescale=1 / 255.0)  # rescale=1 / 255.0

            self.model.fit(self.train_generator(2, self.cnn_train_folder.text(),
                                                self.cnn_images_folder.text(), self.cnn_masks_folder.text(),
                                                data_gen_args, save_to_dir=None, image_color_mode='grayscale',
                                                mask_color_mode='grayscale', target_size=(self.image_shape[0],
                                                                                          self.image_shape[1])),
                           steps_per_epoch=self.cnn_steps_per_epoch.value(), epochs=self.cnn_epochs.value())  # 24

            self.btn_train_start.setEnabled(True)
            self.btn_train_save.setEnabled(True)
        except:
            print(sys.exc_info())

    def train_save(self):
        self.model.save(self.cnn_train_save_file.text())

    def train_generator(self, batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode='grayscale',
                        mask_color_mode='grayscale', image_save_prefix='image', mask_save_prefix='mask',
                        save_to_dir=None, target_size=(256, 256), seed=1):

        image_datagen = ImageDataGenerator(**aug_dict)
        mask_datagen = ImageDataGenerator(**aug_dict)
        image_generator = image_datagen.flow_from_directory(
            train_path,
            classes=[image_folder],
            class_mode=None,
            color_mode=image_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=image_save_prefix,
            seed=seed)
        mask_generator = mask_datagen.flow_from_directory(
            train_path,
            classes=[mask_folder],
            class_mode=None,
            color_mode=mask_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=mask_save_prefix,
            seed=seed)
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            yield img, mask


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion")
    window = LR6()
    window.show()
    app.exec_()
