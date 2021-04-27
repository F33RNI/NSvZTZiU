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

import numpy as np
import cv2
import sys
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
import qimage2ndarray
import random
from tkinter import filedialog
import tkinter

root = tkinter.Tk()
root.withdraw()

import gui_3_Collector


def valmap(value, istart, istop, ostart, ostop):
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))


class CollectorApp(QtWidgets.QMainWindow, gui_3_Collector.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btn_open_file.clicked.connect(self.open_file)
        self.btn_candidates_generate.clicked.connect(self.generate_candidates)

        # self.cvl_original.setStyleSheet("background-color: lightgreen")
        # self.cvl_candidates.setStyleSheet("background-color: lightgreen")

        self.original_image = None
        self.candidates_image = None

    def open_file(self):
        file_path = filedialog.askopenfilename()
        temp_image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
        self.original_image = cv2.resize(temp_image, (20, 20), interpolation=cv2.INTER_NEAREST)
        self.cvl_original.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
                cv2.cvtColor(cv2.resize(self.original_image, (80, 80), interpolation=cv2.INTER_NEAREST),
                             cv2.COLOR_GRAY2RGB))))

        print(file_path)

    def generate_candidates(self):
        self.candidates_image = np.zeros(shape=[320, 320], dtype=np.uint8)  # [16 X 16] images or [16*20 X 16*20] pixels
        generated_images = []

        # Generation of modified images
        for n in range(self.spin_cindidates_n.value()):
            temp_image = self.original_image.copy().astype('float32')

            # Brightness noise
            if self.slider_brightness.value():
                temp_image += random.randrange(-self.slider_brightness.value(), self.slider_brightness.value())

            temp_image = np.clip(temp_image, 0, 255)
            temp_image = temp_image.astype('uint8')

            # Pixel noise
            if self.slider_noise.value():
                for i in range(20):
                    for j in range(20):
                        if random.randrange(0, 2) and temp_image[i][j] > 0:
                            brightness_pixel = temp_image[i][j] + random.randrange(-self.slider_noise.value(),
                                                                                   self.slider_noise.value())
                            if brightness_pixel > 255: brightness_pixel = 255
                            if brightness_pixel < 0: brightness_pixel = 0
                            temp_image[i][j] = brightness_pixel

            # Rotation
            if self.slider_rotation.value() > 0:
                num_rows, num_cols = temp_image.shape[:2]
                rotation_matrix = cv2.getRotationMatrix2D(
                    (num_cols / 2, num_rows / 2),
                    random.randrange(-self.slider_rotation.value(), self.slider_rotation.value()), 1)
                temp_image = cv2.warpAffine(temp_image, rotation_matrix, (num_cols, num_rows))

            # Pixel shift
            if self.slider_shift.value() > 0:
                shift_size_x = random.randrange(-self.slider_shift.value(), self.slider_shift.value())
                shift_size_y = random.randrange(-self.slider_shift.value(), self.slider_shift.value())
                rows, cols = temp_image.shape
                M = np.float32([[1, 0, shift_size_x], [0, 1, shift_size_y]])
                temp_image = cv2.warpAffine(temp_image, M, (cols, rows))

                generated_images.append(temp_image)

        if not os.path.exists('LR3_data'):
            os.makedirs('LR3_data')

        image_counter = 0
        for y in range(16):
            for x in range(16):
                temp_image = generated_images[image_counter]

                # Save to file
                cv2.imwrite('LR3_data/' + str(self.spin_cindidates_file.value())
                            + '_' + str(image_counter) + '.bmp', temp_image)

                self.candidates_image[y * 20: y * 20 + 20, x * 20: x * 20 + 20] = temp_image  # y:y + h, x:x + w
                image_counter += 1
                if image_counter == len(generated_images):
                    break
            if image_counter == len(generated_images):
                break

        self.cvl_candidates.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
            cv2.cvtColor(cv2.resize(self.candidates_image, (512, 512),
                                    interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2RGB))))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion")
    window = CollectorApp()
    window.show()
    app.exec_()
