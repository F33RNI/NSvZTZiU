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
import time

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

import gui_4


def valmap(value, istart, istop, ostart, ostop):
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))


class LR4(QtWidgets.QMainWindow, gui_4.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btn_camera_start.clicked.connect(self.camera_start)
        self.btn_camera_stop.clicked.connect(self.camera_stop)
        self.btn_browse.clicked.connect(self.browse_file)
        self.btn_save_current.clicked.connect(self.save_current)
        self.btn_start_multiple.clicked.connect(self.start_multiple)
        self.btn_stop_multiple.clicked.connect(self.stop_multiple)

        self.cv_cap = None
        self.camera_running = False
        self.save_running = False
        self.current_frame = np.zeros((480, 640, 3), np.uint8)

    def camera_start(self):
        self.camera_running = True
        if self.check_dshow.isChecked():
            self.cv_cap = cv2.VideoCapture(self.camera_id.value(), cv2.CAP_DSHOW)
        else:
            self.cv_cap = cv2.VideoCapture(self.camera_id.value())
        thread = threading.Thread(target=self.cv_thread)
        thread.start()
        pass

    def camera_stop(self):
        self.camera_running = False

    def browse_file(self):
        files = [('PNG Image', '*.png'),
                 ('JPG Image', '*.jpg')]
        file = filedialog.asksaveasfilename(filetypes=files, defaultextension=files)

        if file is not None and len(file) > 0:
            self.line_file.setText(file)

    def save_current(self):
        if self.camera_running:
            if len(self.line_file.text()) > 0:
                cv2.imwrite(self.line_file.text(), self.current_frame)
                print('File ' + str(self.line_file.text()) + ' saved.')
            else:
                print('Empty filename!')
        else:
            print('Camera not started!')

    def start_multiple(self):
        if self.camera_running:
            if len(self.line_file.text()) > 0:
                self.btn_start_multiple.setEnabled(False)
                self.save_running = True
                thread = threading.Thread(target=self.multiple_saving)
                thread.start()
            else:
                print('Empty filename!')
        else:
            print('Camera not started!')

    def multiple_saving(self):
        iterations_counter = 0
        files_counter = 0
        # radio_infinite, radio_limit, spin_limit, spin_interval
        while self.save_running:
            filename_base = self.line_file.text()
            filename = os.path.splitext(filename_base)[0] + '_' + str(files_counter) + \
                       os.path.splitext(filename_base)[1]

            cv2.imwrite(filename, self.current_frame)

            self.label_saved_files.setText('Saved ' + str(files_counter + 1) + ' files.')
            print('File ' + filename + ' saved.')

            if self.radio_limit.isChecked():
                self.label_saved_files.setText('Passed ' + str(iterations_counter + 1) + '/'
                                               + str(self.spin_limit.value()) + ' iterations.')
                time.sleep(self.spin_interval.value() / 2)
                iterations_counter += 1
                if iterations_counter == self.spin_limit.value():
                    self.stop_multiple()
                    print('Done.')

            self.label_saved_files.setText('Saved ' + str(files_counter + 1) + ' files.')
            if self.radio_limit.isChecked():
                time.sleep(self.spin_interval.value() / 2)
            else:
                time.sleep(self.spin_interval.value())
            files_counter += 1

    def stop_multiple(self):
        self.save_running = False
        self.btn_start_multiple.setEnabled(True)

    def cv_thread(self):
        while self.camera_running:
            ret, img = self.cv_cap.read()

            # Color space
            if self.radio_color_hsv.isChecked():
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            if self.radio_color_grayscale.isChecked():
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Brightness + Contrast
            self.current_frame = img.copy().astype('float32')
            self.current_frame = (self.slider_contrast.value() / 50) * self.current_frame \
                                 + ((self.slider_brightness.value() - 50) * 4)
            self.current_frame = np.clip(self.current_frame, 0, 255)
            self.current_frame = self.current_frame.astype('uint8')

            if self.radio_color_rgb.isChecked():
                self.cvl_image.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(
                    cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB))))
            else:
                self.cvl_image.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(self.current_frame)))

        self.cv_cap.release()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion")
    window = LR4()
    window.show()
    app.exec_()
