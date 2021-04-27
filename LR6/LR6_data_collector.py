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
from pathlib import Path
import cv2
import numpy as np

INPUT_IMAGES_DIR = 'LR6_data/train/images'
INPUT_MASKS_DIR = 'LR6_data/train/masks'
OUTPUT_IMAGES_DIR = 'LR6_data/train/train_images2'
OUTPUT_MASKS_DIR = 'LR6_data/train/train_masks2'
BACKGROUDS_DIR = 'LR6_data/train2/backgrounds'
MASK_EXTENSION = '.png'
OUTPUT_EXTENSION = '.png'

"""
# ######### GENERATE FROM CAMERA ######### #
def glue_backgroud(input_image, input_backgroud):
    chroma_mask = cv2.inRange(input_image, (0, 100, 0), (80, 255, 80))
    input_image = np.copy(input_image)
    input_image[chroma_mask != 0] = [0, 0, 0]
    input_backgroud[chroma_mask == 0] = [0, 0, 0]
    output_image = input_image + input_backgroud
    return output_image


file_id_counter = 0
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
kernel_filter = np.ones((7, 7), np.uint8)
kernel = np.ones((15, 15), np.uint8)
while True:
    ret, img = cap.read()
    if img is None:
        break
    if img.shape[1] != 640 and img.shape[0] != 480:
        resized = cv2.resize(img, (854, 480))
        img = np.zeros((480, 640, 3), np.uint8)
        img[0:480, 0:640] = resized[0:480, 107:107 + 640]

    original_image = img.copy()
    final_image = glue_backgroud(img, cv2.imread(BACKGROUDS_DIR + '/' + str(0) + OUTPUT_EXTENSION))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (55, 0, 0), (65, 255, 255))
    mask = cv2.erode(mask, kernel_filter)
    mask = cv2.dilate(mask, kernel_filter)

    final_mask = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    mask_dilated_ext = cv2.dilate(255 - mask, kernel)
    mask_dilated_ext = mask_dilated_ext - (255 - mask)
    mask_dilated_int = cv2.dilate(mask, kernel)
    mask_dilated_int = mask_dilated_int - mask

    blue = np.ones((mask.shape[0], mask.shape[1]), np.uint8) * 255
    red = np.ones((mask.shape[0], mask.shape[1]), np.uint8) * 255
    blue[mask == 0] = 0
    blue[mask_dilated_ext == 255] = 0
    red[mask == 255] = 0
    red[mask_dilated_int == 255] = 0

    final_mask[:, :, 0] = blue
    final_mask[:, :, 1] = mask_dilated_ext + mask_dilated_int
    final_mask[:, :, 2] = red


    # final_mask[final_mask.shape[0] - 6:final_mask.shape[0], :, 2] = 0

    cv2.imshow('original', original_image)
    cv2.imshow('chroma', final_image)
    cv2.imshow('mask', final_mask)
    k = cv2.waitKey(30) & 0xff
    if k == 13:  # Enter
        cv2.imwrite(BACKGROUDS_DIR + '/' + str(file_id_counter) + OUTPUT_EXTENSION, original_image)
        print('File', file_id_counter, 'saved.')
        file_id_counter += 1
    elif k == 32:  # Space
        print('Start saving from', file_id_counter, '...')
        background_id_counter = 0
        entries = Path(BACKGROUDS_DIR)
        for entry in entries.iterdir():
            background = cv2.imread(BACKGROUDS_DIR + '/'
                                                         + str(background_id_counter) + OUTPUT_EXTENSION)
            if background is not None:
                final_image = glue_backgroud(img, background)
                cv2.imwrite(OUTPUT_IMAGES_DIR + '/' + str(file_id_counter) + OUTPUT_EXTENSION, final_image)
                cv2.imwrite(OUTPUT_MASKS_DIR + '/' + str(file_id_counter) + OUTPUT_EXTENSION, final_mask)
                file_id_counter += 1
            background_id_counter += 1
        print('Done', file_id_counter, '.')
    elif k == 27:  # ESC
        break
cap.release()

# ######### GENERATE FROM FILES ######### #

"""
_, _, files = next(os.walk(INPUT_IMAGES_DIR))
file_id_counter = 0
entries = Path(INPUT_IMAGES_DIR)
kernel = np.ones((7, 7), np.uint8)
for entry in entries.iterdir():
    image = cv2.imread(str(entry), 0)
    mask = cv2.imread(INPUT_MASKS_DIR + '/' + os.path.splitext(entry.name)[0] + MASK_EXTENSION, 0)
    if image is not None:
        # image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)  # (640, 480)
        # mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        object_layer = mask.copy()
        object_layer[object_layer == 1] = 255
        object_layer[object_layer == 2] = 0
        object_layer[object_layer == 3] = 0
        contout_layer = mask.copy()
        contout_layer[contout_layer == 1] = 0
        contout_layer[contout_layer == 2] = 0
        contout_layer[contout_layer == 3] = 255
        background_layer = mask.copy()
        background_layer[background_layer == 1] = 0
        background_layer[background_layer == 2] = 255
        background_layer[background_layer == 3] = 0

        mask = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)  # (480, 640, 3)
        mask[:, :] = cv2.dilate(object_layer, kernel)

        cv2.imwrite(OUTPUT_IMAGES_DIR + '/' + str(file_id_counter) + OUTPUT_EXTENSION, image)
        cv2.imwrite(OUTPUT_MASKS_DIR + '/' + str(file_id_counter) + OUTPUT_EXTENSION, mask)

        file_id_counter += 1

        if file_id_counter % 100 == 0:
            print('Collected', file_id_counter, 'out of', len(files))

