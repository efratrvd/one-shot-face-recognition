import os

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


class LFW2DataLoader:
    def __init__(self, input_shape, images_path):
        self.input_shape = input_shape
        self.images_path = images_path

    @staticmethod
    def _pad_image_number(image_number):
        return '0' * (4 - len(image_number)) + image_number

    def _join_image_path(self, name, number):
        return os.path.join(self.images_path,
                            name,
                            f'{name}_{self._pad_image_number(number)}.jpg')

    def _read_dataset_file(self, file_path):
        first_images_paths = []
        second_images_paths = []
        labels = []
        with open(file_path) as f:
            for line in f.readlines()[1:]:
                splitted_line = line.split()
                if len(splitted_line) == 3:
                    name = splitted_line[0]
                    first_images_paths.append(self._join_image_path(name, splitted_line[1]))
                    second_images_paths.append(self._join_image_path(name, splitted_line[2]))
                    labels.append(1)
                else:
                    first_images_paths.append(self._join_image_path(splitted_line[0], splitted_line[1]))
                    second_images_paths.append(self._join_image_path(splitted_line[2], splitted_line[3]))
                    labels.append(0)

        return first_images_paths, second_images_paths, labels

    def _open_images(self, image_paths):
        return np.array([img_to_array(load_img(f_path, color_mode='grayscale').resize(self.input_shape))
                         for f_path in image_paths])

    def load_images_from_path(self, path, normalize=True):
        first_images_paths, second_images_paths, labels = self._read_dataset_file(path)
        first_images = self._open_images(first_images_paths)
        second_images = self._open_images(second_images_paths)

        if normalize:
            first_images /= 255
            second_images /= 255

        return [first_images, second_images], np.array(labels)
