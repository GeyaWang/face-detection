import numpy as np
import random
from PIL import Image
import math

# https://github.com/GeyaWang/py-nn.git
from nn.models import Sequential
from nn.layers import Conv2D, Dense, Activation, Flatten, Dropout, MaxPooling2D
from nn.activations import ReLU
from nn.optimisers import Adam
from nn.losses import MeanSquaredError


def draw_ellipse(arr, thickness, colour, major_axis_radius, minor_axis_radius, angle, center_x, center_y):
    cos_angle = np.cos(-angle)
    sin_angle = np.sin(-angle)

    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            x_shifted = x - center_x
            y_shifted = y - center_y

            x_rotated = x_shifted * cos_angle - y_shifted * sin_angle
            y_rotated = x_shifted * sin_angle + y_shifted * cos_angle

            if 1 > (x_rotated / major_axis_radius) ** 2 + (y_rotated / minor_axis_radius) ** 2 > 1 - thickness:
                arr[y, x] = colour


def resize_img_data(width, arr, data):
    data = data = data.tolist()
    major_axis_radius, minor_axis_radius, angle, center_x, center_y, x = data

    H, W, _ = arr.shape
    scale = width / max(H, W)

    major_axis_radius *= scale
    minor_axis_radius *= scale
    center_x *= scale
    center_y *= scale

    img = Image.fromarray(arr)
    img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    arr = np.array(img)

    H, W, _ = arr.shape
    pad_x = (max(H, W) - H) / 2
    pad_y = (max(H, W) - W) / 2
    arr = np.pad(arr, ((math.ceil(pad_x), math.floor(pad_x)), (math.ceil(pad_y), math.floor(pad_y)), (0, 0)))

    center_x += pad_y
    center_y += pad_x

    return arr, np.array([major_axis_radius, minor_axis_radius, angle, center_x, center_y, x])


def get_img_dict():
    img_dict = {}

    for i in range(1, 11):
        anno_filepath = 'anno/FDDB-fold-' + f'{i}'.rjust(2, '0') + '-ellipseList.txt'

        with open(anno_filepath, 'r') as f:
            lines = list(f)
            for idx, line in enumerate(lines):
                # print(line.strip(), idx)
                if line.strip().isdigit():
                    n = int(line.strip())
                    if n != 1:
                        continue

                    filepath = 'faces-img/' + lines[idx - 1].strip() + '.jpg'
                    data = np.array(lines[idx + 1].strip().split(' ')[:5] + ['1'], dtype=np.float64)
                    img_dict[filepath] = data
    return img_dict


def main():
    width = 128

    img_dict = get_img_dict()
    x_train = np.zeros((len(img_dict), width, width, 3))
    y_train = np.zeros((len(img_dict), 6))

    for i, (filepath, data) in enumerate(img_dict.items()):
        with Image.open(filepath).convert('RGB') as img:
            arr = np.array(img)
            arr, data = resize_img_data(width, arr, data)
            x_train[i] = arr
            y_train[i] = data

    model = Sequential()
    model.add(Conv2D(64, 5, input_shape=(width, width, 3)))
    model.add(Activation(ReLU()))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, 3))
    model.add(Activation(ReLU()))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, 3))
    model.add(Activation(ReLU()))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Dense(128))
    model.add(Dense(6))

    model.compile(Adam(), MeanSquaredError())

    model.summary()

    model.fit(x_train, y_train, batch_size=32, epochs=20, running_mean_size=200, save_filepath='training.ptd')


if __name__ == '__main__':
    main()
