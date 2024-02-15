import numpy as np
from PIL import Image
import os

# https://github.com/GeyaWang/py-nn.git
from nn.models import Sequential
from nn.branch import Branch
from nn.layers import Conv2D, Dense, Activation, Flatten, Dropout, MaxPooling2D
from nn.activations import ReLU, Sigmoid, SoftMax
from nn.optimisers import Adam
from nn.losses import MeanSquaredError, CrossEntropy

IMG_WIDTH = 224


def process_img(img):
    W, H = img.size
    scale = IMG_WIDTH / max(H, W)

    img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    arr = np.array(img).astype(dtype=np.double)

    H, W, _ = arr.shape
    pad_y = (max(H, W) - H) // 2
    pad_x = (max(H, W) - W) // 2
    arr = np.pad(arr, ((pad_y, IMG_WIDTH - H - pad_y), (pad_x, IMG_WIDTH - W - pad_x), (0, 0)))

    return arr


def get_training_data():
    with open('cube-dataset.txt', 'r') as f:
        lines = list(f)
        n_images = len(lines) // 2

        x_train = []
        y_train = []

        for i in range(n_images):
            print(f'\rProcessing images {i + 1}/{n_images}', end='')

            data = np.array(lines[i * 2 + 1].strip().split(' '), dtype=np.double)
            arr = process_img(Image.open(lines[i * 2].strip()))

            if data[0] == 0:
                continue

            x_train.append(arr / 255)
            y_train.append(data[1:])
        print('\rDone.')

        return np.array(x_train), np.array(y_train)


def main():
    x_train, y_train = get_training_data()

    if os.path.isfile('training.ptd'):
        model = Sequential.load('training.ptd')
    else:
        model = Sequential()
        model.add(Conv2D(16, 5, input_shape=(IMG_WIDTH, IMG_WIDTH, 3)))
        model.add(Activation(ReLU()))
        model.add(MaxPooling2D())
        model.add(Conv2D(32, 3))
        model.add(Activation(ReLU()))
        model.add(MaxPooling2D())
        model.add(Conv2D(64, 3))
        model.add(Activation(ReLU()))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.compile(Adam())

        branch = Branch(2)
        branch.active_branch = 0

        branch.add(Dense(128), 0)
        branch.add(Dense(64), 0)
        branch.add(Dense(4), 0)
        branch.add(Sigmoid(), 0)
        branch.compile(Adam(), MeanSquaredError(), 0)

        branch.add(Dense(128), 1)
        branch.add(Dense(64), 1)
        branch.add(Dense(2), 1)
        branch.add(SoftMax(), 1)
        branch.compile(Adam(), CrossEntropy(), 1)

        model.add(branch)

    model.summary()

    model.fit(x_train, y_train, epochs=30, save_filepath='training.ptd', batch_size=8)


if __name__ == '__main__':
    main()
