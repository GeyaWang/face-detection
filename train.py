import numpy as np
from datasets import load_dataset
from PIL import Image

# https://github.com/GeyaWang/py-nn.git
from nn.models import Sequential
from nn.layers import Conv2D, Dense, Activation, Flatten, Dropout, MaxPooling2D
from nn.activations import ReLU, Sigmoid
from nn.optimisers import Adam
from nn.losses import MeanSquaredError

IMG_WIDTH = 224


def process_img(img):
    W, H = img.size
    scale = IMG_WIDTH / max(H, W)

    img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    arr = np.array(img)

    H, W, _ = arr.shape
    pad_y = (max(H, W) - H) // 2
    pad_x = (max(H, W) - W) // 2
    arr = np.pad(arr, ((pad_y, IMG_WIDTH - H - pad_y), (pad_x, IMG_WIDTH - W - pad_x), (0, 0)))

    return Image.fromarray(arr)


def get_training_data():
    print("Getting Training Data...", end='')
    dataset = load_dataset("wider_face")
    training = dataset['train']

    N = len(training)

    x_train = []
    y_train = []

    for i in range(100):
        print(f'\rProcessing images {i}/{N}', end='')

        img = training[i]['image']
        bbox = training[i]['faces']['bbox']

        # only one face
        if len(bbox) > 1:
            continue

        W, H = img.size
        x, y, w, h = bbox[0]
        rel_bbox = [x / W, y / H, w / W, h / H]

        img = process_img(img)

        x_train.append(np.array(img))
        y_train.append(rel_bbox)

    print('\rDone.')
    return np.array(x_train), np.array(y_train)


def main():
    x_train, y_train = get_training_data()

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
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(4))
    model.add(Activation(Sigmoid()))

    model.compile(Adam(), MeanSquaredError())

    model.summary()

    model.fit(x_train, y_train, batch_size=32, epochs=20, running_mean_size=200, save_filepath='training.ptd')


if __name__ == '__main__':
    main()
