import numpy as np
from datasets import load_dataset
from PIL import Image, ImageDraw

# https://github.com/GeyaWang/py-nn.git
from nn.models import Sequential
from nn.layers import Conv2D, Dense, Activation, Flatten, Dropout, MaxPooling2D, Reshape
from nn.activations import ReLU
from nn.optimisers import Adam
from nn.losses import MeanSquaredError

IMG_WIDTH = 224
MAX_FACES = 3


def process_img_bboxes(img, bboxes):
    W, H = img.size
    scale = IMG_WIDTH / max(H, W)

    img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    arr = np.array(img)

    H, W, _ = arr.shape
    pad_y = (max(H, W) - H) // 2
    pad_x = (max(H, W) - W) // 2
    arr = np.pad(arr, ((pad_y, IMG_WIDTH - H - pad_y), (pad_x, IMG_WIDTH - W - pad_x), (0, 0)))

    for i, (x, y, w, h) in enumerate(bboxes):
        bboxes[i] = (x * scale + pad_x, y * scale + pad_y, w * scale, h * scale)

    return Image.fromarray(arr), bboxes


def get_training_data():
    print("Getting Training Data...", end='')
    dataset = load_dataset("wider_face")
    training = dataset['train']

    N = len(training)

    x_train = np.zeros((N, IMG_WIDTH, IMG_WIDTH, 3))
    y_train = np.zeros((N, MAX_FACES, 4))

    for i in range(N):
        img = training[i]['image']
        bboxes = training[i]['faces']['bbox'][:MAX_FACES]

        img, bboxes = process_img_bboxes(img, bboxes)

        x_train[i] = np.array(img)

        for idx, bbox in enumerate(bboxes):
            y_train[i, idx] = bbox

        # img_draw = ImageDraw.Draw(img)
        # for x, y, w, h in bboxes:
        #     img_draw.rectangle((x, y, x + w, y + h), outline='red')
        #
        # img.show()

    print('     Done.')
    return x_train, y_train


def main():
    x_train, y_train = get_training_data()

    model = Sequential()
    model.add(Conv2D(128, 5, input_shape=(IMG_WIDTH, IMG_WIDTH, 3)))
    model.add(Activation(ReLU()))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3))
    model.add(Activation(ReLU()))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, 3))
    model.add(Activation(ReLU()))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dense(MAX_FACES * 4))
    model.add(Reshape((MAX_FACES, 4)))

    model.compile(Adam(), MeanSquaredError())

    model.summary()

    model.fit(x_train, y_train, batch_size=32, epochs=20, running_mean_size=200, save_filepath='training.ptd')


if __name__ == '__main__':
    main()
