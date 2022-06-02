import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class Generator(tf.keras.utils.Sequence):
    """
    A Custom image data generator to supply two image inputs to a model.
    The images flow from a dataframe path. 
    """

    def __init__(self, df, X_col, y_col, batch_size, model_input_shape, color='grayscale', shuffle=False, zoom=0.1,
                 h_flip=True):

        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shape = model_input_shape
        self.color = color
        self.shuffle = shuffle
        self.n = len(self.df)
        self.zoom = zoom
        self.h_flip = h_flip

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    # Add padding to image
    # Function taken from:
    # https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
    def __expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def __augment(self, im):

        if self.h_flip and np.random.randint(0, 2):
            im = tf.image.flip_left_right(im)

        im = tf.image.central_crop(im, 1 - self.zoom)
        im = tf.image.resize(
            im,
            size=(self.shape[:2]),
            preserve_aspect_ratio=True,
            antialias=False,
        )

        return im

    def __load(self, img_path):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=None, color_mode=self.color)
        #tf.image.pad_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
        img = self.__expand2square(img, 0)
        img = tf.image.resize(img, size=self.shape[:2], preserve_aspect_ratio=True, antialias=False)
        img = self.__augment(img)
        x = tf.keras.preprocessing.image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        x = x / 255.
        return x

    def __get_data_from_frame(self, batch):

        X = [batch[col] for col in self.X_col]

        y = batch[self.y_col]

        X = [np.asarray([self.__load(im) for im in x]) for x in X]

        y = np.asarray(y)
        #print('({}, {}), {}'.format(x0.shape, x1.shape, y.shape))
        return tuple(X), y

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data_from_frame(batches)
        return X, y

    def get_at(self, ind):

        return self.__getitem__(ind)

    def __len__(self):
        return self.n // self.batch_size

    def preview_at(self, index):

        print('hi')
        batch = self.get_at(index)
        print(batch.shape)
        X, y = batch
        fig, ax = plt.subplots(len(y), len(self.X_col), figsize=(4 * len(self.X_col), 4 * len(y)))

        for i, weight in enumerate(y):

            if len(self.X_col == 2):
                for j in range(len(self.X_col)):
                    ax[i, 0].imshow(X[j])
                    ax[i, 1].imshow(X[j])
                ax.set_title('Weight: {} lbs'.format(weight))

            elif len(self.X_col == 1):
                ax[i].imshow(X)
                ax[i].set_title('Weight: {} lbs'.format(weight))

        plt.show()
