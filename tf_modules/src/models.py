# from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
from keras_efficientnets import EfficientNetB0
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy

import config

class CassavaEffNetB0:
    def __init__(self, num_classes, target_size, lr, ls=False):
        self.num_classes = num_classes
        self.target_size = target_size
        self.lr = lr
        self.ls = ls
        
    def scce_with_ls(self, y, y_hat):
        y = tf.one_hot(tf.cast(y, tf.int32), self.num_classes)
        return CategoricalCrossentropy(y, y_hat, label_smoothing=0.1)

    def create_model(self):
        base_layer = EfficientNetB0(include_top=False, 
                                        weights='imagenet', 
                                        input_shape=(self.target_size, self.target_size, 3))
        model = base_layer.output
        model = layers.GlobalAveragePooling2D()(model)
        model = layers.Dense(self.num_classes, activation='softmax')(model)
        model = models.Model(base_layer.input, model)
        
        if self.ls:
            loss_fn = self.scce_with_ls
        else:
            loss_fn = SparseCategoricalCrossentropy()

        model.compile(optimizer = Adam(lr=self.lr),
                        loss=loss_fn,
                        metrics=['acc']
                        )
        return model

if __name__ == "__main__":
    model = CassavaEffNetB0(config.NUM_CLASSES, config.TARGET_SIZE, config.LR).create_model()
    print(model.summary())
