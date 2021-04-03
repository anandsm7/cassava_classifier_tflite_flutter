import os
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config

class CassavaDataGenerator:
    def __init__(self, train_df):
        self.train_df = train_df
        self.train_img_path = config.TRAIN_IMG_PATH

        self.train_df.label = self.train_df.label.astype('str')

    def train_generator(self):
        train_datagen = ImageDataGenerator(validation_split=0.2,
                                            preprocessing_function=None,
                                            rotation_range = 45,
                                            zoom_range=0.2,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            fill_mode='nearest',
                                            shear_range=0.1,
                                            height_shift_range=0.1,
                                            width_shift_range=0.1
                                            )
        train_gen = train_datagen.flow_from_dataframe(self.train_df, 
                                                        directory=self.train_img_path,
                                                        subset='training',
                                                        x_col='image_id',
                                                        y_col='label',
                                                        target_size=(config.TARGET_SIZE,config.TARGET_SIZE),
                                                        batch_size = config.BATCH_SIZE,
                                                        class_mode='sparse'
                                                        ) 
        return train_gen

    def valid_generator(self):
        valid_dataset = ImageDataGenerator(validation_split=0.2)

        valid_gen = valid_dataset.flow_from_dataframe(self.train_df,
                                                        directory=self.train_img_path,
                                                        subset = 'validation',
                                                        x_col='image_id',
                                                        y_col='label',
                                                        target_size = (config.TARGET_SIZE, config.TARGET_SIZE),
                                                        batch_size= config.BATCH_SIZE,
                                                        class_mode='sparse'
                                                        )
        return valid_gen


if __name__ == "__main__":
    train_df = pd.read_csv(config.TRAIN_PATH)

    casava_data_obj = CassavaDataGenerator(train_df)
    xtrain_gen = casava_data_obj.train_generator()
    xvalid_gen = casava_data_obj.valid_generator()
    print(xtrain_gen.class_indices, xvalid_gen.class_indices)

                                    
