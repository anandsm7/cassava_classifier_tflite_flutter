import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, \
    ReduceLROnPlateau,TensorBoard
from sklearn.model_selection import train_test_split

import config
import models 
import dataset

class Trainer:
    def __init__(self, data, model,num_epochs, validation_step, steps_per_epochs):
        self.data = data
        self.model = model
        self.num_epochs = num_epochs
        self.validation_step = validation_step
        self.steps_per_epochs = steps_per_epochs

    def __len__(self):
        return data.shape

    def train(self):
        model_save = ModelCheckpoint(config.MODEL_PATH+'best_effnetB0.h5',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    monitor='val_loss',
                                    mode='min', verbose=1
                                    )
        early_stopping = EarlyStopping(monitor='val_loss',
                                        min_delta=0.00,
                                        patience=5,
                                        mode='min',
                                        verbose=1
                                        )
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, 
                                        patience=2, min_delta=0.001,
                                        mode='min', verbose=1
                                        )
        tensorboard_callbacks = TensorBoard(log_dir=config.TB_LOGS)
        
        datagenerator = dataset.CassavaDataGenerator(self.data)
        train_gen = datagenerator.train_generator()
        valid_gen = datagenerator.valid_generator()

    
        history = self.model.fit(
            train_gen,
            steps_per_epoch = self.steps_per_epochs,
            epochs = self.num_epochs,
            validation_data= valid_gen,
            validation_steps = self.validation_step,
            callbacks = [tensorboard_callbacks,model_save, early_stopping, reduce_lr]
        )

        return history

if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_PATH)
    steps_per_epoch = len(df)*0.8 / config.BATCH_SIZE
    validation_steps = len(df)*0.2 / config.BATCH_SIZE
    model = models.CassavaEffNetB0(config.NUM_CLASSES, config.TARGET_SIZE, config.LR).create_model() 
    trainer = Trainer(df, model, config.EPOCHS, validation_steps, steps_per_epoch)
    history= trainer.train()