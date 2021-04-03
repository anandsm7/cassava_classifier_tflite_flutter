import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import models
import config

class TfLiteConverter:
    def __init__(self, model, lite_model_name, quantize=False):
        self.model = model
        self.lite_model_name = lite_model_name
        self.quantize = quantize
        
    def converter(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        if self.quantize:
            optimization = tf.lite.Optimize.OPTIMIZE_FOR_LATENCY
            converter.optimizations = [optimization]
        tflite_model = converter.convert()
        with open(f'{config.MODEL_PATH}+{self.lite_model_name}.tflite', 'wb') as f:
            f.write(tflite_model)
        
            
    def interpret(self, img_path):
        #TODO
        interperter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
            

if __name__ == '__main__':
    model = models.CassavaEffNetB0(config.NUM_CLASSES, config.TARGET_SIZE, config.LR).create_model()
    model.load_weights(config.MODEL_PATH + 'best_effnetB0.h5')
    tflite = TfLiteConverter(model, 'EffNetB0')
    tflite.converter()

        
        