## Cassava Leaf Disease Classification Android Application using TensorFlow Lite + Flutter


**TapioTest** is an Android application to detect diseases in Cassava leaf, the application leverages DeepLearning model to inference and identify leaf diseases.

### **Model**
Leaf classifier is trained on EfficientNetB0 model architecure. 

Basic EfficientNetB0 architecture
![effnet](https://1.bp.blogspot.com/-DjZT_TLYZok/XO3BYqpxCJI/AAAAAAAAEKM/BvV53klXaTUuQHCkOXZZGywRMdU9v9T_wCLcBGAs/s1600/image2.png)
credits: https://heartbeat.fritz.ai/r

EfficieNetB0 is the smallest architecture within EffNet group and it was able to gain very good validation accuracy. B0 architecture also has the advantage of being small in size due to lesser parameters which makes it easy and faster to deploy within edge devices without reducing the model size using quanitization, pruning ..etc

All the model training, model testing, TFlite conversion scripts are available in  **tf_module/src/** 

create your virtual env based on **requirements.txt**

model training
```
python trainer.py
```
To convert your keras model to .tflite


```
python liteConverter.py
```


### **Flutter App**

All the android UI modules, TFlite inference scripts are available in **flutter_module**


`Flutter SDK version should be sdk: >=2.7.0 <3.0.0`


To start the flutter app

```
flutter run
```

## DEMO
![demo](https://imgur.com/VTHijU0)

App is just for education purpose 
* image credit: https://www.shutterstock.com/
