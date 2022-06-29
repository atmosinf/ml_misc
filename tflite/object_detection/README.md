## object detection tf

#### tflite_load_model.ipynb
load a model from tfhub using the provided link, get the prediction and visualize the outputs <br>
load a model from a .tflite file, create an interpreter and predict the output <br>
time the tflite model's invoke method (the invoke method calculated the model prediction) <br>

#### coco_filter.py
1. saves images/annotations from categories <br>
2. creates new json by filtering the main json file <br>

#### filtered_cocodataset_test.ipynb
examine the filtered coco dataset which was created from coco_filter.py. visualize some images <br>

### in tflite_load_model.ipynb check why inference takes so long for the tflite model (time shown in seconds). the inference time should be ~50 ms (scroll down for findings)
#### local CPU
![1](screenshots/tlocalcpu.jpg)<br><br>

#### colab GPU
![2](screenshots/tcolangpu.jpg)<br><br>

#### colab TPU
![3](screenshots/colantpu.jpg)<br><br>

#### colab CPU
![4](screenshots/colabcpu.jpg)<br><br>

#### reasons for inference time being slow
TFLite focuses more on on-device performance. So it is not as optimized when you run it on a x86 machines compared to arm devices for example.
https://stackoverflow.com/questions/70911977/is-there-any-method-to-decrease-tensorflow-lite-invoke-time-in-c 