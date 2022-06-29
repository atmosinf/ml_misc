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

#### in tflite_load_model.ipynb check why inference takes so long
#### local CPU
![1](screenshots/tlocalcpu.jpg)<br><br>

#### colab GPU
![2](screenshots/tcolangpu.jpg)<br><br>

#### colab TPU
![3](screenshots/colantpu.jpg)<br><br>

#### colab CPU
![4](screenshots/colabcpu.jpg)<br><br>
