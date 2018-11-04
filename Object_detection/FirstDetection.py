from imageai.Detection import ObjectDetection
import os
#execution_path = os.getcwd()
h5_path = "D:/MyProject/Model-AI"
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(h5_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
execution_path = ""
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

from PIL import Image
image = Image.open('imagenew.jpg')
image.show()
'''
- Copy code from the good article:
https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606
- using library:
https://github.com/OlafenwaMoses/ImageAI
- how to install
pip install tensorflow
pip install numpy
pip install scipy
pip install opencv-python
pip install pillow
pip install matplotlib
pip install h5py
pip install keras
pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl
- download resnet50_coco_best_v2.0.1.h5
https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
- "image.jpg" is a source file and "imagenew.jpg" is a output file
- picture from thai Drama: "เลือดข้นคนจาง"
 '''

