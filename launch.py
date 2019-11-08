from imageai.Detection import ObjectDetection
import os

def recognize(arg):
	execution_path = os.getcwd()
	detector = ObjectDetection()
	detector.setModelTypeAsRetinaNet()
	detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))
	detector.loadModel()
	res_path = os.path.splitext(arg)[0] + "_res" + os.path.splitext(arg)[1]
	detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , './uploads/' + arg), output_image_path=os.path.join(execution_path , './uploads/' + res_path))
	f = open("./output/" + res_path + ".txt", "a")
	for eachObject in detections:
		print(eachObject["name"] , " : " , eachObject["percentage_probability"], file=f)
		print(eachObject["name"] , " : " , eachObject["percentage_probability"])
	return res_path
