# Machine_Learning_Based_Plant_Detection_on_3D_Canopy_scans
This repository implements the paper ["Machine Learning Based Plant Detection Algorithms to Automate Counting Tasks using 3D Canopy scans"](https://www.mdpi.com/1424-8220/21/23/8022) 

![Flowchart of the plant counting pipeline](/docs/flowchart.JPG)


## Requirements
- Python 3.6
- Tensorflow 1.14
- Pyqt5 5.15.4
- Others

## Introduction 
The study intended to take advantage of the recent development in ML approaches and attempt to build alternative and more flexible PlantEye data processing architecture. The main objective was to automatically count plants from complex canopy scans.

Our proposals included: 
1. Development of a 3D point cloud segmentation algorithm to separate plants from their surroundings
2. Development of ML-based mung bean and chickpea plant detection models  
3. Evaluation of the proposed pipeline performance

## Example Results
Faster RCNN Inception-v2 and Faster RCNN ResNet50 models produced mung bean and chickpea detection results for randomly selected five test images and their original views
![Results](/docs/results.JPG)

## Run
1. If you want to retrain models, first go "plant_detection_models",  (**Skip this step if you want to work with the pre-trained model**)
   - Training 
     - Run python model_main.py --alsologtostderr --model_dir=training/ --pipeline_config_path=training/[model name]
     - For Example: python model_main.py --alsologtostderr --model_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config


   - Create inference grap
     - For example: python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-20000 --output_directory inference_graph
   - move the created "inference graph"  file under the "Plant_Detection" directory 

2. Testing
   - Execute "Plant_Detection" folder 
   - Select  "sample_mungbean_ply" file (file with raw point cloud mung bean files) via user interface
   - Press "Start Plant Detection" button
   ![User Interface](/docs/user_interface.JPG)


## Citing
