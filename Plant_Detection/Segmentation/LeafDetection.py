import os
import sys
from Segmentation.SegmentMungbean import *
from ObjectDetectionFolder import *
import pcl
from PIL import Image


sys.path.append("..")

class LeafDetection():
    def __init__(self,inference_folder):
        self.MODEL_NAME = os.getcwd() + "\\" + inference_folder+'\\inference_graph'
        self.inference_folder=inference_folder
        # Grab path to current working directory
        self.CWD_PATH = os.getcwd() + "\\" + inference_folder

    def Detect(self,cleaned_points):
        euclidean_cls=EuclideanLeafCluster()
        ply2JpgConverter=Ply2JpgConverter()

        objectDetection=ObjectDetectionFolder(self.inference_folder)

        leaf_boxes={}
        result_images={}

        outputFolder="C:/Users/serkan/Desktop/f/output"
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        for part_count, cloud in cleaned_points.items():
            leaf_clusters=euclidean_cls.ClusterLeaf(cloud)
            tray_leaf_imgs={}
            for count,cls in leaf_clusters.items():
                tray_leaf_imgs[count] = ply2JpgConverter.ConvertPly2Jpg(cls.to_array())

            detected_imgs,object_boxes=objectDetection.Detect(tray_leaf_imgs)
            boxes = np.empty([0, 4])
            for key,box_temp in object_boxes.items():
                if (len(box_temp) == 0):
                    continue
                if (box_temp.ndim == 1):
                    box_temp = box_temp.reshape([1, 4])
                boxes = np.concatenate((boxes, box_temp), axis=0)

            img=ply2JpgConverter.ConvertPly2Jpg(cloud)
            leaf_boxes[part_count]=boxes

            result_images[part_count]=self.combineDetectedObjectBoxes(img,boxes)

        return result_images,leaf_boxes

    def combineDetectedObjectBoxes(self,img, boxes):
        scores = np.ones(len(boxes))
        classes = np.ones(len(boxes))
        category_index = {"id": 1, 'name': 'leaf'}
        vis_util.visualize_boxes_and_labels_on_image_array(
            img,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            None,
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            skip_scores=True,
            skip_labels=True,
            max_boxes_to_draw=200,
            min_score_thresh=0.50)
        return img


class EuclideanLeafCluster():

    def ClusterLeaf(self,cloud):
        tolerance = 5
        clusters={}
        cloud_t=pcl.PointCloud()
        cloud_t.from_array(cloud[:, :-1])
        tree = cloud_t.make_kdtree()
        ec=cloud_t.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(tolerance)
        ec.set_MinClusterSize(200)
        ec.set_MaxClusterSize(300000)
        ec.set_SearchMethod(tree)
        cluster_indices=ec.Extract()

        for j, indices in enumerate(cluster_indices):
            pc=pcl.PointCloud_PointXYZI()
            pc.from_array(cloud[indices])
            clusters[j]=pc
        return clusters
