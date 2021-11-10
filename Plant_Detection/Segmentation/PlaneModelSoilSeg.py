import numpy as np

from utils.Globals import *

class PlaneModelSoilSeg:
    def SegmentSoil(self,cloud_int, smoothness = 15):
        if type(cloud_int) != np.ndarray:
            cloud_int=cloud_int.to_array()
        cloud=pcl.PointCloud()
        cloud.from_array(cloud_int[:,:-1])
        # cloud.from_array(cloud_int.to_array()[:,:-1])

        model_p = pcl.SampleConsensusModelPlane(cloud)
        ransac = pcl.RandomSampleConsensus(model_p)
        ransac.set_DistanceThreshold(smoothness)
        ransac.computeModel()
        inliers = ransac.get_Inliers()

        # cloud_int = cloud_int.extract(inliers, True)
        mask = np.ones(len(cloud_int), np.bool)
        mask[inliers] = 0
        other_data = cloud_int[mask]
        # cloud_int= np.delete(cloud_int,inliers)
        return other_data

    def SegmentSoilLeaf(self,cloud_int, smoothness = 10):

        if type(cloud_int) == np.ndarray:
            cloud = pcl.PointCloud()
            cloud.from_array(cloud_int[:, :-1])
            c_temp = pcl.PointCloud_PointXYZI()
            c_temp.from_array(cloud_int)
            cloud_int = c_temp
        else:
            temp=cloud_int.to_array()
            cloud = pcl.PointCloud()
            cloud.from_array(temp[:, :-1])


        globals = Globals
        tray_ref=globals.trayRefPoint

        curvature = 100
        tree = cloud.make_kdtree()
        segment = cloud.make_RegionGrowing(ksearch=50)
        segment.set_MinClusterSize(30)
        segment.set_MaxClusterSize(250000)
        segment.set_NumberOfNeighbours(30)
        segment.set_SmoothnessThreshold((smoothness / 180.0) *(3.14))
        segment.set_CurvatureThreshold(curvature)
        segment.set_SearchMethod(tree)
        clusters = segment.Extract()

        pcl.RegionGrowing()

        biggest_cluster=[]
        cls_size = len(clusters[0])
        for i,index in enumerate(clusters):
            d= cloud_int.to_array()[index,2]
            if ((tray_ref +10) < np.mean(d)):
                biggest_cluster.append(i)

        crop_cloud_index=[]
        for i, a in enumerate(clusters):
            if i not in  biggest_cluster:
                crop_cloud_index.extend(clusters[i])

        cloud_int = cloud_int.extract(crop_cloud_index, False)
        return cloud_int

