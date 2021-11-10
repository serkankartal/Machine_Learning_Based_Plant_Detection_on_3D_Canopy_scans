import numpy as np

from utils.Globals import *
import random

class RegionGrowCropSoilSeg:
    def SegmentBigClusters(self,cloud_int, smoothness = 20,cls_size=500):
        if type(cloud_int) == np.ndarray:
            cloud=pcl.PointCloud()
            cloud.from_array(cloud_int[:,:-1])
            c_temp=pcl.PointCloud_PointXYZI()
            c_temp.from_array(cloud_int)
            cloud_int=c_temp
        else:
            c_temp=cloud_int.to_array()
            cloud=pcl.PointCloud()
            cloud.from_array(c_temp[:,:-1])

        # cloud=pcl.PointCloud()
        # cloud.from_array(cloud_int.to_array()[:,:-1])

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
        solid_index=[]
        # print("\n\n *******YENI KUME******** \n\n ")
        for i,a in enumerate(clusters):
            # a=len(clusters[i])
            # print("küme: {} sayi :{}".format(i,a))
            if (cls_size < len(clusters[i])):
                solid_index.extend(clusters[i])
        if len(solid_index)!=0:
            cloud_int = cloud_int.extract(solid_index, True)

        return cloud_int

    def SegmentBottomClusters(self,cloud_int, smoothness = 10):
        if type(cloud_int) != np.ndarray:
            cloud_int = cloud_int.to_array()

        cloud=pcl.PointCloud()
        cloud.from_array(cloud_int[:,:-1])

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

        crop_cloud_index = []
        for i, index in enumerate(clusters):
            intensity = cloud_int[index, 3]
            z = cloud_int[index, 2]
            # if ((-5) < np.mean(z)):  # intensity 120 kalsın
            if ((120) > np.mean(intensity) or (-5) < np.mean(z)):  # intensity 120 kalsın
                crop_cloud_index.extend(clusters[i])

        mask = np.ones(len(cloud_int), np.bool)
        mask[crop_cloud_index] = 0
        other_data = cloud_int[mask]

        c = pcl.PointCloud_PointXYZI()
        c.from_array(other_data)
        return c

    def SegmentSoil(self,cloud_int, smoothness=8):

        if type(cloud_int) == np.ndarray:
            cloud=pcl.PointCloud()
            cloud.from_array(cloud_int[:,:-1])
            c_temp=pcl.PointCloud_PointXYZI()
            c_temp.from_array(cloud_int)
            cloud_int=c_temp
        else:
            c_temp = cloud_int.to_array()
            cloud = pcl.PointCloud()
            cloud.from_array(c_temp[:, :-1])

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

        biggest_cluster = 0
        cls_size = len(clusters[0])
        for i,a in enumerate(clusters):
            if (cls_size < len(clusters[i])):
                biggest_cluster = i
                cls_size = len(clusters[i])

        crop_cloud_index=[]
        for i, a in enumerate(clusters):
            if i != biggest_cluster:
                crop_cloud_index.extend(clusters[i])

        cloud_int = cloud_int.extract(crop_cloud_index, False)
        return cloud_int

    def SegmentSoilLeaf(self,cloud_int, smoothness=10):
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
            intensity= cloud_int.to_array()[index,3]
            if (-10 < np.mean(d)):
                biggest_cluster.append(i)
            # if ((tray_ref +10) < np.mean(d)):
            #     biggest_cluster.append(i)

        crop_cloud_index=[]
        for i, a in enumerate(clusters):
            if i not in  biggest_cluster:
                crop_cloud_index.extend(clusters[i])

        cloud_int = cloud_int.extract(crop_cloud_index, False)
        return cloud_int

    def SegmentAcc2Seed(self, cloud_int,coordinates, smoothness=8):

        if type(cloud_int) == np.ndarray:
            cloud_np=cloud_int
            cloud = pcl.PointCloud()
            cloud.from_array(cloud_int[:, :-1])
            c_temp = pcl.PointCloud_PointXYZI()
            c_temp.from_array(cloud_int)
            cloud_int = c_temp
        else:
            cloud_np = cloud_int.to_array()
            cloud = pcl.PointCloud()
            cloud.from_array(cloud_np[:, :-1])

        curvature = 100

        clusters={}
        cloud_base=cloud
        cloud_np_base=cloud_np

        point_clouds = {}
        for k,values in coordinates.items():
            # values=line.split(' ')
            minx=float(values[0])
            miny=float(values[1])
            maxx=float(values[2])
            maxy=float(values[3])

            cloud=cloud_base
            cloud_np=cloud_np_base

            tree = cloud.make_kdtree()
            segment = cloud.make_RegionGrowing(ksearch=50)
            segment.set_MinClusterSize(10)
            segment.set_MaxClusterSize(250000)
            segment.set_NumberOfNeighbours(30)
            segment.set_SmoothnessThreshold((smoothness / 180.0) * (3.14))
            segment.set_CurvatureThreshold(curvature)
            segment.set_SearchMethod(tree)

            c_temp=[]
            while (True):
                indexList=np.where((cloud_np[:, 0] > minx) & (cloud_np[:, 0] < maxx)& (cloud_np[:, 1] > miny) & (cloud_np[:, 1] < maxy)&(cloud_np[:, 2] > -300) & (cloud_np[:, 2] < 300))

                if indexList[0].size==0:
                    break

                c_len = indexList[0].size
                index = random.randint(0, c_len-1)
                index=indexList[0][index]
                point=cloud_np[index,:]
                index_base=np.where((cloud_np_base[:, 0] ==point[0]) &(cloud_np_base[:, 1] ==point[1]) &(cloud_np_base[:, 2] ==point[2]))

                temp=segment.get_SegmentFromPoint(index_base[0][0])
                c_temp.extend(temp)

                tree = cloud.make_kdtree()
                segment2 = cloud.make_RegionGrowing(ksearch=50)
                segment2.set_MinClusterSize(10)
                segment2.set_MaxClusterSize(250000)
                segment2.set_NumberOfNeighbours(30)
                segment2.set_SmoothnessThreshold((smoothness / 180.0) * (3.14))
                segment2.set_CurvatureThreshold(curvature)
                segment2.set_SearchMethod(tree)
                temp=segment2.get_SegmentFromPoint(index)

                cloud=cloud.extract(temp,True)
                cloud_np=cloud.to_array()

            # clusters[k]=np.unique(c_temp)
            clusters[k]=c_temp

            p_temp=cloud_np_base[clusters[k],:]
            pc_temp = pcl.PointCloud_PointXYZI()
            pc_temp.from_array(p_temp)
            point_clouds[k]=pc_temp

        return point_clouds # clusters
