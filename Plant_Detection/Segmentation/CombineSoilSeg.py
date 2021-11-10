import numpy as np

from utils.Globals import *

class CombineSoilSeg:
    def SegmentSoil(self,cloud_int, p_smoothness = 15, r_smoothness = 20):
        if type(cloud_int) != np.ndarray:
            cloud_int = cloud_int.to_array()
        cloud = pcl.PointCloud_PointXYZI()
        cloud.from_array(cloud_int)

        cloud_int = cloud.to_array()
        cloud = pcl.PointCloud()
        cloud.from_array(cloud_int[:, :-1])

        # region growing1
        curvature = 100
        tree = cloud.make_kdtree()
        segment = cloud.make_RegionGrowing(ksearch=50)
        segment.set_MinClusterSize(20)
        segment.set_MaxClusterSize(250000)
        segment.set_NumberOfNeighbours(30)
        segment.set_SmoothnessThreshold((r_smoothness / 180.0) * (3.14))
        segment.set_CurvatureThreshold(curvature)
        segment.set_SearchMethod(tree)
        clusters = segment.Extract()

        pcl.RegionGrowing()

        inliers = []
        for i, index in enumerate(clusters):
            d = cloud_int[index, 3]
            z = cloud_int[index, 2]
            if ((120) > np.mean(d) or len(clusters[i]) > 5000 or (0) < np.mean(z)):
                inliers.extend(clusters[i])

        mask = np.ones(len(cloud_int), np.bool)
        mask[inliers] = 0
        other_data = cloud_int[mask]
        cloud_int = other_data
        cloud.from_array(cloud_int[:, :-1])

        c = pcl.PointCloud_PointXYZI()
        c.from_array(other_data)
        other_data = c

        # region growing2
        r_smoothness = 12
        curvature = 100
        tree = cloud.make_kdtree()
        segment = cloud.make_RegionGrowing(ksearch=50)
        segment.set_MinClusterSize(100)
        segment.set_MaxClusterSize(250000)
        segment.set_NumberOfNeighbours(30)
        segment.set_SmoothnessThreshold((r_smoothness / 180.0) * (3.14))
        segment.set_CurvatureThreshold(curvature)
        segment.set_SearchMethod(tree)
        clusters = segment.Extract()

        pcl.RegionGrowing()

        crop_cloud_index = []
        for i, index in enumerate(clusters):
            intensity = cloud_int[index, 3]
            z = cloud_int[index, 2]
            if ((150) < np.mean(intensity) and (-1) > np.mean(z) and len(clusters[i]) < 5000):  # intensity 120 kalsın
                crop_cloud_index.extend(clusters[i])
                print("len:{}".format(len(clusters[i])))

        c = pcl.PointCloud_PointXYZI()
        c.from_array(cloud_int)
        other_data = c.extract(crop_cloud_index, False)
        return other_data

    def SegmentSoilLeaf(self,cloud_int, p_smoothness = 15, r_smoothness = 20):
        if type(cloud_int) != np.ndarray:
            cloud_int=cloud_int.to_array()
        cloud=pcl.PointCloud_PointXYZI()
        cloud.from_array(cloud_int)
        # cloud=pcl.PointCloud()
        # cloud.from_array(cloud_int[:,:-1])
        # cloud.from_array(cloud_int.to_array()[:,:-1])

        # min=np.amin(cloud_int[:,0])
        # max=np.amax(cloud_int[:,0])
        # x_crop_size=50
        # ptfilter = cloud.make_passthrough_filter()
        # ptfilter.set_filter_field_name("x")
        # ptfilter.set_filter_limits(min + x_crop_size, max - x_crop_size)
        # cloud = ptfilter.filter()
        #
        cloud_int = cloud.to_array()
        cloud=pcl.PointCloud()
        cloud.from_array(cloud_int[:,:-1])
        #
        # model_p = pcl.SampleConsensusModelPlane(cloud)
        # ransac = pcl.RandomSampleConsensus(model_p)
        # ransac.set_DistanceThreshold(p_smoothness)
        # ransac.computeModel()
        # inliers = ransac.get_Inliers()

        # region growing1
        curvature = 100
        tree = cloud.make_kdtree()
        segment = cloud.make_RegionGrowing(ksearch=50)
        segment.set_MinClusterSize(20)
        segment.set_MaxClusterSize(250000)
        segment.set_NumberOfNeighbours(30)
        segment.set_SmoothnessThreshold((r_smoothness / 180.0) * (3.14))
        segment.set_CurvatureThreshold(curvature)
        segment.set_SearchMethod(tree)
        clusters = segment.Extract()

        pcl.RegionGrowing()

        inliers = []
        for i, index in enumerate(clusters):
            d = cloud_int[index, 3]
            z = cloud_int[index, 2]
            # mean=np.mean(d)
            # zmean=np.mean(z)
            # if len(clusters[i])>10000:
            #     print("i {} z{}, int {} size {} \n".format(i,zmean,mean,len(clusters[i])))
            if ((+5)<np.mean(z) or len(clusters[i])>10000):
                inliers.extend(clusters[i])

        #region growing end

        mask = np.ones(len(cloud_int), np.bool)
        mask[inliers] = 0
        other_data = cloud_int[mask]
        cloud_int = other_data
        cloud.from_array(cloud_int[:, :-1])

        c = pcl.PointCloud_PointXYZI()
        c.from_array(other_data)
        other_data =c

        # region growing2
        # r_smoothness=15
        # curvature = 100
        # tree = cloud.make_kdtree()
        # segment = cloud.make_RegionGrowing(ksearch=50)
        # segment.set_MinClusterSize(50)
        # segment.set_MaxClusterSize(250000)
        # segment.set_NumberOfNeighbours(30)
        # segment.set_SmoothnessThreshold((r_smoothness / 180.0) * (3.14))
        # segment.set_CurvatureThreshold(curvature)
        # segment.set_SearchMethod(tree)
        # clusters = segment.Extract()
        #
        # pcl.RegionGrowing()
        #
        # crop_cloud_index = []
        # for i, index in enumerate(clusters):
        #     intensity = cloud_int[index, 3]
        #     z = cloud_int[index, 2]
        #     if (len(clusters[i]) <100): #intensity 120 kalsın
        #         # if not ( (180>np.mean(intensity) and len(clusters[i])<100)) or (16>np.mean(intensity) and len(clusters[i])>200):
        #         crop_cloud_index.extend(clusters[i])
        #
        # mask = np.ones(len(cloud_int), np.bool)
        # mask[crop_cloud_index] = 0
        # other_data = cloud_int[mask]
        # cloud_int = other_data
        # cloud.from_array(cloud_int[:, :-1])
        #
        # c = pcl.PointCloud_PointXYZI()
        # c.from_array(other_data)
        # other_data = c

        # c = pcl.PointCloud_PointXYZI()
        # c.from_array(cloud_int)
        # other_data = c.extract(crop_cloud_index, False)
        return other_data


    def SegmentSoilLeafBottom(self,cloud_int,fileName, p_smoothness = 15, r_smoothness = 20):
        globals = Globals()
        fileGlobals = globals.getPlyFileGlobals(fileName)
        cloud = pcl.PointCloud_PointXYZI()

        if type(cloud_int) == np.ndarray:
            cloud.from_array(cloud_int)

        z_start = fileGlobals["marker_z_start"]

        self.tray_ref_point = z_start + 135  #
        ptfilter = cloud.make_passthrough_filter()
        ptfilter.set_filter_field_name("z")
        ptfilter.set_filter_limits(z_start - 300, self.tray_ref_point)
        cloud_top = ptfilter.filter()
        cloud_top_part=cloud_top.to_array()

        ptfilter.set_filter_limits(self.tray_ref_point  - 1, self.tray_ref_point + 70)
        cloud_bottom = ptfilter.filter()

        cloud_int = cloud_bottom.to_array()
        cloud=pcl.PointCloud()
        cloud.from_array(cloud_int[:,:-1])

        curvature = 100
        tree = cloud.make_kdtree()
        segment = cloud.make_RegionGrowing(ksearch=50)
        segment.set_MinClusterSize(20)
        segment.set_MaxClusterSize(250000)
        segment.set_NumberOfNeighbours(30)
        segment.set_SmoothnessThreshold((r_smoothness / 180.0) * (3.14))
        segment.set_CurvatureThreshold(curvature)
        segment.set_SearchMethod(tree)
        clusters = segment.Extract()

        pcl.RegionGrowing()
        inliers = []
        for i, index in enumerate(clusters):
            z = cloud_int[index, 2]
            if ((+5)<np.mean(z) or len(clusters[i])>10000):
                inliers.extend(clusters[i])

        mask = np.ones(len(cloud_int), np.bool)
        mask[inliers] = 0
        other_data = cloud_int[mask]
        cloud_int = other_data

        c = pcl.PointCloud_PointXYZI()
        c.from_array( np.concatenate(cloud_int, cloud_top_part), axis=0)

        return c

    def SegmentSoil(self,cloud_int, p_smoothness = 15, r_smoothness = 20):
        if type(cloud_int) != np.ndarray:
            cloud_int = cloud_int.to_array()
        cloud = pcl.PointCloud_PointXYZI()
        cloud.from_array(cloud_int)

        cloud_int = cloud.to_array()
        cloud = pcl.PointCloud()
        cloud.from_array(cloud_int[:, :-1])

        # region growing1
        curvature = 100
        tree = cloud.make_kdtree()
        segment = cloud.make_RegionGrowing(ksearch=50)
        segment.set_MinClusterSize(20)
        segment.set_MaxClusterSize(250000)
        segment.set_NumberOfNeighbours(30)
        segment.set_SmoothnessThreshold((r_smoothness / 180.0) * (3.14))
        segment.set_CurvatureThreshold(curvature)
        segment.set_SearchMethod(tree)
        clusters = segment.Extract()

        pcl.RegionGrowing()

        inliers = []
        for i, index in enumerate(clusters):
            d = cloud_int[index, 3]
            z = cloud_int[index, 2]
            if ((120) > np.mean(d) or len(clusters[i]) > 5000 or (0) < np.mean(z)):
                inliers.extend(clusters[i])

        mask = np.ones(len(cloud_int), np.bool)
        mask[inliers] = 0
        other_data = cloud_int[mask]
        cloud_int = other_data
        cloud.from_array(cloud_int[:, :-1])

        c = pcl.PointCloud_PointXYZI()
        c.from_array(other_data)
        other_data = c

        # region growing2
        r_smoothness = 12
        curvature = 100
        tree = cloud.make_kdtree()
        segment = cloud.make_RegionGrowing(ksearch=50)
        segment.set_MinClusterSize(100)
        segment.set_MaxClusterSize(250000)
        segment.set_NumberOfNeighbours(30)
        segment.set_SmoothnessThreshold((r_smoothness / 180.0) * (3.14))
        segment.set_CurvatureThreshold(curvature)
        segment.set_SearchMethod(tree)
        clusters = segment.Extract()

        pcl.RegionGrowing()

        crop_cloud_index = []
        for i, index in enumerate(clusters):
            intensity = cloud_int[index, 3]
            z = cloud_int[index, 2]
            if ((150) < np.mean(intensity) and (-1) > np.mean(z) and len(clusters[i]) < 5000):  # intensity 120 kalsın
                crop_cloud_index.extend(clusters[i])
                print("len:{}".format(len(clusters[i])))

        c = pcl.PointCloud_PointXYZI()
        c.from_array(cloud_int)
        other_data = c.extract(crop_cloud_index, False)
        return other_data
