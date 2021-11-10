import os
import pcl
from pcl import StatisticalOutlierRemovalFilter_PointXYZI
import numpy as np

class StatisticalOutlierRemoval():

    def CleanData(self, cloud,threshold=1):
        if type(cloud)==np.ndarray:
            cloud_np=pcl.PointCloud_PointXYZI()
            cloud_np.from_array(cloud)
            cloud=cloud_np
        statistical_filter=pcl.StatisticalOutlierRemovalFilter_PointXYZI(cloud)
        statistical_filter.set_mean_k(50)
        statistical_filter.set_std_dev_mul_thresh(threshold)
        return statistical_filter.filter()