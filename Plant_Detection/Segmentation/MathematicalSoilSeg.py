import pcl
import numpy as np

class MathematicalSoilSeg:
    def SegmentSoil(self,cloud, bottom=-20,xside=20,yside=20,max_height=220):
        if type(cloud) != np.ndarray:
            cloud_np = cloud.to_array()
        else:
            cloud_np=cloud

        max_values=np.amax(cloud_np[:,:],axis=0)
        min_values=np.amin(cloud_np[:,:],axis=0)

        indexList = np.where((cloud_np[:, 2] < bottom))
        # indexList = np.where( (cloud_np[:, 2] > (max_values[2]-max_height)) & (cloud_np[:, 2] < bottom) &
        #          (cloud_np[:, 1] > (min_values[1]-yside)) & (cloud_np[:, 1] < (max_values[1]-yside)) &
        #           (cloud_np[:, 0] > (min_values[0]-xside)) & (cloud_np[:, 0] < (max_values[0]-xside)))


        cloud_temp=pcl.PointCloud_PointXYZI()
        cloud_temp.from_array(cloud_np[indexList[0],:])
        return cloud_temp
           # indexList[0][0]




