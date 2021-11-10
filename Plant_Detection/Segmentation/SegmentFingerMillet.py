from Segmentation.PlaneModelSoilSeg import *
from utils.Ply2JpgConverter import *
import os
import pcl
import numpy as np

class SegmentFingerMillet():

    def __init__(self):
        globals = Globals
        # self.tray_ref_point =1133 #30c
        #1130 90c
        self.path = globals.projectPath

    def Run(self):

        data_directory = os.listdir(self.path+"/raw_data/")
        for i in data_directory:
            self.SegmentGround(self.path+"/raw_data/"+i,i)

    def SegmentGround(self,fileName,cloud_name,type="plant"):
        # soilSegm=RegionGrowCropSoilSeg()

        globals = Globals()
        fileGlobals=globals.getPlyFileGlobals(fileName)
        cloud = pcl.load_XYZI(fileName)
        tray_interval = 25
        # z_start=480
        # y_start=50
        # y_stop=5000
        # y_period=400
        # x_start=-250
        # x_stop=280

        z_start=fileGlobals["marker_z_start"]
        z_step=(z_start-fileGlobals["marker_z_stop"])/fileGlobals["y_sectors"]
        y_start=fileGlobals["field_y_origin"]
        y_stop=fileGlobals["marker_y_stop"]

        self.tray_ref_point = z_start+130  #
        # self.tray_ref_point = z_start+95  #
        cloud_name = cloud_name[:-4]
        ptfilter=cloud.make_passthrough_filter()
        ptfilter.set_filter_field_name("z")
        ptfilter.set_filter_limits(self.tray_ref_point - 300, self.tray_ref_point + tray_interval)
        cloud_tray=ptfilter.filter()

        ptfilter.set_filter_limits(z_start, self.tray_ref_point - tray_interval)
        cloud_top=ptfilter.filter()

        ptfilter.set_filter_limits(self.tray_ref_point - tray_interval - 1, self.tray_ref_point + 70)
        cloud_bottom=ptfilter.filter()

        y_limit = 200
        part_count=0
        imgs={}
        cleaned_data={}
        while y_start + 300 <y_stop:
            ptfilter=cloud_tray.make_passthrough_filter()
            ptfilter.set_filter_field_name("y")
            ptfilter.set_filter_limits(y_start - 1, y_start + 1)
            tray_temp=ptfilter.filter()
            temp = tray_temp.size
            #print("%d temp: %d"%(y_start,temp))
            if temp > y_limit:
                ptfilter = cloud_top.make_passthrough_filter()
                ptfilter.set_filter_field_name("y")
                if (y_start < 350):
                   ptfilter.set_filter_limits(y_start, y_start + 370)
                else:
                   ptfilter.set_filter_limits(y_start + 40, y_start + 430)
                cloud_top_part=ptfilter.filter()

                # bottom part border
                ptfilter=cloud_bottom.make_passthrough_filter()
                ptfilter.set_filter_field_name("y")
                if (y_start < 350):
                    ptfilter.set_filter_limits(y_start + 40, y_start + 330)
                else:
                    ptfilter.set_filter_limits(y_start + 90, y_start + 370)
                cloud_bottom_part_full=ptfilter.filter()

                cloud_np=cloud_bottom_part_full.to_array()
                x_min = np.amin(cloud_np[:, 0])
                x_max = np.amax(cloud_np[:, 0])
                x_crop_size = (abs(x_max - x_min) - 480) / 2

                ptfilter=cloud_bottom_part_full.make_passthrough_filter()
                ptfilter.set_filter_field_name("x")
                ptfilter.set_filter_limits(x_min + x_crop_size, x_max - x_crop_size)
                cloud_bottom_part_full=ptfilter.filter()

                # directory =  self.path+ '/segmentation/top_part/' + cloud_name
                # if not os.path.exists(directory):
                #     os.makedirs(directory)
                # directory += "/"+str(part_count)+".ply"
                # pcl.save(cloud_top_part,directory)
                #
                # directory =  self.path+ '/segmentation/bottom_part/' + cloud_name
                # if not os.path.exists(directory):
                #     os.makedirs(directory)
                # directory += "/"+str(part_count)+".ply"
                # pcl.save(cloud_bottom_part_full,directory)

                # if type=="plant":
                # else:
                # cloud = pcl.PointCloud_PointXYZI()
                # cloud.from_array(np.concatenate((cloud_bottom_part_full.to_array(),cloud_top_part.to_array()), axis=0))
                # crop_points=soilSegm.SegmentSoilLeaf(cloud)


                # crop_points.extend(cloud_top_part.to_list())
                # cp=pcl.PointCloud_PointXYZI()
                # cp.from_list(crop_points)



                if type == "leaf":
                    cloud_bottom_part_full.from_array(np.concatenate((cloud_bottom_part_full.to_array(), cloud_top_part.to_array()), axis=0))
                    cleaned_data[part_count] = cloud_bottom_part_full
                else:
                    cleaned_data[part_count]=cloud_bottom_part_full
                # cleaned_points=statisticalFilter.CleanData(crop_points)
                # imgs[part_count]=ply2JpgConverter.ConvertPly2Jpg(cleaned_points, self.path,cloud_name,part_count)
                # imgs.append(ply2JpgConverter.ConvertPly2Jpg(crop_points, self.path,cloud_name,part_count))

                # directory = fileName[0:-4] + '/segmented/'
                # if not os.path.exists(directory):
                #     os.makedirs(directory)
                # directory += "/" + str(part_count) + ".ply"
                # pcl.save(cloud_bottom_part_full, directory)

                part_count +=1
                self.tray_ref_point+=z_step
                if (y_start < 350):
                    y_start = y_start + 340
                else:
                    y_start = y_start + 390

            y_start+=1
        return cleaned_data
