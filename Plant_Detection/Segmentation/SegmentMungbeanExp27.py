from Segmentation.RegionGrowSoilSeg import *
from Segmentation.PlaneModelSoilSeg import *
from utils.Ply2JpgConverter import *
import os
import pcl
import numpy as np

class SegmentMungbeanExp27():

    def __init__(self):
        globals = Globals
        self.path = globals.projectPath

    def Run(self):
        data_directory = os.listdir(self.path+"/raw_data/")
        for i in data_directory:
            self.SegmentGround(self.path+"/raw_data/"+i,i)

    def SegmentGround(self,fileName,cloud_name,type="plant"):
        soilSegm=RegionGrowCropSoilSeg()
        # ply2JpgConverter=Ply2JpgConverter()
        # statisticalFilter=StatisticalOutlierRemoval()
        self.tray_ref_point = 935  #
        cloud = pcl.load_XYZI(fileName)
        tray_interval = 20
        top_part_interval = 50
        z_start=450
        y_start=10
        y_stop=5000
        x_start=-300
        x_stop=300
        # z_heiht_filter=60 #plant detection yapılacaksa 60, yaprak detection yapılacasksa çok büyük olmalı

        cloud_name = cloud_name[:-4]
        ptfilter=cloud.make_passthrough_filter()
        ptfilter.set_filter_field_name("z")
        ptfilter.set_filter_limits(self.tray_ref_point - tray_interval, self.tray_ref_point + tray_interval)
        cloud_tray=ptfilter.filter()

        ptfilter.set_filter_limits(z_start, self.tray_ref_point - tray_interval)
        cloud_top=ptfilter.filter()

        ptfilter.set_filter_limits(self.tray_ref_point - tray_interval - 1, self.tray_ref_point + 70)
        cloud_bottom=ptfilter.filter()

        y_limit = 200
        part_count=0
        imgs={}
        cleaned_data={}
        while y_start + 200 <y_stop:
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
                   ptfilter.set_filter_limits(y_start, y_start + 400)
                else:
                   ptfilter.set_filter_limits(y_start + 30, y_start + 430)
                cloud_top_part=ptfilter.filter()

                # bottom part border
                ptfilter=cloud_bottom.make_passthrough_filter()
                ptfilter.set_filter_field_name("y")
                if (y_start < 350):
                    ptfilter.set_filter_limits(y_start + 40, y_start + 360)
                else:
                    ptfilter.set_filter_limits(y_start + 70, y_start + 390)
                cloud_bottom_part_full=ptfilter.filter()
                x_crop_size = 30

                ptfilter=cloud_bottom_part_full.make_passthrough_filter()
                ptfilter.set_filter_field_name("x")
                ptfilter.set_filter_limits(x_start + x_crop_size, x_stop - x_crop_size)
                cloud_bottom_part_full=ptfilter.filter()


                if type == "leaf":
                    cloud_bottom_part_full.from_array(np.concatenate((cloud_bottom_part_full.to_array(), cloud_top_part.to_array()), axis=0))


                cleaned_data[part_count]=cloud_bottom_part_full
                part_count +=1
                if (y_start < 350):
                    y_start = y_start + 350
                else:
                    y_start = y_start + 380

            y_start+=1
        return cleaned_data

    def SegmentGroundYstep(self,fileName,cloud_name,type="plant"):
        # soilSegm=RegionGrowCropSoilSeg()

        globals = Globals()
        fileGlobals = globals.getPlyFileGlobals(fileName)
        cloud = pcl.load_XYZI(fileName)
        tray_interval = 15

        y_sector_count = fileGlobals["y_sectors"]
        z_start = fileGlobals["marker_z_start"]
        z_step = (z_start - fileGlobals["marker_z_stop"]) / y_sector_count
        y_start = fileGlobals["field_y_origin"]
        y_stop = fileGlobals["marker_y_stop"]
        y_step = fileGlobals["field_y_period"]

        self.tray_ref_point = z_start + 235  #
        ptfilter = cloud.make_passthrough_filter()
        ptfilter.set_filter_field_name("z")
        ptfilter.set_filter_limits(z_start - 300, self.tray_ref_point - tray_interval)
        cloud_top = ptfilter.filter()

        ptfilter.set_filter_limits(self.tray_ref_point + tray_interval - 1, self.tray_ref_point + 70)
        cloud_bottom = ptfilter.filter()

        ptfilter = cloud.make_passthrough_filter()
        ptfilter.set_filter_field_name("z")
        ptfilter.set_filter_limits(self.tray_ref_point - tray_interval, self.tray_ref_point + tray_interval)
        cloud_tray = ptfilter.filter()

        part_count = 0
        cleaned_data = {}
        while part_count < y_sector_count:
            ptfilter = cloud_top.make_passthrough_filter()
            ptfilter.set_filter_field_name("y")
            ptfilter.set_filter_limits(y_start, y_start + y_step)
            cloud_top_part = ptfilter.filter()

            # bottom part border
            ptfilter = cloud_bottom.make_passthrough_filter()
            ptfilter.set_filter_field_name("y")
            ptfilter.set_filter_limits(y_start, y_start + y_step)
            cloud_bottom_part_full = ptfilter.filter()

            ptfilter = cloud_tray.make_passthrough_filter()
            ptfilter.set_filter_field_name("y")
            ptfilter.set_filter_limits(y_start + 60, y_start + y_step - 60)
            cloud_tray_part = ptfilter.filter()

            cloud_np = cloud_tray_part.to_array()
            x_min = np.amin(cloud_np[:, 0])
            x_max = np.amax(cloud_np[:, 0])
            # x_crop_size = (abs(x_max - x_min) - 480) / 2

            ptfilter = cloud_tray_part.make_passthrough_filter()
            ptfilter.set_filter_field_name("x")
            ptfilter.set_filter_limits(x_min + 25, x_max - 25)
            cloud_tray_part = ptfilter.filter()

            cloud_bottom_part_full.from_array(
                np.concatenate((cloud_bottom_part_full.to_array(), cloud_top_part.to_array()), axis=0))
            cloud_bottom_part_full.from_array(
                np.concatenate((cloud_bottom_part_full.to_array(), cloud_tray_part.to_array()), axis=0))
            cleaned_data[part_count] = cloud_bottom_part_full

            part_count += 1
            y_start = y_start + y_step
            print("ystart: {}".format(y_start))

        return cleaned_data