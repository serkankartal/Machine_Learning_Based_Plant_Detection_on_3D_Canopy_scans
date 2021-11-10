from Segmentation.SegmentMungbean import *

from PIL import Image


plyConverter=Ply2JpgConverter()

globals = Globals
path = globals.projectPath + "/segmentation/euclidean_leaf_cls/"
path="C:/Users/serkan/Desktop/m_rgs/"
output_path="C:/Users/serkan/Desktop/m_rgs/output/"
statisticalFilter=StatisticalOutlierRemoval()

if not os.path.exists(output_path):
    os.makedirs(output_path)

data_directory = os.listdir(path)
for i in data_directory:
    plant_directory=os.listdir(path+"/"+i)
    for j in plant_directory:
        file_name=path+"/"+i+"/"+j
        cloud=pcl.load_XYZI(file_name)
        # cloud=plyConverter.NormalizeCloud(cloud)
        cloud = plyConverter.EqualizeHistogram(cloud)
        # cloud=statisticalFilter.CleanData(cloud,2)
        img=plyConverter.ConvertPly2Jpg(cloud,output_path,i+"_",j[:-4])

        im = Image.fromarray(img)
        im.save(output_path + '/' + i + "_" + j[:-4]+ '.jpg')

# data_directory = os.listdir(path)
# for i in data_directory:
#     plant_directory=os.listdir(path+"/"+i)
#     for j in plant_directory:
#         plant_directory_j = os.listdir(path + "/" + i+"/"+j)
#         for k in plant_directory_j:
#             file_name=path+"/"+i+"/"+j+"/"+k
#             cloud=pcl.load_XYZI(file_name)
#             plyConverter.ConvertPly2Jpg(cloud,globals.projectPath,i+"_"+j,k[:-4])