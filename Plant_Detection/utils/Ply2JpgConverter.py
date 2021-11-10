import pcl.pcl_visualization
import  pcl
import numpy as np
import os
import vtk
from skimage import exposure
from ILCC import utility
from vtk.util.numpy_support import vtk_to_numpy
from PIL import Image, ImageDraw

class Ply2JpgConverter:
    def NormalizeCloud(self, cloud_np):

        if type(cloud_np) != np.ndarray:
            cloud_np = cloud_np.to_array()
        # max_intensity = np.amax(cloud_np[:, 3])
        centred = cloud_np - np.mean(cloud_np, 0)
        cloud_np[:, :-1] = centred[:, :-1]
        return cloud_np

    def MakeMaxZero(self, cloud_np):

        if type(cloud_np) != np.ndarray:
            cloud_np = cloud_np.to_array()
        # max_intensity = np.amax(cloud_np[:, 3])
        centred = cloud_np - np.amax(cloud_np, 0)
        cloud_np[:, :-1] = centred[:, :-1]
        return cloud_np
    def EqualizeHistogram(self, cloud):
        if type(cloud) != np.ndarray:
            cloud_np = cloud.to_array()
        else:
            cloud_np = cloud
        # max_intensity = np.amax(cloud_np[:, 3])
        cloud_np[:, 3] = exposure.equalize_hist(cloud_np[:, 3]) * 255
        return cloud_np

    def ConvertPly2Jpg(self,cloud_np,cloud_path="", cloud_name="",tray_count=0):
        #actor

        # cloud_np[:, 3] = exposure.equalize_hist(cloud_np[:, 3]) *255
        if type(cloud_np)!=np.ndarray:
            cloud_np=cloud_np.to_array()
        if len(cloud_np)==0:
            cloud_np=np.concatenate((cloud_np,[[1,1,1,1]]))
        actor=utility.vis_3D_points(cloud_np,color_style="by_height")
        # actor=utility.vis_3D_points(cloud_np,color_style="intens_rg")
        actor.GetProperty().SetPointSize(3)

        # Renderer
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(1, 1, 1)
        renderer.ResetCamera()

        #camera
        camera = vtk.vtkCamera()
        # camera.SetViewUp(0.0, -1, 0.0)
        camera.SetPosition(0.0, 0.0,780) #z top
        # camera.SetPosition(0.0, 0.0,-780) #z bottom
        # camera.SetPosition(780.0,380, 0) #x1

        # camera.SetPosition(0.0, 0.0,680)

        camera.SetFocalPoint(0.0, 0.0, 0.0)
        # self.camera.SetClippingRange(0.0, 100000)

        renderer.SetActiveCamera(camera)
        # Render Window
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        # renderWindow.SetSize(432,768)
        renderWindow.SetSize(768, 432)

        # Interactor
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        # screenshot code:
        renderWindow.Render()
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(renderWindow)
        w2if.Update()

        # create directory
        # directory = cloud_path+"/pcl_jpg_intensity/"+cloud_name
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # directory += "/"+str(tray_count)+".jpg"

        # directory = cloud_path + "/pcl_jpg_intensity_leaf/"
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # directory += cloud_name+"_"+ str(tray_count) + ".jpg"
        #
        # # write jpg
        #
        # writer = vtk.vtkJPEGWriter()
        # writer.SetFileName(directory)
        # writer.SetInputConnection(w2if.GetOutputPort())
        # writer.Write()


        vtk_win_im = vtk.vtkWindowToImageFilter()
        vtk_win_im.SetInput(renderWindow)
        vtk_win_im.Update()

        vtk_image = vtk_win_im.GetOutput()

        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()

        arr = vtk_to_numpy(vtk_array).reshape(height, width, components)

        return arr

        # renderWindow.Render()
        # renderWindowInteractor.Start()

    def getRGBTuple(self,xyzi_arr):
        a = xyzi_arr[:,3].min()
        b = xyzi_arr[:,3].max()
        color_ls = []
        tmp = (xyzi_arr[:, 3] - a) / (b - a) * 255
        for k in range(xyzi_arr.shape[0]):
            # rgb_tuple = np.array([tmp[k]*0.299, tmp[k]*0.587,tmp[k]*0.144 ]).astype(np.int32)
            rgb_tuple = np.array([255 - xyzi_arr[k, 3], tmp[k], 0]).astype(np.int32)
            color_ls.append(rgb_tuple)
        return color_ls

    def ConvertPly2JpgWithBox(self,cloud_np,coordinates):
        # cloud_np[:, 3] = exposure.equalize_hist(cloud_np[:, 3]) *255
        if type(cloud_np) != np.ndarray:
            cloud_np = cloud_np.to_array()
        if len(cloud_np) == 0:
            cloud_np = np.concatenate((cloud_np, [[1, 1, 1, 1]]))
        #  actor=utility.vis_3D_points(cloud_np,color_style="by_height")
        #kutu çizimi

        # Create a PolyData
        polygonPolyData = vtk.vtkPolyData()
        lines = vtk.vtkCellArray()
        points = vtk.vtkPoints()
        for k, values in coordinates.items():
            minx = float(values[0])
            miny = float(values[1])
            maxx = float(values[2])
            maxy = float(values[3])

            points.InsertNextPoint(minx,miny, 0.0)
            points.InsertNextPoint(maxx,miny, 0.0)
            points.InsertNextPoint(maxx,maxy, 0.0)
            points.InsertNextPoint(minx,maxy, 0.0)

            line0 = vtk.vtkLine()
            line0.GetPointIds().SetId(0, k*4)
            line0.GetPointIds().SetId(1, k*4+1)

            line1 = vtk.vtkLine()
            line1.GetPointIds().SetId(0, k*4+1)
            line1.GetPointIds().SetId(1, k*4+2)

            line2 = vtk.vtkLine()
            line2.GetPointIds().SetId(0, k*4+2)
            line2.GetPointIds().SetId(1,k*4+ 3)

            line3 = vtk.vtkLine()
            line3.GetPointIds().SetId(0, k*4+3)
            line3.GetPointIds().SetId(1, k*4+0)

            lines.InsertNextCell(line0)
            lines.InsertNextCell(line1)
            lines.InsertNextCell(line2)
            lines.InsertNextCell(line3)

        polygon = vtk.vtkPolyData()
        polygon.SetPoints(points)
        polygon.SetLines(lines)

            # # Add the polygon to a list of polygons
            # polygons = vtk.vtkCellArray()
            # polygons.InsertNextCell(polygon)

            # polygonPolyData.SetLines(lines)

        polygonMapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            polygonMapper.SetInputConnection(polygon.GetProducerPort())
        else:
            polygonMapper.SetInputData(polygon)
            polygonMapper.Update()

        polygonActor = vtk.vtkActor()
        polygonActor.SetMapper(polygonMapper)
        polygonActor.GetProperty().SetColor([0.0, 0.0, 1.0])
        polygonActor.GetProperty().SetLineWidth(3)
        #normal resim başlangıcı
        actor = utility.vis_3D_points(cloud_np, color_style="intens_rg")
        actor.GetProperty().SetPointSize(3)

        # Renderer
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.AddActor(polygonActor)
        renderer.SetBackground(1, 1, 1)
        # renderer.SetBackground(0.1, 0.2, 0.4)
        renderer.ResetCamera()

        # camera
        camera = vtk.vtkCamera()
        # camera.SetViewUp(0.0, -1, 0.0)
        camera.SetPosition(0.0, 0.0, 780)
        # camera.SetPosition(0.0, 0.0,680)

        camera.SetFocalPoint(0.0, 0.0, 0.0)
        # self.camera.SetClippingRange(0.0, 100000)

        renderer.SetActiveCamera(camera)
        # Render Window
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindow.SetSize(768, 432)

        # Interactor
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        # screenshot code:
        renderWindow.Render()
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(renderWindow)
        w2if.Update()

        vtk_win_im = vtk.vtkWindowToImageFilter()
        vtk_win_im.SetInput(renderWindow)
        vtk_win_im.Update()

        vtk_image = vtk_win_im.GetOutput()

        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()

        arr = vtk_to_numpy(vtk_array).reshape(height, width, components)

        return arr

    def ConvertCoordinates(self,cloud_np,coordinates):
        width = 768
        height = 432
        w_ref = -384
        h_ref = -216

        new_coor={}
        for k, line in enumerate(coordinates):
            values = line.split(' ')
            miny = (float(values[0]) * height + h_ref)  # 768 432
            minx = (float(values[1]) * width + w_ref)
            maxy = (float(values[2]) * height + h_ref)
            maxx = (float(values[3]) * width + w_ref)

            new_coor[k]=[minx,miny,maxx,maxy]
        return new_coor

    def ConvertCoordinates2Centers(self,coordinates):
        centers={}
        for i,values in coordinates.items():
            minx=values[0]
            miny=values[1]
            maxx=values[2]
            maxy=values[3]
            centers[i]= [(minx+maxx)/2,(miny+maxy)/2]
        return centers
