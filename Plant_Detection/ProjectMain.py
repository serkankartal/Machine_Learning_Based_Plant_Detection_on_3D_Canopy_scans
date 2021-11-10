from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QImage  # icon and load image
from PyQt5.QtCore import Qt
from Segmentation.LeafDetection import *
from Segmentation.CombineSoilSeg import *
from Segmentation.SegmentFingerMillet import *
from Segmentation.SegmentMungbeanExp27 import *
from Segmentation.SegmentChickPeaExp33 import *
from Segmentation.StatisticalOutlierRemoval import *
import matplotlib.cm as cm
from Segmentation.MathematicalSoilSeg import *
from PIL import Image
from ObjectDetectionFolder import *
from Clustering.Clustering import *
import matplotlib.pyplot as plt
import cv2
import time

class Window(QMainWindow):

    def __init__(self):
        super().__init__()
        random.seed(1)
        self.regionGrowSoilSegm = RegionGrowCropSoilSeg()
        self.planeModelSoilSegm = PlaneModelSoilSeg()
        self.combineSoilSeg = CombineSoilSeg()
        self.mathematicalSoilSeg = MathematicalSoilSeg()
        globals = Globals
        self.path = globals.projectPath
        # main window
        self.width = 1200
        self.height = 640

        self.setWindowTitle("Plant Detection")
        self.setGeometry(50,100,self.width, self.height)
        self.setWindowIcon(QIcon("icon1.png"))
        #
        self.tabWidget()
        self.widgets()
        self.layouts()
        self.show()

    def tabWidget(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab1 = QWidget()
        self.tab2 = QWidget()

        self.tabs.addTab(self.tab1, "Classification")
        self.tabs.addTab(self.tab2, "Parameters")

    def widgets(self):

        #Input data widgets
        self.cbDataType=QComboBox()
        self.cbDataType.addItems(["Mungbean","MungbeanExp27","FingerMillet","ChickpeaExp33","Sorghum2019","Sorghum2017"])

        self.inputFolderHbox=QHBoxLayout()
        self.lblInputFolder=QLabel("Enter Input Folder Path Below OR")
        self.btnInputFolder=QPushButton("Select From dialog Box")
        self.btnInputFolder.clicked.connect(self.selectInputFolder)
        self.inputFolderHbox.addWidget(self.lblInputFolder)
        self.inputFolderHbox.addWidget(self.btnInputFolder)
        self.txtInputFolder=QLineEdit()


        self.saveResultLayout=QHBoxLayout()
        self.cbSaveResultImages=QCheckBox("Save Result Image",self)
        self.cbSaveResultImages.setChecked(True)
        self.btnTest=QPushButton("Start Test")
        self.btnTest.clicked.connect(self.startTestFunction)
        self.saveResultLayout.addWidget(self.cbSaveResultImages)
        self.saveResultLayout.addWidget(self.btnTest)


        self.processButtonsLayout=QHBoxLayout()
        self.btnPlantDetection=QPushButton("Start Plant Detection")
        self.btnPlantDetection.clicked.connect(self.startPlantDetection)
        self.btnLeafDetection=QPushButton("Start Leaf Detection")
        self.btnLeafDetection.clicked.connect(self.startLeafDetection)
        self.processButtonsLayout.addWidget(self.btnPlantDetection)
        self.processButtonsLayout.addWidget(self.btnLeafDetection)

        self.clusteringLayout=QHBoxLayout()
        self.cbClusterType=QComboBox()
        self.cbClusterType.addItems(["Kmeans","Spectral","Hierarchial","Gaussian","Optics"])
        self.clusteringLayout.addWidget(self.cbClusterType)
        self.btnCluster=QPushButton("Start Clustering")
        self.btnCluster.clicked.connect(self.startClustering)
        self.clusteringLayout.addWidget(self.btnCluster)

        self.growingAnalyseLayout = QHBoxLayout()
        self.btnGrowingAnalyse = QPushButton("Start Growing Analyse")
        self.btnGrowingAnalyse.clicked.connect(self.startGrowingAnalyse)
        self.growingAnalyseLayout.addWidget(self.btnGrowingAnalyse)

        self.btnTestDataCounting = QPushButton("Test data counting")
        self.btnTestDataCounting.clicked.connect(self.startTestDataCounting)
        self.growingAnalyseLayout.addWidget(self.btnTestDataCounting)

        self.listinputDataList = QListWidget()
        self.listinputDataList.selectionModel().selectionChanged.connect(self.selectedInputDataChanged)

        #output images widgets

        #output text widgets

        self.lblOutputFolder=QLabel("Enter Output Folder Path: ")
        self.txtOutputFolder=QLineEdit()

    def startTestDataCounting(self):
        inputFolder = self.txtInputFolder.text()
        outputFolder = self.txtOutputFolder.text()
        saveImage = self.cbSaveResultImages.isChecked()
        dataType = self.cbDataType.currentText()
        objectType = dataType

        objectDetection = ObjectDetectionFolder(objectType + "_plant")
        excelt_data = {}
        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().deleteLater()

        cleaned_imgs = {}
        for fileCount, img_file in enumerate(self.inputFileNames):
            # img=Image.open(inputFolder+"//"+img_file)
            img=cv2.imread(inputFolder+"//"+img_file)
            # cleaned_imgs[img_file]=np.asarray(img)
            cleaned_imgs[img_file] =img
        imgs, counts = objectDetection.Detect(cleaned_imgs)
        plant_counts = {}

        if saveImage:
            if not os.path.exists(outputFolder):
                os.makedirs(outputFolder)

        for count, img in imgs.items():
            # cv2.imshow("resim",img)
            # im = Image.fromarray(img)
            # cv2.imshow("resim2",im)

            if saveImage:
                # im.save(outputFolder + '/' + str(count))
                cv2.imwrite(outputFolder + '/' + str(count),img)


            labelstr=QLabel(str(count))
            label = QLabel()
            qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap(qimage)
            pixmap = pixmap.scaled(384, 216, Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
            self.vbox.addWidget(labelstr)
            self.vbox.addWidget(label)
            # plant_counts[count] = len(counts[count])
            excelt_data[count] = len(counts[count])

        self.writeTestCountingData2ExcelAll(excelt_data)
        self.showTestCountingDataOnTable(excelt_data)

    def startGrowingAnalyse(self):
        inputFolder = self.txtInputFolder.text()
        outputFolder = self.txtOutputFolder.text()
        saveImage = self.cbSaveResultImages.isChecked()
        dataType = self.cbDataType.currentText()

        leafDetection=LeafDetection(dataType+"_leaf")

        tray_count=12
        if  dataType == "MungbeanExp27":
            tray_count=8

        ply2JpgConverter = Ply2JpgConverter()
        statisticalFilter = StatisticalOutlierRemoval()
        excelt_data_counts = {}
        excelt_data_heights = {}
        excelt_data_biomass = {}

        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().deleteLater()

        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        for fileCount, i in enumerate(self.inputFileNames):
            cleaned_points = self.SegmentData(inputFolder + "/" + i, i,"leaf")
            # self.saveClouds(cleaned_points,i,outputFolder)
            cleaned_imgs = {}
            average_heights = {}
            biomass = {}
            for part_count, crop_points in cleaned_points.items():

                cloud_bottom_part = ply2JpgConverter.MakeMaxZero(crop_points)
                cloud_bottom_part = ply2JpgConverter.EqualizeHistogram(cloud_bottom_part)  # normalde bunun olması lazım mathematicalde kaldırıldı bir alttaki fonksiyonla beraber
                # self.saveCloud(cloud_bottom_part, i, outputFolder,part_count)
                cloud_bottom_part = self.mathematicalSoilSeg.SegmentSoil(cloud_bottom_part,bottom=-50)
                # cloud_bottom_part = self.combineSoilSeg.SegmentSoilLeafBottom(cloud_bottom_part,inputFolder + "/" + i, 15, 12)
                # cloud_bottom_part = self.regionGrowSoilSegm.SegmentBigClusters(cloud_bottom_part, 8,5000)
                cloud_bottom_part = ply2JpgConverter.EqualizeHistogram(cloud_bottom_part)
                cleaned_points[part_count] = cloud_bottom_part

            for part_count, crop_points in cleaned_points.items():
                cleaned_points[part_count] = statisticalFilter.CleanData(crop_points, 1)
                # self.saveCloud(cleaned_points[part_count], i, outputFolder,part_count)
                average_heights[part_count]=abs(self.getAverageHeight(cleaned_points[part_count]))
                cleaned_points[part_count]  = ply2JpgConverter.NormalizeCloud(cleaned_points[part_count] )
                cleaned_imgs[part_count] = ply2JpgConverter.ConvertPly2Jpg(cleaned_points[part_count] , self.path, i, part_count)
                biomass[part_count]= self.getBiomass(cleaned_imgs[part_count])

            imgs,counts=leafDetection.Detect(cleaned_points)
            # imgs=cleaned_imgs
            plant_counts={}
            if fileCount == 0:
                label = QLabel("File: " + i[0:-4])
                self.vbox.addWidget(label)
                self.trayCount = len(imgs)
                self.fileCount = len(self.inputFileNames)
            if saveImage:
                f = outputFolder + "/" + i[0:-4]
                if not os.path.exists(f):
                    os.makedirs(f)
            for count, img in imgs.items():
                im = Image.fromarray(img)
                if saveImage:
                    im.save(outputFolder + '/' + i[0:-4] + "/" + str(count) + '.jpg')
                if fileCount == 0:
                    label = QLabel()
                    qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
                    pixmap = QPixmap(qimage)
                    pixmap = pixmap.scaled(384, 216, Qt.KeepAspectRatio)
                    label.setPixmap(pixmap)
                    self.vbox.addWidget(label)
                plant_counts[count] = len(counts[count])

            excelt_data_counts[i[0:-4]] = plant_counts
            excelt_data_biomass[i[0:-4]] = biomass
            excelt_data_heights[i[0:-4]] = average_heights
        self.writeGrowingData2Excel(excelt_data_counts,excelt_data_biomass,excelt_data_heights)
        self.showDataOnTable(excelt_data_counts)

    def getAverageHeight(self, cloud):
        if type(cloud) != np.ndarray:
            cloud = cloud.to_array()
        mean_z =np.mean(cloud[:,2])
        return mean_z

    def getBiomass(self,img):
        binary_img=cv2.threshold(img,100,255,cv2.THRESH_BINARY)
        black_pixsel_count=  np.sum(binary_img[1]<1)/3
        # cv2.imshow("kara",binary_img[1])
        return black_pixsel_count
        # return binary_img[1]

    def startClustering(self):
        clsType = self.cbClusterType.currentText()
        inputFolder = self.txtInputFolder.text()
        outputFolder = self.txtOutputFolder.text()
        saveImage = self.cbSaveResultImages.isChecked()
        ply2JpgConverter = Ply2JpgConverter()
        clustering = Clustering()
        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().deleteLater()
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        fileName_txt = "plant_counts.txt"
        text_file = open(inputFolder + "/" + fileName_txt, "r")
        plant_counts = text_file.readlines()
        colored_imgs = {}
        for k, i in enumerate(self.inputFileNames):
            cloud = pcl.load_XYZI(inputFolder + "/" + i)

            if clsType == "Kmeans":
                clusters = clustering.KmeansCluster(cloud,int(plant_counts[k]))
            elif clsType == "Spectral":
                clusters = clustering.SpectralCluster(cloud, int(plant_counts[k]))
            elif clsType == "Hierarchial":
                clusters = clustering.HierarchialCluster(cloud,int(plant_counts[k]))
            elif clsType == "Gaussian":
                clusters = clustering.GaussianCluster(cloud, int(plant_counts[k]))
            elif clsType == "Optics":
                clusters = clustering.OpticsCluster(cloud,int(plant_counts[k]))
            cluster1=self.changeClusterNumberAcc2Center(clusters)
            c_cloud = self.ColoringClouds2(cluster1)
            # self.saveCloud(c_cloud, i, outputFolder + '/' + clsType , int(i[0:-4]))
            colored_imgs[int(i[0:-4])] = ply2JpgConverter.ConvertPly2Jpg(c_cloud, self.path, i,  int(i[0:-4]))

        if saveImage:
            f = outputFolder + "/" + clsType
            if not os.path.exists(f):
                os.makedirs(f)
        for count, img in colored_imgs.items():
            im = Image.fromarray(img)
            if saveImage:
                im.save(outputFolder + '/' + clsType + "/" + str(count) + '.jpg')
            labelstr = QLabel(str(count))
            self.vbox.addWidget(labelstr)

            label = QLabel()
            qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap(qimage)
            pixmap = pixmap.scaled(384, 216, Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
            self.vbox.addWidget(label)

    def selectedInputDataChanged(self):
        selectedFile = self.listinputDataList.currentItem().text()
        folderName = self.txtOutputFolder.text() + "/" + selectedFile[0:-4]+".jpg"

        if not os.path.exists(folderName):
            return

        data_directory = os.listdir(folderName)


        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().deleteLater()

        label = QLabel("File: "+selectedFile[0:-4])

        self.vbox.addWidget(label)
        imgs={}
        for file in data_directory:
            pixmap = QPixmap(folderName+"/"+file)
            imgs[file[0:-4]]=pixmap
        if len(imgs)==0:
            return

        sector = int(selectedFile[4:6])
        trench = int(selectedFile[1:3])
        for i,img in enumerate(imgs.values()):
            label = QLabel()

            labelstr=QLabel(str(i+1)+"___"+str(trench + 1) + "_" + str(i + 1 + (sector - 1) * 12))
            pixmap=imgs[str(i)]
            pixmap = pixmap.scaled(384, 216, Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
            self.vbox.addWidget(labelstr)
            self.vbox.addWidget(label)
        # self.resultTable.selectRow(self.listinputDataList.currentRow()+1)

    def startLeafDetection(self):
        inputFolder = self.txtInputFolder.text()
        outputFolder = self.txtOutputFolder.text()
        saveImage = self.cbSaveResultImages.isChecked()
        dataType = self.cbDataType.currentText()
        leafDetection=LeafDetection(dataType+"_leaf")

        tray_count=12
        if  dataType == "MungbeanExp27":
            tray_count=8
        if dataType.find("Mungbean") > -1:
            objectType="Mungbean"

        ply2JpgConverter = Ply2JpgConverter()
        statisticalFilter = StatisticalOutlierRemoval()
        excelt_data = {}

        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().deleteLater()

        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        for fileCount, i in enumerate(self.inputFileNames):
            cleaned_points = self.SegmentData(inputFolder + "/" + i, i,"leaf")
            # self.saveClouds(cleaned_points,i,outputFolder)
            cleaned_imgs = {}
            for part_count, crop_points in cleaned_points.items():

                cloud_bottom_part = ply2JpgConverter.NormalizeCloud(crop_points)
                cloud_bottom_part = ply2JpgConverter.EqualizeHistogram(
                    cloud_bottom_part)  # normalde bunun olması lazım mathematicalde kaldırıldı bir alttaki fonksiyonla beraber
                if dataType == "MungbeanExp27":
                    cloud_bottom_part = self.combineSoilSeg.SegmentSoilLeaf(cloud_bottom_part, 15, 14)
                else:
                    cloud_bottom_part = self.combineSoilSeg.SegmentSoilLeaf(cloud_bottom_part, 15, 12)

                cloud_bottom_part = self.regionGrowSoilSegm.SegmentBigClusters(cloud_bottom_part, 6, 2000)
                cloud_bottom_part = ply2JpgConverter.EqualizeHistogram(cloud_bottom_part)
                cleaned_points[part_count] = cloud_bottom_part

            for part_count, crop_points in cleaned_points.items():
                cleaned_points[part_count] = statisticalFilter.CleanData(crop_points, 1)
                cleaned_points[part_count]  = ply2JpgConverter.NormalizeCloud(cleaned_points[part_count] )
                cleaned_imgs[part_count] = ply2JpgConverter.ConvertPly2Jpg(cleaned_points[part_count] , self.path, i, part_count)

            imgs,counts=leafDetection.Detect(cleaned_points)
            plant_counts={}
            if fileCount == 0:
                label = QLabel("File: " + i[0:-4])
                self.vbox.addWidget(label)
                self.trayCount = len(imgs)
                self.fileCount = len(self.inputFileNames)
            if saveImage:
                f = outputFolder + "/" + i[0:-4]
                if not os.path.exists(f):
                    os.makedirs(f)
            for count, img in imgs.items():
                im = Image.fromarray(img)
                if saveImage:
                    im.save(outputFolder + '/' + i[0:-4] + "/" + str(count) + '.jpg')
                if fileCount == 0:
                    label = QLabel()
                    qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
                    pixmap = QPixmap(qimage)
                    pixmap = pixmap.scaled(384, 216, Qt.KeepAspectRatio)
                    label.setPixmap(pixmap)
                    self.vbox.addWidget(label)
                    plant_counts[count] = len(counts[count])

            excelt_data[i[0:-4]] = plant_counts
        self.writeData2Excel(excelt_data)
        self.showDataOnTable(excelt_data)

    def startPlantDetection(self):
        inputFolder = self.txtInputFolder.text()
        outputFolder = self.txtOutputFolder.text()
        saveImage = self.cbSaveResultImages.isChecked()
        dataType = self.cbDataType.currentText()
        objectType=dataType
        tray_count=12
        if dataType == "MungbeanExp27":
            tray_count=8
        if dataType.find("Mungbean") > -1:
            objectType="Mungbean"

        objectDetection = ObjectDetectionFolder(objectType + "_plant")
        ply2JpgConverter = Ply2JpgConverter()
        statisticalFilter = StatisticalOutlierRemoval()
        excelt_data = {}

        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().deleteLater()
        start_time = time.time()
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        for fileCount, i in enumerate(self.inputFileNames):
            print("0000")
            cleaned_points = self.SegmentData(inputFolder + "/" + i, i, "leaf")
            cleaned_imgs = {}
            for part_count, crop_points in cleaned_points.items():
                # self.saveCloud(crop_points, i, outputFolder, part_count)
                cloud_bottom_part = ply2JpgConverter.NormalizeCloud(crop_points)
                cloud_bottom_part = ply2JpgConverter.EqualizeHistogram(
                    cloud_bottom_part)  # normalde bunun olması lazım mathematicalde kaldırıldı bir alttaki fonksiyonla beraber
                if dataType == "MungbeanExp27":
                    cloud_bottom_part = self.combineSoilSeg.SegmentSoilLeaf(cloud_bottom_part, 15, 14)
                else:
                    cloud_bottom_part = self.combineSoilSeg.SegmentSoilLeaf(cloud_bottom_part, 15, 12)

                cloud_bottom_part = self.regionGrowSoilSegm.SegmentBigClusters(cloud_bottom_part, 6, 2000)
                cloud_bottom_part = ply2JpgConverter.EqualizeHistogram(cloud_bottom_part)
                cleaned_points[part_count] = cloud_bottom_part
            for part_count, crop_points in cleaned_points.items():
                if dataType == "Mungbean" or dataType == "MungbeanExp27":
                    cleaned_points = statisticalFilter.CleanData(crop_points, 1)
                elif dataType == "FingerMillet":
                    cleaned_points = statisticalFilter.CleanData(crop_points, 2)
                elif dataType == "ChickpeaExp33":
                    cleaned_points = statisticalFilter.CleanData(crop_points, 1)
                # segment point less then z -150 to get better result for heigt coloring
                cleaned_points = self.FiltterTopNoisyDataByHeight(cleaned_points, -150)
                cleaned_points = ply2JpgConverter.NormalizeCloud(cleaned_points)
                cleaned_imgs[part_count] = ply2JpgConverter.ConvertPly2Jpg(cleaned_points, self.path, i, part_count)
            print("111")
                # self.saveCloud(cleaned_points, i, outputFolder, part_count)
            imgs, counts = objectDetection.Detect(cleaned_imgs)
            print("222")
            # imgs=cleaned_imgs
            plant_counts = {}
            if fileCount == 0:
                label = QLabel("File: " + i[0:-4])
                self.vbox.addWidget(label)
                self.trayCount = len(cleaned_imgs)
                self.fileCount = len(self.inputFileNames)
            if saveImage:
                f = outputFolder + "/" + i[0:-4]
                if not os.path.exists(f):
                    os.makedirs(f)

                f = outputFolder + "/images"
                if not os.path.exists(f):
                    os.makedirs(f)
            if len(cleaned_imgs) == 0:
                return
            print("333")
            for count, img in cleaned_imgs.items():
                im = Image.fromarray(img)
                if saveImage:
                    # im.save(outputFolder+'/'+i[0:-4]+"_"+str(count)+'.jpg')
                    im.save(outputFolder + '/' + i[0:-4] + "/" + str(count) + '.jpg')
                    sector = int(i[4:6])
                    trench = int(i[1:3])
                    im.save(outputFolder + '/images/' +str(trench+1)+"_"+str(count + 1+(sector-1)*tray_count) + '.jpg')
                if fileCount == 0:
                    label = QLabel()
                    qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
                    pixmap = QPixmap(qimage)
                    pixmap = pixmap.scaled(384, 216, Qt.KeepAspectRatio)
                    label.setPixmap(pixmap)
                    self.vbox.addWidget(label)
                plant_counts[count] = len(counts[count])
            excelt_data[i[0:-4]] = plant_counts
            print("444")

        print("555")
        print("--- %s seconds ---" % (time.time() - start_time))
        self.writeData2ExcelAll(excelt_data)
        self.showDataOnTable(excelt_data)

    def SegmentData(self,fileName,cloud_name,type="plant"):
        dataType = self.cbDataType.currentText()

        if dataType=="Mungbean":
            self.segmentMungbean=SegmentMungbean(990)
            return self.segmentMungbean.SegmentGround(fileName,cloud_name,type)
        elif dataType =="FingerMillet":
            self.segmentFingerMillet=SegmentFingerMillet()
            return self.segmentFingerMillet.SegmentGround(fileName,cloud_name,type)
        elif dataType =="ChickpeaExp33":
            self.SegmentChickPeaExp33=SegmentChickPeaExp33()
            return self.SegmentChickPeaExp33.SegmentGround(fileName,cloud_name,type)
        elif dataType =="MungbeanExp27":
            self.SegmentMungbeanExp27=SegmentMungbeanExp27()
            return self.SegmentMungbeanExp27.SegmentGroundYstep(fileName,cloud_name,type)

    def showDataOnTable(self,data):
        self.resultTable = QTableWidget()
        self.resultTable.setColumnCount(self.trayCount+1)
        self.resultTable.setRowCount(self.fileCount+1)
        excel_column_name = ["0.Tray","1.Tray", "2.Tray", "3.Tray", "4.Tray", "5.Tray", "6.Tray", "7.Tray", "8.Tray", "9.Tray", "10.Tray", "11.Tray"]
        self.resultTable.setHorizontalHeaderItem(0, QTableWidgetItem("File Name"))


        for i in reversed(range(self.hbox.count())):
            self.hbox.itemAt(i).widget().deleteLater()

        for i in range(0, 12):
            self.resultTable.setHorizontalHeaderItem(i+1, QTableWidgetItem(excel_column_name[i]))
        row=1
        for fileName, plantCounts in data.items():
            self.resultTable.setItem(row, 0, QTableWidgetItem(fileName))
            for tray_count, plant_count in plantCounts.items():
                self.resultTable.setItem(row, (int(tray_count) + 1), QTableWidgetItem(str(plant_count)))
            row = row + 1
        self.hbox.addWidget(self.resultTable)

    def showTestCountingDataOnTable(self,data):
        self.resultTable = QTableWidget()
        self.resultTable.setColumnCount(2)
        self.resultTable.setRowCount(len(data)+1)
        excel_column_name = ["File Name","Plant Count"]

        for i in reversed(range(self.hbox.count())):
            self.hbox.itemAt(i).widget().deleteLater()

        for i in range(0, 2):
            self.resultTable.setHorizontalHeaderItem(i, QTableWidgetItem(excel_column_name[i]))
        row=1
        for fileName, plantCount in data.items():
            self.resultTable.setItem(row, 0, QTableWidgetItem(fileName[:-4]))
            self.resultTable.setItem(row, 1, QTableWidgetItem(str(plantCount)))
            row = row + 1
        self.hbox.addWidget(self.resultTable)

    def writeGrowingData2Excel(self,excelt_data_counts,excelt_data_biomass,excelt_data_heights):
        outputFolder = self.txtOutputFolder.text()
        workbook = xlsxwriter.Workbook(outputFolder + '/PlantCount.xlsx')
        worksheet = workbook.add_worksheet()

        worksheet.write(0, 1, "Sector")
        excel_row = 1
        file_count=0
        tray_len=12

        worksheet.write(0, 0, "Leaf Count")
        for fileName, plantCounts in excelt_data_counts.items():
            worksheet.write(0, file_count+2, fileName)
            excel_row = 1
            for tray_count, plant_count in plantCounts.items():
                if file_count==0:
                    tray_len=len(plantCounts)
                    worksheet.write(excel_row, 1, str(int(tray_count) + 1))
                worksheet.write(excel_row,file_count+ 2, plant_count)
                excel_row = excel_row + 1
            file_count=file_count+1

        file_count=0
        worksheet.write(tray_len+2, 0, "Biomass")
        for fileName, plantCounts in excelt_data_biomass.items():
            excel_row = tray_len+2
            for tray_count, plant_count in plantCounts.items():
                if file_count==0:
                    worksheet.write(excel_row, 1, str(int(tray_count) + 1))
                worksheet.write(excel_row, file_count+2, plant_count)
                excel_row = excel_row + 1
            file_count=file_count+1

        file_count=0
        worksheet.write((tray_len+2)*2, 0, "Average Heights")
        for fileName, plantCounts in excelt_data_heights.items():
            excel_row = (tray_len+2)*2
            for tray_count, plant_count in plantCounts.items():
                if file_count==0:
                    worksheet.write(excel_row, 1, str(int(tray_count) + 1))
                worksheet.write(excel_row, file_count+2, plant_count)
                excel_row = excel_row + 1
            file_count=file_count+1

        workbook.close()

    def writeData2Excel(self,data):
        outputFolder=self.txtOutputFolder.text()
        excel_column_name = ["0.Tray","1.Tray", "2.Tray", "3.Tray", "4.Tray", "5.Tray", "6.Tray", "7.Tray", "8.Tray", "9.Tray", "10.Tray", "11.Tray"]
        workbook = xlsxwriter.Workbook(outputFolder + '/PlantCount.xlsx')
        worksheet = workbook.add_worksheet()
        for i in range(0,12):
            worksheet.write(0, i+1, excel_column_name[i])
        excel_row=1
        for fileName,plantCounts in data.items():
            worksheet.write(excel_row,0, fileName)
            for tray_count, plant_count in plantCounts.items():
                worksheet.write(excel_row,(int(tray_count)+1),plant_count)
            excel_row=excel_row+1
        workbook.close()

    def writeTestCountingData2ExcelAll(self, data):
        outputFolder = self.txtOutputFolder.text()
        excel_column_name = ["FileName", "Plant Count"]
        workbook = xlsxwriter.Workbook(outputFolder + '/PlantCount.xlsx')
        worksheet = workbook.add_worksheet()
        for i in range(0, 2):
            worksheet.write(0, i, excel_column_name[i])
        excel_row = 1
        for fileName, plantCounts in data.items():
            worksheet.write(excel_row, 0, fileName[:-4])
            worksheet.write(excel_row, 1, plantCounts)
            excel_row = excel_row + 1
        workbook.close()

    def writeData2ExcelAll(self, data):
        outputFolder = self.txtOutputFolder.text()
        excel_column_name = ["Sector", "Plant Count"]
        workbook = xlsxwriter.Workbook(outputFolder + '/PlantCount.xlsx')
        worksheet = workbook.add_worksheet()
        for i in range(0, 2):
            worksheet.write(0, i + 1, excel_column_name[i])
        excel_row = 1
        for fileName, plantCounts in data.items():
            sector = int(fileName[4:6])
            trench = int(fileName[1:3])
            worksheet.write(excel_row, 0, fileName)
            for tray_count, plant_count in plantCounts.items():
                worksheet.write(excel_row, 1, str(trench + 1) + "-" + str(int(tray_count) + 1 + (sector - 1) * 12))
                worksheet.write(excel_row, 2, plant_count)
                excel_row = excel_row + 1
        workbook.close()

    def selectInputFolder(self):
        fileDialog=QFileDialog()
        fileDialog.setFileMode(QFileDialog.DirectoryOnly)
        # dialog.setFileMode(QFileDialog::Directory);
        if fileDialog.exec_():
            self.folderName = fileDialog.selectedFiles()[0]
        print(self.folderName)
        self.txtInputFolder.setText(self.folderName)
        self.txtOutputFolder.setText(self.folderName+"/output")
        self.inputFileNames=self.readInputFileName()
        # if self.listinputDataList.selectedItems():
        #     self.listinputDataList.clearSelection()
        #     self.listinputDataList.clearFocus()
        self.listinputDataList.clear()
        self.listinputDataList.addItems(self.inputFileNames)

    def layouts(self):

        # tab1 layout
        self.mainLayout = QHBoxLayout()
        self.dataLayout = QFormLayout()
        self.imageLayout = QFormLayout()
        self.resultLayout = QFormLayout()

        # data layout
        self.dataLayoutGroupBox = QGroupBox("Input Data")
        self.dataLayout.addRow(self.cbDataType)
        self.dataLayout.addRow(self.inputFolderHbox)
        self.dataLayout.addRow(self.txtInputFolder)
        self.dataLayout.addRow(self.lblOutputFolder)
        self.dataLayout.addRow(self.txtOutputFolder)
        self.dataLayout.addRow(self.listinputDataList)
        self.dataLayout.addRow(self.saveResultLayout)
        self.dataLayout.addRow(self.processButtonsLayout)
        self.dataLayout.addRow(self.clusteringLayout)
        self.dataLayout.addRow(self.growingAnalyseLayout)
        self.dataLayoutGroupBox.setLayout(self.dataLayout)

        #  imageLayout
        self.imageLayoutGroupBox = QGroupBox("Output-Images")
        # self.leftMiddleLayout.addRow(self.methodSelection)
        self.scrollImages = QScrollArea()
        self.widget = QWidget()
        self.vbox = QVBoxLayout()
        self.widget.setLayout(self.vbox)
        self.scrollImages.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollImages.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollImages.setWidgetResizable(True)
        self.scrollImages.setWidget(self.widget)
        self.imageLayout.addRow(self.scrollImages)
        self.imageLayoutGroupBox.setLayout(self.imageLayout)
        self.imageLayoutGroupBox.setFixedSize(450,640)


        # resultLayout
        self.resultLayoutGroupBox = QGroupBox("Output-text")
        # self.rightMiddleLayout.addRow(self.outputImage)
        self.scrollTableResult = QScrollArea()
        self.widgetTable = QWidget()
        self.hbox = QHBoxLayout()
        self.widgetTable.setLayout(self.hbox)
        self.scrollTableResult.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollTableResult.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollTableResult.setWidgetResizable(True)
        self.scrollTableResult.setWidget(self.widgetTable)
        self.resultLayout.addRow(self.scrollTableResult)
        self.resultLayoutGroupBox.setLayout(self.resultLayout)
        self.resultLayoutGroupBox.setFixedSize(600,640)


        # tab1 main layout
        self.mainLayout.addWidget(self.dataLayoutGroupBox, 25)
        self.mainLayout.addWidget(self.imageLayoutGroupBox, 25)
        self.mainLayout.addWidget(self.resultLayoutGroupBox, 25)
        self.tab1.setLayout(self.mainLayout)

    def saveCloud(self,cloud, filename, outputFolder,part_count):
        # if cloud==None:
        #     return
        if type(cloud) == np.ndarray:
            c_temp=pcl.PointCloud_PointXYZI()
            c_temp.from_array(cloud)
            cloud=c_temp
        # outputFolder = outputFolder + "/ply/"
        outputFolder = outputFolder + "/ply/" + filename[0:-4]

        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        # pcl.save(cloud,outputFolder+".ply")
        pcl.save(cloud,outputFolder+"/"+str(part_count)+".ply")

    def saveClouds(self,clouds,filename,outputFolder):
        outputFolder=outputFolder+"/ply/"+filename[0:-4]
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        for part_count,cloud in clouds.items():
        # for part_count,cloud in enumerate(clouds):
            if type(cloud) == np.ndarray:
                c_temp = pcl.PointCloud_PointXYZI()
                c_temp.from_array(cloud)
                cloud = c_temp
            pcl.save(cloud,outputFolder+"/"+str(part_count)+".ply")

    def readInputFileName(self):
        folderName=self.txtInputFolder.text()
        fileName=[]

        data_directory = os.listdir(folderName)
        for file in data_directory:
            # if ".jpg" in file:
            #     fileName.append(file)
            fileName.append(file)
        return fileName

    def ColoringClouds(self,clouds):
        color_cloud=[]
        for i,cloud in clouds.items():
            if type(cloud)!=np.ndarray:
                cloud=cloud.to_array()
            cloud[:,3]=(20*(i+1))%255
            color_cloud.extend(cloud)
        cloud_tem = pcl.PointCloud_PointXYZI()
        if len(color_cloud)==0:
            return  None
        cloud_tem.from_array(np.asarray(color_cloud))
        return cloud_tem

    def ColoringClouds2(self,clouds):
        color_cloud=[]
        for i,cloud in clouds.items():
            if type(cloud)!=np.ndarray:
                cloud=cloud.to_array()

            xm= np.min(cloud[:, 0])
            cloud[:,3]=(100*(i+1))%500
            color_cloud.extend(cloud)
        cloud_tem = pcl.PointCloud_PointXYZI()
        if len(color_cloud)==0:
            return  None
        cloud_tem.from_array(np.asarray(color_cloud))
        return cloud_tem

    def startTestFunction(self):
        # self.segmentSaveSoilPlant()
        # self.segmentLeafUncentered()
        # self.startPlantDetectionSoil()
        # self.colorizeClusterData()
        # self.clusterFromSeedPoint()
        # self.segmentPlantsAcc2DetectedBox()
        # self.KmeansCls()
        # self.createbox()
        # self.MathematicalSeg()
        self.findDiffBetweenImages()

    #clustering

    def findDiffBetweenImages(self):
        imgFile1="C:/Users/serkan/Desktop/sonuclar/ground_truth"
        imgFile2="C:/Users/serkan/Desktop/sonuclar/Gaussian"
        # Gaussian Kmeans Hierarchial_avr Hierarchial_ward
        file1 = open(imgFile2+"/diff.txt", "w")
        data_directory1 = os.listdir(imgFile1)
        diff={}
        for f1 in data_directory1:
            if not f1.__contains__(".jpg"):
                continue
            img1 = cv2.imread(imgFile1+"/"+f1)
            img2 = cv2.imread(imgFile2+"/"+f1)

            color_th=10
            im_thresh =  np.zeros([432,768])
            total_color_count=0
            diff_count=0
            same_count=0
            for r,row in enumerate(img1):
                for c,pimg1 in enumerate(row):
                    pimg2=img2[r,c,:]
                    if (255-pimg1[0])>color_th  or (255-pimg1[1])>color_th or(255-pimg1[2])>color_th:
                        total_color_count+=1
                        if abs(np.int(pimg1[0])-np.int(pimg2[0]))<color_th and abs(np.int(pimg1[1])-np.int(pimg2[1]))<color_th and abs(np.int(pimg1[2])-np.int(pimg2[2]))<color_th:
                            im_thresh[r,c]=255
                            same_count+=1
                        else:
                            diff_count+=1
                            im_thresh[r, c]=125
            plt.imsave(imgFile2+"/"+f1[:-4]+'_th.jpg', np.array(im_thresh).reshape(432, 768), cmap=cm.gray)
            file1.write(str(same_count) + "\n")
            # file1.write(f1[:-4]+" "+str(same_count) + "\n")

        file1.close()

    def colorizeClusterData(self):
        clsType = self.cbClusterType.currentText()
        inputFolder = self.txtInputFolder.text()
        outputFolder = self.txtOutputFolder.text()
        saveImage = self.cbSaveResultImages.isChecked()
        ply2JpgConverter = Ply2JpgConverter()

        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().deleteLater()

        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        colored_imgs = {}
        clusters = {}
        for k, i in enumerate(self.inputFileNames):
            cloud = pcl.load_XYZI(inputFolder + "/" + i)
            clusters[k]=cloud
        clusters=self.changeClusterNumberAcc2Center(clusters)
        c_cloud = self.ColoringClouds2(clusters)
        # self.saveCloud(c_cloud, i, outputFolder + '/' + clsType , int(i[0:-4]))
        colored_imgs[int(i[0:-4])] = ply2JpgConverter.ConvertPly2Jpg(c_cloud, self.path, i, int(i[0:-4]))

        if saveImage:
            f = outputFolder + "/" + clsType
            if not os.path.exists(f):
                os.makedirs(f)
        for count, img in colored_imgs.items():
            im = Image.fromarray(img)
            if saveImage:
                im.save(outputFolder + '/' + clsType + "/" + str(count) + '.jpg')
            labelstr = QLabel(str(count))
            self.vbox.addWidget(labelstr)

            label = QLabel()
            qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap(qimage)
            pixmap = pixmap.scaled(384, 216, Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
            self.vbox.addWidget(label)

    def changeClusterNumberAcc2Center(self,clusters):
        means={}
        for i,cloud in clusters.items():
            if type(cloud)!=np.ndarray:
                cloud=cloud.to_array()
            xm= np.min(cloud[:, 0])
            ym = np.min(cloud[:, 1])
            xme= np.max(cloud[:, 0])
            yme = np.max(cloud[:, 1])
            means[i]=(xme+yme)

        for t in range(means.__len__()):
            for j in range(means.__len__()-1):
                if means[j]>means[j+1]:
                    cloud_temp=clusters[j]
                    clusters[j]=clusters[j+1]
                    clusters[j+1]=cloud_temp
                    xm_t= means[j]
                    means[j]=means[j+1]
                    means[j+1]=xm_t
        return clusters

    def KmeansCls(self):
        inputFolder = self.txtInputFolder.text()
        outputFolder = self.txtOutputFolder.text()
        saveImage = self.cbSaveResultImages.isChecked()

        ply2JpgConverter = Ply2JpgConverter()
        kmeansCluster=Clustering()

        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().deleteLater()

        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        colored_imgs = {}
        for fileCount, i in enumerate(self.inputFileNames):
            cloud = pcl.load_XYZI(inputFolder + "/" + i)
            fileName_txt = str(i[0:-4]) + ".txt"
            text_file = open(inputFolder + "/" + fileName_txt, "r")
            coordinates = text_file.readlines()
            if len(coordinates) == 0:
                continue
            coordinates = ply2JpgConverter.ConvertCoordinates(cloud, coordinates)
            centers = ply2JpgConverter.ConvertCoordinates2Centers(coordinates)
            clusters = kmeansCluster.KmeansCluster(cloud,len(coordinates.items()))
            # clusters = kmeansCluster.ClusterData(cloud,len(coordinates.items()),centers)
            c_cloud = self.ColoringClouds(clusters)
            self.saveCloud(c_cloud, i, outputFolder, int(i[0:-4]))
            colored_imgs[fileCount]= ply2JpgConverter.ConvertPly2Jpg(c_cloud, self.path, i, fileCount)

        if saveImage:
            f = outputFolder + "/" + i[0:-4]
            if not os.path.exists(f):
                os.makedirs(f)
        for count, img in colored_imgs.items():
            im = Image.fromarray(img)
            if saveImage:
                im.save(outputFolder + '/' + i[0:-4] + "/" + str(count) + '.jpg')
            labelstr = QLabel(str(i))
            self.vbox.addWidget(labelstr)

            label = QLabel()
            qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap(qimage)
            pixmap = pixmap.scaled(384, 216, Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
            self.vbox.addWidget(label)

    def MathematicalSeg(self):
        inputFolder = self.txtInputFolder.text()
        outputFolder = self.txtOutputFolder.text()+"/MathSegmentedData"
        saveImage = self.cbSaveResultImages.isChecked()
        ply2JpgConverter=Ply2JpgConverter()
        mathematicalSeg=MathematicalSoilSeg()
        statisticalFilter=StatisticalOutlierRemoval()
        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().deleteLater()
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        for fileCount, i in enumerate(self.inputFileNames):
            cleaned_imgs = {}
            cleaned_points= self.SegmentData(inputFolder + "/" + i, i,"leaf")
            for part_count, crop_points in cleaned_points.items():
                    cloud_bottom_part = ply2JpgConverter.NormalizeCloud(crop_points)
                    cloud_bottom_part = mathematicalSeg.SegmentSoil(cloud_bottom_part,-15,80,30,220)
                    cloud_bottom_part=statisticalFilter.CleanData(cloud_bottom_part,3)
                    cleaned_points[part_count]=cloud_bottom_part
                    cloud_bottom_part = ply2JpgConverter.EqualizeHistogram(cloud_bottom_part)
                    self.saveCloud(cloud_bottom_part, i, outputFolder, part_count)
                    cleaned_imgs[part_count]=ply2JpgConverter.ConvertPly2Jpg(cloud_bottom_part, self.path,i,part_count)
            if saveImage:
                f=outputFolder+"/jpg/"+i[0:-4]
                if not os.path.exists(f):
                    os.makedirs(f)
            for count, img in cleaned_imgs.items():
                im = Image.fromarray(img)
                if saveImage:
                    im.save(f + "/" + str(count) + '.jpg')
                labelstr = QLabel(str(i))
                self.vbox.addWidget(labelstr)

                label = QLabel()
                qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap(qimage)
                pixmap = pixmap.scaled(384, 216, Qt.KeepAspectRatio)
                label.setPixmap(pixmap)
                self.vbox.addWidget(label)

    def MathematicalSegAndCls(self):
        inputFolder = self.txtInputFolder.text()
        outputFolder = self.txtOutputFolder.text()
        saveImage = self.cbSaveResultImages.isChecked()
        ply2JpgConverter=Ply2JpgConverter()
        mathematicalSeg=MathematicalSoilSeg()
        statisticalFilter=StatisticalOutlierRemoval()
        clustering=Clustering()
        clsType = self.cbClusterType.currentText()
        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().deleteLater()
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        for fileCount, i in enumerate(self.inputFileNames):
            cleaned_imgs = {}
            cleaned_points= self.SegmentData(inputFolder + "/" + i, i,"leaf")
            for part_count, crop_points in cleaned_points.items():
                    cloud_bottom_part = ply2JpgConverter.NormalizeCloud(crop_points)
                    cloud_bottom_part = mathematicalSeg.SegmentSoil(cloud_bottom_part,-10,20,40,220)
                    cloud_bottom_part=statisticalFilter.CleanData(cloud_bottom_part,3)
                    cleaned_points[part_count]=cloud_bottom_part
                    cloud_bottom_part = ply2JpgConverter.EqualizeHistogram(cloud_bottom_part)
                    # self.saveCloud(cloud_bottom_part, i, outputFolder, part_count)
                    cleaned_imgs[part_count]=ply2JpgConverter.ConvertPly2Jpg(cloud_bottom_part, self.path,i,part_count)

            fileName_txt = str(i[0:-4]) + ".txt"
            text_file = open(inputFolder + "/" + fileName_txt, "r")
            plant_counts = text_file.readlines()
            colored_clouds={}
            colored_imgs={}
            for k,cloud in cleaned_points.items():
                if clsType == "Kmeans":
                    clusters = clustering.KmeansCluster(cloud,int(plant_counts[k]))
                elif clsType == "Spectral":
                    clusters = clustering.SpectralCluster(cloud, int(plant_counts[k]))
                elif clsType == "Hierarchial":
                    clusters = clustering.HierarchialCluster(cloud,int(plant_counts[k]))
                elif clsType == "Gaussian":
                    clusters = clustering.GaussianCluster(cloud, int(plant_counts[k]))
                elif clsType == "Optics":
                    clusters = clustering.OpticsCluster(cloud,int(plant_counts[k]))
                c_cloud= self.ColoringClouds(clusters)
                # colored_clouds[k] = c_cloud
                colored_imgs[k] = ply2JpgConverter.ConvertPly2Jpg(c_cloud, self.path, i, k)

            if saveImage:
                f=outputFolder+"/"+i[0:-4]
                if not os.path.exists(f):
                    os.makedirs(f)
            for count, img in cleaned_imgs.items():
                im = Image.fromarray(img)
                if saveImage:
                    im.save(outputFolder + '/' + i[0:-4] + "/" + str(count) + '.jpg')
                labelstr = QLabel(str(i))
                self.vbox.addWidget(labelstr)

                label = QLabel()
                qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap(qimage)
                pixmap = pixmap.scaled(384, 216, Qt.KeepAspectRatio)
                label.setPixmap(pixmap)
                self.vbox.addWidget(label)

                c_img= colored_imgs[count]
                label = QLabel()
                qimage = QImage(c_img, c_img.shape[1], c_img.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap(qimage)
                pixmap = pixmap.scaled(384, 216, Qt.KeepAspectRatio)
                label.setPixmap(pixmap)
                self.vbox.addWidget(label)


    #test functions
    def FiltterTopNoisyDataByHeight(self,cloud,ref_height):
        ptfilter = cloud.make_passthrough_filter()
        ptfilter.set_filter_field_name("z")
        ptfilter.set_filter_limits(ref_height,1000)
        cloud = ptfilter.filter()
        return cloud

    def segmentLeafUncentered(self):

        inputFolder = self.txtInputFolder.text()
        outputFolder = self.txtOutputFolder.text()
        dataType = self.cbDataType.currentText()
        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().deleteLater()

        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        for fileCount, i in enumerate(self.inputFileNames):
            cleaned_points = self.SegmentData(inputFolder + "/" + i, i, "leaf")
            self.saveClouds(cleaned_points,i,outputFolder)
            for part_count, crop_points in cleaned_points.items():
                if dataType == "Mungbean":
                    # cloud_bottom_part = ply2JpgConverter.NormalizeCloud(crop_points)
                    # cloud_bottom_part = self.regionGrowSoilSegm.SegmentSoilLeaf(crop_points)

                    cloud_bottom_part = self.regionGrowSoilSegm.SegmentSoilLeaf(crop_points)
                    # cloud_bottom_part=self.combineSoilSeg.SegmentSoil(crop_points,15,20)
                    # cloud_bottom_part=self.regionGrowSoilSegm.SegmentBigClusters(cloud_bottom_part,20,1000)
                    self.saveCloud(cloud_bottom_part, i, outputFolder, part_count)
                    # cloud_bottom_part = ply2JpgConverter.EqualizeHistogram(cloud_bottom_part)
                    # cleaned_points[part_count] = cloud_bottom_part
                    # cleaned_points[part_count] = statisticalFilter.CleanData(cloud_bottom_part)
                    # cleaned_points[part_count]=ply2JpgConverter.NormalizeCloud(cleaned_points[part_count])

    def segmentSaveSoilPlant(self):
        inputFolder = self.txtInputFolder.text()
        outputFolder = self.txtOutputFolder.text()

        statisticalFilter=StatisticalOutlierRemoval()
        ply2JpgConverter = Ply2JpgConverter()

        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().deleteLater()

        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        for fileCount, i in enumerate(self.inputFileNames):
            cleaned_points = self.SegmentData(inputFolder + "/" + i, i, "plant")
            for part_count, crop_points in cleaned_points.items():
                # cloud_bottom_part = ply2JpgConverter.NormalizeCloud(crop_points)
                # cloud_bottom_part = self.regionGrowSoilSegm.SegmentSoil(crop_points, 8)
                # cloud_bottom_part = self.planeModelSoilSegm.SegmentSoil(cloud_bottom_part, 10)
                cloud_bottom_part=ply2JpgConverter.EqualizeHistogram(crop_points)
                cloud_bottom_part = self.combineSoilSeg.SegmentSoilLeaf(cloud_bottom_part,10,5)
                # cloud_bottom_part = self.combineSoilSeg.SegmentSoilLeaf(cloud_bottom_part,8,6)

                cloud_bottom_part = statisticalFilter.CleanData(cloud_bottom_part, 1)
                cleaned_points[part_count] = cloud_bottom_part
                self.saveCloud(cloud_bottom_part, i, outputFolder, part_count)
                # cloud_bottom_part = ply2JpgConverter.EqualizeHistogram(cloud_bottom_part)
                # cleaned_points[part_count] = cloud_bottom_part

    def clusterFromSeedPoint(self):
        inputFolder=self.txtInputFolder.text()
        outputFolder=self.txtOutputFolder.text()
        fileName_ply=self.inputFileNames[0]
        fileName_txt=fileName_ply[0:-4]+".txt"
        ply2JpgConverter=Ply2JpgConverter()
        cleaned_imgs = {}

        text_file = open(inputFolder + "/" + fileName_txt, "r")
        coordinates = text_file.readlines()
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        cloud= pcl.load_XYZI(inputFolder + "/" + fileName_ply)
        # cloud=ply2JpgConverter.NormalizeCloud(cloud)
        # self.saveCloud(cloud, fileName_ply, outputFolder, 0)
        cloud=ply2JpgConverter.EqualizeHistogram(cloud)
        # c_len=len(cloud)
        # for i in range(cluster_count):
        # index = random.randint(0, c_len)
        coordinates=ply2JpgConverter.ConvertCoordinates(cloud,coordinates)
        cluster = self.regionGrowSoilSegm.SegmentAcc2Seed(cloud,0,coordinates,15)
        cleaned_imgs[0] = ply2JpgConverter.ConvertPly2Jpg(cloud, self.path, 0, 0)
        cleaned_imgs[1] = ply2JpgConverter.ConvertPly2JpgWithBox(cloud,coordinates)

        for i,cls_index in cluster.items():
            cloud_cls=cloud[cls_index,:]
            # cloud_cls=cloud.extract(cls_index,False)
            # self.saveCloud(cloud_cls, fileName_ply, outputFolder, i)
            cleaned_imgs[i+2] = ply2JpgConverter.ConvertPly2Jpg(cloud_cls, self.path,i+1,i+1)
            # print("index:{}, x:{}, y:{}, z:{} count:{}".format(0,cluster[0][0],cluster[0][1],cluster[0][2],cluster.size))
        for count, img in cleaned_imgs.items():
            # im = Image.fromarray(img)
            # im.save(outputFolder + '/' + fileName_ply[0:-4] + "_" + str(count) + '.jpg')

            label = QLabel()
            qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap(qimage)
            pixmap = pixmap.scaled(384, 216, Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
            self.vbox.addWidget(label)

    def segmentPlantsAcc2DetectedBox(self):
        # tek bir dosya içerisinde bulunan 0 dan n e kadar olan cloud ve onlara ait text dosyaları olacak
        inputFolder = self.txtInputFolder.text()
        outputFolder = self.txtOutputFolder.text()

        ply2JpgConverter = Ply2JpgConverter()

        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().deleteLater()

        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        for fileCount, i in enumerate(self.inputFileNames):
            cloud = pcl.load_XYZI(inputFolder + "/" + i)
            fileName_txt = str(i[0:-4]) + ".txt"
            text_file = open(inputFolder + "/" + fileName_txt, "r")
            coordinates = text_file.readlines()
            if len(coordinates)==0:
                continue
            coordinates=ply2JpgConverter.ConvertCoordinates(cloud,coordinates)
            clusters= self.regionGrowSoilSegm.SegmentAcc2Seed(cloud, coordinates,15)
            clouds=self.ColoringClouds(clusters)
            self.saveCloud(clouds,i,outputFolder,int(i[0:-4]))

    def createbox(self):
        inputFolder = self.txtInputFolder.text()
        outputFolder = self.txtOutputFolder.text()
        ply2JpgConverter = Ply2JpgConverter()
        cleaned_imgs = {}

        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().deleteLater()

        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        for fileCount, i in enumerate(self.inputFileNames):
            cloud = pcl.load_XYZI(inputFolder + "/" + i)
            fileName_txt = str(i[0:-4]) + ".txt"
            text_file = open(inputFolder + "/" + fileName_txt, "r")
            coordinates = text_file.readlines()

            cloud = ply2JpgConverter.EqualizeHistogram(cloud)
            coordinates = ply2JpgConverter.ConvertCoordinates(cloud, coordinates)
            img = ply2JpgConverter.ConvertPly2JpgWithBox(cloud,coordinates)
            cleaned_imgs[fileCount]=img

        for img in cleaned_imgs.values():
            label = QLabel()
            qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap(qimage)
            pixmap = pixmap.scaled(384, 216, Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
            self.vbox.addWidget(label)

    # thinninh öncesi kücük fasulye icin kullanılan
    def startPlantDetectionEskiKucuk(self):
        inputFolder = self.txtInputFolder.text()
        outputFolder = self.txtOutputFolder.text()
        saveImage = self.cbSaveResultImages.isChecked()
        dataType = self.cbDataType.currentText()

        objectDetection = ObjectDetectionFolder("multiple_plant")
        # objectDetection=ObjectDetectionFolder(dataType+"_plant")
        ply2JpgConverter = Ply2JpgConverter()

        statisticalFilter = StatisticalOutlierRemoval()
        excelt_data = {}

        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().deleteLater()

        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        for fileCount, i in enumerate(self.inputFileNames):
            cleaned_points = self.SegmentData(inputFolder + "/" + i, i, "plant")
            cleaned_imgs = {}
            for part_count, crop_points in cleaned_points.items():
                cloud_bottom_part = ply2JpgConverter.NormalizeCloud(crop_points)
                if dataType == "Mungbean":
                    # cloud_bottom_part=self.regionGrowSoilSegm.SegmentSoil(cloud_bottom_part,6)
                    cloud_bottom_part = ply2JpgConverter.EqualizeHistogram(cloud_bottom_part)  # alttaki ile beraber
                    cloud_bottom_part = self.combineSoilSeg.SegmentSoil(cloud_bottom_part, 10, 5)
                    cloud_bottom_part = ply2JpgConverter.EqualizeHistogram(cloud_bottom_part)
                    cleaned_points[part_count] = cloud_bottom_part
                elif dataType == "FingerMillet":
                    cloud_bottom_part = self.planeModelSoilSegm.SegmentSoil(cloud_bottom_part, 15)
                    cloud_bottom_part = self.regionGrowSoilSegm.SegmentBigClusters(cloud_bottom_part, 20)
                    cloud_bottom_part = ply2JpgConverter.EqualizeHistogram(cloud_bottom_part)
                    cleaned_points[part_count] = cloud_bottom_part
                    self.saveCloud(cloud_bottom_part, i, outputFolder, part_count)
            for part_count, crop_points in cleaned_points.items():
                if dataType == "Mungbean":
                    cleaned_points = statisticalFilter.CleanData(crop_points, 1)
                elif dataType == "FingerMillet":
                    cleaned_points = statisticalFilter.CleanData(crop_points, 2)
                # self.saveCloud(cleaned_points, i, outputFolder,part_count)
                cleaned_imgs[part_count] = ply2JpgConverter.ConvertPly2Jpg(cleaned_points, self.path, i, part_count)
            imgs, counts = objectDetection.Detect(cleaned_imgs, outputFolder + "/box_coor/" + i[0:-4])
            plant_counts = {}
            if fileCount == 0:
                label = QLabel("File: " + i[0:-4])
                self.vbox.addWidget(label)
                self.trayCount = len(cleaned_imgs)
                self.fileCount = len(self.inputFileNames)
            if saveImage:
                f = outputFolder + "/" + i[0:-4]
                if not os.path.exists(f):
                    os.makedirs(f)
            if len(cleaned_imgs) == 0:
                return
            for count, img in imgs.items():
                im = Image.fromarray(img)
                if saveImage:
                    im.save(outputFolder + '/' + i[0:-4] + "/" + str(count) + '.jpg')
                if fileCount == 0:
                    labelstr = QLabel(str(count))
                    label = QLabel()
                    qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
                    pixmap = QPixmap(qimage)
                    pixmap = pixmap.scaled(384, 216, Qt.KeepAspectRatio)
                    label.setPixmap(pixmap)
                    self.vbox.addWidget(labelstr)
                    self.vbox.addWidget(label)
                plant_counts[count] = len(counts[count])

            excelt_data[i[0:-4]] = plant_counts
        self.writeData2Excel(excelt_data)
        self.showDataOnTable(excelt_data)

    # thinning sonrası plant detection için kullanılan fonksiyon startPlantDetection içerisinde taşındıs
    def startPlantDetectionSoil(self):
        inputFolder=self.txtInputFolder.text()
        outputFolder=self.txtOutputFolder.text()
        saveImage=self.cbSaveResultImages.isChecked()
        dataType=self.cbDataType.currentText()
        # mathematicalSeg=MathematicalSoilSeg()

        objectDetection=ObjectDetectionFolder("multiple_plant")
        objectDetection=ObjectDetectionFolder(dataType+"_plant")
        ply2JpgConverter=Ply2JpgConverter()
        statisticalFilter=StatisticalOutlierRemoval()
        excelt_data={}

        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().deleteLater()

        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        for fileCount,i in enumerate(self.inputFileNames):
            cleaned_points= self.SegmentData(inputFolder + "/" + i, i,"leaf")
            cleaned_imgs={}
            for part_count,crop_points in cleaned_points.items():
                # self.saveCloud(crop_points, i, outputFolder, part_count)
                cloud_bottom_part=ply2JpgConverter.NormalizeCloud(crop_points)
                cloud_bottom_part=ply2JpgConverter.EqualizeHistogram(cloud_bottom_part) #normalde bunun olması lazım mathematicalde kaldırıldı bir alttaki fonksiyonla beraber
                cloud_bottom_part = self.combineSoilSeg.SegmentSoilLeaf(cloud_bottom_part,-1,12)
                cloud_bottom_part = self.regionGrowSoilSegm.SegmentBigClusters(cloud_bottom_part, 6,2000)
                # cloud_bottom_part = self.planeModelSoilSegm.SegmentSoil(cloud_bottom_part, 8)
                # cloud_bottom_part = self.regionGrowSoilSegm.SegmentBottomClusters(cloud_bottom_part)
                # cloud_bottom_part = self.regionGrowSoilSegm.SegmentSoilLeaf(cloud_bottom_part)

                # cloud_bottom_part = mathematicalSeg.SegmentSoil(cloud_bottom_part,-10,20,40,220)
                cloud_bottom_part=ply2JpgConverter.EqualizeHistogram(cloud_bottom_part)
                cleaned_points[part_count] =cloud_bottom_part
            for part_count,crop_points in cleaned_points.items():
                if dataType == "Mungbean":
                    cleaned_points=statisticalFilter.CleanData(crop_points,1)
                elif dataType == "FingerMillet":
                    cleaned_points=statisticalFilter.CleanData(crop_points,2)
                elif dataType == "ChickpeaExp33":
                    cleaned_points=statisticalFilter.CleanData(crop_points,1)
                # segment point less then z -150 to get better result for heigt coloring
                cleaned_points=self.FiltterTopNoisyDataByHeight(cleaned_points,-150)
                cleaned_points = ply2JpgConverter.NormalizeCloud(cleaned_points)
                cleaned_imgs[part_count]=ply2JpgConverter.ConvertPly2Jpg(cleaned_points, self.path,i,part_count)

            imgs,counts=objectDetection.Detect(cleaned_imgs)
            plant_counts={}
            if fileCount == 0:
                label = QLabel("File: "+i[0:-4])
                self.vbox.addWidget(label)
                self.trayCount=len(cleaned_imgs)
                self.fileCount=len(self.inputFileNames)
            if saveImage:
                f=outputFolder+"/"+i[0:-4]
                if not os.path.exists(f):
                    os.makedirs(f)
            if len(cleaned_imgs)==0:
                return
            for count,img in cleaned_imgs.items():
                im= Image.fromarray(img)
                if saveImage:
                   # im.save(outputFolder+'/'+i[0:-4]+"_"+str(count)+'.jpg')
                   im.save(outputFolder + '/' + i[0:-4] + "/" + str(count) + '.jpg')
                if fileCount==0:
                    label=QLabel()
                    qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
                    pixmap = QPixmap(qimage)
                    pixmap = pixmap.scaled(384, 216, Qt.KeepAspectRatio)
                    label.setPixmap(pixmap)
                    self.vbox.addWidget(label)
                plant_counts[count]=len(counts[count])

            excelt_data[i[0:-4]]=plant_counts
        self.writeData2Excel(excelt_data)
        self.showDataOnTable(excelt_data)


app = QApplication(sys.argv)
m = Window()
app.exec_()
