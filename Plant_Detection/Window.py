from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from collections import deque
from PyQt5.QtGui import QIcon, QPixmap, QImage, QPainter, QPen # icon and load image
from PyQt5.QtCore import Qt, QPoint
from Segmentation.SegmentMungbean import *
import sys

class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        # main window
        self.width = 1080
        self.height = 640

        self.setWindowTitle("Digit Classification")
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
        self.cbDataType.addItems(["Mungbean","Sorghum2019","Sorghum2017"])

        self.inputFolderHbox=QHBoxLayout()
        self.lblInputFolder=QLabel("Enter Input Folder Path Below OR")
        self.btnInputFolder=QPushButton("Select From dialog Box")
        self.btnInputFolder.clicked.connect(self.selectInputFolder)
        self.inputFolderHbox.addWidget(self.lblInputFolder)
        self.inputFolderHbox.addWidget(self.btnInputFolder)
        self.txtInputFolder=QLineEdit()

        self.cbSaveResultImages=QCheckBox("Save Result Image",self)

        self.processButtonsLayout=QHBoxLayout()
        self.btnPlantDetection=QPushButton("Start Plant Detection")
        self.btnPlantDetection.clicked.connect(self.startPlantDetection)
        self.btnLeafDetection=QPushButton("Start Leaf Detection")
        self.btnLeafDetection.clicked.connect(self.startLeafDetection)
        self.processButtonsLayout.addWidget(self.btnPlantDetection)
        self.processButtonsLayout.addWidget(self.btnLeafDetection)

        self.listinputDataList = QListWidget()

        #output images widgets

        #output text widgets


        self.lblOutputFolder=QLabel("Enter Output Folder Path: ")
        self.txtOutputFolder=QLineEdit()

    def startLeafDetection(self):
        pass

    def startPlantDetection(self):
        pass

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
        self.dataLayout.addRow(self.cbSaveResultImages)
        self.dataLayout.addRow(self.processButtonsLayout)
        self.dataLayoutGroupBox.setLayout(self.dataLayout)

        #  imageLayout
        self.imageLayoutGroupBox = QGroupBox("Output-Images")
        # self.leftMiddleLayout.addRow(self.methodSelection)
        self.imageLayoutGroupBox.setLayout(self.imageLayout)

        # resultLayout
        self.resultLayoutGroupBox = QGroupBox("Output-ttext")
        # self.rightMiddleLayout.addRow(self.outputImage)
        self.resultLayoutGroupBox.setLayout(self.resultLayout)

        # tab1 main layout
        self.mainLayout.addWidget(self.dataLayoutGroupBox, 25)
        self.mainLayout.addWidget(self.imageLayoutGroupBox, 25)
        self.mainLayout.addWidget(self.resultLayoutGroupBox, 25)
        self.tab1.setLayout(self.mainLayout)

    def readInputFileName(self):
        folderName=self.txtInputFolder.text()
        fileName=[]

        data_directory = os.listdir(folderName)
        for file in data_directory:
            if ".ply" in file:
                fileName.append(file)
        return fileName
