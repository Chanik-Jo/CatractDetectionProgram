import cv2,os,sys,time
from keras.models import load_model
from PyQt5 import QtGui, uic
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, qRgb
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot

form_class = uic.loadUiType("videoShow.ui")[0]
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#머리통 학습값
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')#눈알 학습값.
catractModel = load_model('eye2.h5')#백내장 검사.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #ignore warnings

class camThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self,parent):
        super().__init__(parent)
        self.parent = parent

    def run(self):
        camera = cv2.VideoCapture(0)  # 0번째 장치에서 비디오를 캡쳐합니다. 웹캠이 하나니 당연히 0번..
        ret, image = camera.read()  # ret: 다음 프레임이 있을경우 true/ image는 다음 이미지
        height, width = image.shape[:2]  # 가로세로 해상도.
        run_video = True

        while run_video:
            if ret == False:  # 카메라가 정상자동하지 않을시.
                run_video = False
            ret, image = camera.read()
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cvtColor=convertColor  rgb type.
            '''
                        웹캠 정상작동여부 테스트
                        testImg = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                        testImg = Image.fromarray(testImg,'RGB')
                        testImg.show()
            '''
            qt_image1 = QtGui.QImage(color_swapped_image.data,
                                     width,
                                     height,
                                     color_swapped_image.strides[0],
                                     QtGui.QImage.Format_RGB888)

            videoWidth =self.parent.videoWidget.width()
            videoHeight = self.parent.videoWidget.height()
            resizeImage = qt_image1.scaled(videoWidth,videoHeight, Qt.KeepAspectRatio)
            self.changePixmap.emit(resizeImage)



            if qt_image1.isNull():
                print("viewer Dropped")

            if qt_image1.size() != self.parent.videoWidget.size():
                self.parent.videoWidget.setPixmap(QPixmap.fromImage(qt_image1))
                # 'QWidget' object has no attribute 'setPixmap'





class WindowClass(QMainWindow, form_class):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("검사검사검사")
        # resultTxtWidget
        # videoWidget
        # eyeLeft
        # eyeRight
        self.videoWidget.setStyleSheet("background-color:rgb(255, 0, 255);")
        self.resultTxtWidget.setText("abcaBCCCC")
        x = camThread(self)
        x.changePixmap.connect(self.webCamOn)
        x.start()
        self.show()

    @pyqtSlot(QImage)
    def webCamOn(self,image):
        self.videoWidget.setPixmap(QPixmap.fromImage(image))



    def frameWrite(self):
        pass



if __name__ == '__main__':


    app = QApplication(sys.argv)
    myWindow = WindowClass()
    print("helloworld")
    myWindow.show()

    app.exec_()
