import cv2,os,sys,PIL
import numpy as np
from threading import Thread

import qimage2ndarray as qimage2ndarray
from PIL.ImageQt import ImageQt
from keras.models import load_model
from PyQt5 import QtGui, uic, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QMutex, Qt

form_class = uic.loadUiType("videoShow.ui")[0]
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#머리통 학습값
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')#눈알 학습값.
catractModel = load_model('eye2.h5')#백내장 검사.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #ignore warnings

class camThread(QThread):

    changePixmap = pyqtSignal(QImage)
    send_instance_singal = pyqtSignal("PyQt_PyObject")
    eyeList=[]


    def __init__(self,parent):
        super().__init__(parent)
        self.parent = parent
        self.camera = cv2.VideoCapture(0)



    def run(self):

        run_video = True

        while run_video:
            #print("run_video cam on  (thread 1) " )

            ret, image = self.camera.read()

            height, width = image.shape[:2]  # 캠에서의 해상도 미사용.

            if ret == False:  # 카메라가 정상자동하지 않을시.
                run_video = False
            '''
            print("cam read  (thread 1) ")


            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cvtColor=convertColor  rgb type.
            
                        웹캠 정상작동여부 테스트
                        testImg = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                        testImg = Image.fromarray(testImg,'RGB')
                        testImg.show()
            
            qt_image1 = QtGui.QImage(color_swapped_image.data,
                                     width,
                                     height,
                                     color_swapped_image.strides[0],
                                     QtGui.QImage.Format_RGB888)

            videoWidth =self.parent.videoWidget.width()
            videoHeight = self.parent.videoWidget.height()

            if qt_image1.isNull():
                print("viewer Dropped (thread 1) ")
                run_video=False

            #resizeImage = qt_image1.scaled(videoWidth, videoHeight) 비활성화  ... 눈알 찍히는 이미지 출력예정.
            print("shoot cam!! (thread 1) ")
            #self.changePixmap.emit(resizeImage)#전체 얼굴 프린트.
            print("emit하면 다음으로 넘어가긴 하나? ")

            '''
            #얼굴과 눈알 계산하기.  detectMultiScale q붙은곳은 무조건 무언가를 감지하는 곳이다. 얼굴/눈
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#camera.read 의 image
            faces = detector.detectMultiScale(img, 1.3, 5)
            #print("face detect  (thread2)")
            for (x, y, w, h) in faces:  # 결과값 좌표. 좌상 xy 와 높이 너비 얼굴 이미지는 1개여서 루프는 1회만 돌아간다.
                # 생각해보니 rgb가 아닌 bgr순서였다. 이건 파란색 네모.
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)#얼굴에 파란네모 씌우기.
                roi_color = img[y:y + h, x:x + w]  # 풀컬러 얼굴 범위한정 이미지.
                eyes = eye_cascade.detectMultiScale(roi_color, minNeighbors=15)
                #print("eye detect  (thread2) ")
                for (ex, ey, ew, eh) in eyes:  # 눈알은 2개니 루프는 2회.
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    eyeImgReturn = roi_color[ey:ey + eh, ex:ex + ew, :]  # 배열 컷  가로세로가 상당히 헷갈린다 주의주의.
                    # eyeImgReturn = cv2.cvtColor(eyeImgReturn, cv2.COLOR_BGR2RGB)  불필요한 코드.
                    #print("eye image arr shape", str(np.shape(eyeImgReturn)))
                    image = PIL.Image.fromarray(eyeImgReturn, 'RGB')  # image.show()
                    #image.show()  # 눈깔따기 알고리즘 완성.
                    image = ImageQt(image).copy()
                    #print("eyeImage  (thread2) ", image)
                    # 이시점에서 image를 qImage로 바꾸어야한다.

                    #h, w, _ = eyeImgReturn.shape

                    #qimage = QImage(eyeImgReturn.data, w, h, 3 * w, QImage.Format_RGB888)
                    # 줄당 바이트 갯수는 어짜피 3원색 *가로줄 전체 일테니...
                    #print("qimage print ",qimage)
                    self.eyeList.append(image)
                    #print(self.eyeList)
            eyefaceImage = PIL.Image.fromarray(img,'RGB')
            eyefaceImage = ImageQt(eyefaceImage).copy()
            videoWidth = self.parent.videoWidget.width()
            videoHeight = self.parent.videoWidget.height()
            eyefaceImage = eyefaceImage.scaled(videoWidth, videoHeight)
            #전체 얼굴 해상도를 줄이는게 나중에있으니 눈알 해상도와는 관계 없음.
            self.changePixmap.emit(eyefaceImage)
            #print("눈알 인식된 이미지 갯수",len(self.eyeList))
            if(len(self.eyeList)<=1):
                #print("양쪽눈이 인식되지 않았습니다. ")
                continue
            #print("emit 준비.")
            self.send_instance_singal.emit(self.eyeList)



class WindowClass(QMainWindow, form_class):

    eyeImages=[1,3]
    camImage = QImage()
    mutex = QMutex()
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("타이틀")

        # videoWidget
        # eyeLeft
        # eyeRight
        #self.videoWidget.setStyleSheet("background-color:rgb(255, 0, 255);")

        self.show()

        x = camThread(self)
        x.changePixmap.connect(self.webCamOn)
        x.send_instance_singal.connect(self.dsplyEyes)
        x.start()

        self.calResultBtn.clicked.connect(self.predictionThread)


    @QtCore.pyqtSlot(QImage)
    def webCamOn(self,image):
        self.camImage = image
        self.videoWidget.setPixmap(QPixmap.fromImage(self.camImage))

    @pyqtSlot("PyQt_PyObject")
    def dsplyEyes(self,eyeList):

        #self.eyeImages = eyeList[:]
        #이미지 크기 위젯에 일치시키기.
        leftWidgetWidth=self.eyeLeftWidget.width()
        leftWidgetHeight=self.eyeLeftWidget.width()
        self.eyeImages[0] = eyeList[0]
        eyeList[0] = eyeList[0].scaled(leftWidgetWidth,leftWidgetHeight)


        rightWidgetWidth = self.eyeRightWidget.width()
        rightWidgetHeight = self.eyeRightWidget.width()
        self.eyeImages[1] = eyeList[1]
        eyeList[1] = eyeList[1].scaled(rightWidgetWidth, rightWidgetHeight)


        self.eyeLeftWidget.setPixmap(QPixmap.fromImage(eyeList[0]))
        self.eyeRightWidget.setPixmap(QPixmap.fromImage(eyeList[1]))
        eyeList.clear()  # 그다음 다시 새로운 얼굴.
        #확률 계산작업은 부하가 걸려서 사진이 못 올라올것을 의심하여, 별도의 스레드로 분리하게됨.

        if (len(eyeList) == 2):
            print("write 222")
            #thread = Thread(target=self.writePerdiction, args=(eyeList,))
            #self.writePerdiction(eyeList)
            #print("thread call")
            #thread.start()

    def predictionThread(self):
        print('fucntioin prediton thread1 ')
        try:
            thread = Thread(target=self.writePerdiction ,args=(self.eyeImages,))
        except Exception as e:
            print ("Thread call exception : ",e)
        print('fucntioin prediton thread3 ')
        thread.start()



    def writePerdiction(self,eyeImages):

        #eyeImages are numpyImage.
        if (len(eyeImages) != 2):
            print("눈알 두 쪽이 정상적으로 인식되지 않았음.")
            return

        print("write Perdicitoin 메소드 실행.")
        print("eye images 변수의 타입 " ,type(eyeImages[0]))


        print(np.array(eyeImages).shape)
        print("이미지 내부 상세정보",eyeImages[0])
        resultPrediction = np.ndarray(shape=(2, 128, 128, 3), dtype=np.float32)
        eyeList = eyeImages


        #Qimage(<class 'PyQt5.QtGui.QImage'>  로그찍었을때의 타입.) resize
        tempImg1 = eyeList[0].scaled(128, 128)
        tempImg2 = eyeList[1].scaled(128, 128)
        tempImg1 = tempImg1.QtGui.QImage.format(QImage.Format_RGBA8888)

        #QImage to numpy 수정중....


        #resultPrediction[0]=PIL.Image.fromarray(tempImg1,'RGB')
        #resultPrediction[1]=PIL.Image.fromarray(tempImg2,'RGB')

        resultPrediction[0] = qimage2ndarray.recarray_view(tempImg1) #사용불가
        resultPrediction[1]= qimage2ndarray.recarray_view(tempImg2)  # recarry_view 32bit rgb only

        prediction = catractModel.predict(resultPrediction)

        _, a = prediction[0]
        _, b = prediction[1]

        self.leftEyeWidgetResult.setText(a)
        self.rightEyeWidgetResult.setText(b)





        '''
        주석처진데는 무쓸모지만 필요할까봐서 남겨둠.
        width = incomingImage1.width()
        height = incomingImage1.height()

        ptr = incomingImage1.constBits()
        resultPrediction[0] = np.array(ptr).reshape(height, width, 3)

        width = incomingImage2.width()
        height = incomingImage2.height()

        ptr = incomingImage2.constBits()
        resultPrediction[1] = np.array(ptr).reshape(height, width, 3)
        


       

        '''











if __name__ == '__main__':

    app = QApplication(sys.argv)
    myWindow = WindowClass()
    print("helloworld")


    app.exec_()
