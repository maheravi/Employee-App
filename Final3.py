import sys
import keyboard  # using module keyboard
import numpy as np
from PySide6.QtWidgets import *
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QImage, QIcon, QPixmap, QImage
from PySide6.QtCore import *
from sqlite3 import connect
import cv2
from functools import partial
import numpy as np

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

from database import Database
from filter import Filter

# Convert an opencv image to QPixmap
def convertCvImage2QtImage(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    height, width, channel = cv_img.shape
    bytesPerLine = 3 * width
    qImg = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888)

    return QPixmap.fromImage(qImg)


class FirstForm(QMainWindow):

    def __init__(self):
        super(FirstForm, self).__init__()

        loader = QUiLoader()
        self.ui = loader.load("form.ui")
        self.ui.show()
        self.readList()

        self.ui.new_employee.clicked.connect(self.openOtherForm)
        self.ui.del_item.clicked.connect(self.delSelect)
        self.ui.edit_item.clicked.connect(self.editList)

        # self.window = SecondForm()
        # main.my_signal.connect(self.readList())

    def delSelect(self):

        indices = self.ui.tableWidget.selectionModel().selectedRows()

        for index in sorted(indices):
            Database.delete(self.ui.tableWidget.item(index.row(), 2).text())
            self.ui.tableWidget.removeRow(index.row())

    def readList(self):

        result = Database.select()

        for row, data in enumerate(result):
            self.ui.tableWidget.insertRow(row)
            for col, col_data in enumerate(data):
                if col != 0:
                    if col == 5:

                        # method I
                        pic = QPixmap(f"{col_data}")
                        pic_scaled = pic.scaled(100, 100)
                        self.label = QLabel()
                        self.label.setPixmap(pic_scaled)
                        self.ui.tableWidget.setCellWidget(row, col - 1, self.label)

                        # method II

                        # icon = QIcon(QPixmap(f"{col_data}"))
                        # item = QTableWidgetItem(icon, "")
                        # self.ui.tableWidget.setItem(row, col - 1, item)
                    else:
                        item = str(col_data)
                        self.ui.tableWidget.setItem(row, col-1, QTableWidgetItem(item))
                    self.ui.tableWidget.verticalHeader().setDefaultSectionSize(100)

    def openOtherForm(self):

        self.hide()
        self.w = SecondForm(self)
        # otherview.show()

    def editList(self):

        self.hide()
        self.w2 = ThirdForm(self)
        # otherview.show()


class SecondForm(QWidget, QThread):

    my_signal = Signal()

    def __init__(self, parent=None):
        super(SecondForm, self).__init__(parent)
        loader = QUiLoader()
        self.ui2 = loader.load("form4.ui")
        self.ui2.btn_add.clicked.connect(self.add)
        self.ui2.btn_camera.clicked.connect(self.webcam)
        self.frame_face = None
        self.ui2.show()

    def add(self):
        # self.parent().show()

        name = self.ui2.tb_name.text()
        lastname = self.ui2.tb_lastname.text()
        nationalcode = self.ui2.tb_nationalcode.text()
        dateofbirth = self.ui2.tb_dateofbirth.text()
        cv2.imwrite(f"{nationalcode}.png", self.frame_face)
        Database.insert(name, lastname, nationalcode, dateofbirth)
        mytuple = (name, lastname, nationalcode, dateofbirth)
        row = window.ui.tableWidget.rowCount()
        window.ui.tableWidget.insertRow(row)
        print(row)
        for col in range(0, 4):
            window.ui.tableWidget.setItem(row, col, QTableWidgetItem(str(mytuple[col])))
        pic = QPixmap(f"{nationalcode}"+'.png')
        pic_scaled = pic.scaled(100, 100)
        self.label = QLabel()
        self.label.setPixmap(pic_scaled)
        window.ui.tableWidget.setCellWidget(row, 4, self.label)
        self.close()

    def webcam(self):

        my_video = cv2.VideoCapture(0)
        while True:

            validation, frame = my_video.read()
            width, height, channel = frame.shape
            image = np.zeros((width, height, channel), dtype='uint8')

            if validation is not True:
                break

            face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(frame_gray, 1.3)

            for i, face in enumerate(faces):
                x, y, w, h = face

                self.frame_face = frame[y:y + h, x:x + w]

                # if self.ui2.takepicture.clicked is True:
                #     self.ui2.takepicture.clicked.connect(self.takepicture)
                #     break

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 8)

            image_frame = cv2.resize(frame, (height // 3, width // 3))
            self.ui2.lbl1.setIconSize(QSize(100, 100))
            my_pixmap = convertCvImage2QtImage(image_frame)
            self.ui2.lbl1.setIcon(QIcon(my_pixmap))
            self.ui2.lbl1.clicked.connect(self.add)
            # if self.ui2.lbl1.pressed():
            #     print("yes")
            #     self.ui2.lbl1.clicked.connect(self.add)
            #     # break

            invert = Filter.apply_invert(image_frame)
            self.ui2.lbl2.setIconSize(QSize(100, 100))
            my_pixmap = convertCvImage2QtImage(invert)
            self.ui2.lbl2.setIcon(QIcon(my_pixmap))
            self.ui2.lbl2.clicked.connect(self.add)

            portrait = Filter.apply_portrait_mode(image_frame)
            self.ui2.lbl3.setIconSize(QSize(100, 100))
            my_pixmap = convertCvImage2QtImage(portrait)
            self.ui2.lbl3.setIcon(QIcon(my_pixmap))
            self.ui2.lbl3.clicked.connect(self.add)

            pencil = Filter.pencil_sketch(frame)
            self.ui2.lbl4.setIconSize(QSize(100, 100))
            my_pixmap = convertCvImage2QtImage(pencil)
            self.ui2.lbl4.setIcon(QIcon(my_pixmap))
            self.ui2.lbl4.clicked.connect(self.add)

            stylize = Filter.apply_stylize(image_frame)
            self.ui2.lbl5.setIconSize(QSize(100, 100))
            my_pixmap = convertCvImage2QtImage(stylize)
            self.ui2.lbl5.setIcon(QIcon(my_pixmap))
            self.ui2.lbl5.clicked.connect(self.add)

            gray = Filter.gray_scale(image_frame)
            self.ui2.lbl6.setIconSize(QSize(100, 100))
            my_pixmap = convertCvImage2QtImage(gray)
            self.ui2.lbl6.setIcon(QIcon(my_pixmap))
            self.ui2.lbl6.clicked.connect(self.add)


            stylize = Filter.apply_stylize(image_frame)
            self.ui2.lbl7.setIconSize(QSize(100, 100))
            my_pixmap = convertCvImage2QtImage(stylize)
            self.ui2.lbl7.setIcon(QIcon(my_pixmap))
            self.ui2.lbl7.clicked.connect(self.add)


            stylize = Filter.apply_stylize(image_frame)
            self.ui2.lbl8.setIconSize(QSize(100, 100))
            my_pixmap = convertCvImage2QtImage(stylize)
            self.ui2.lbl8.setIcon(QIcon(my_pixmap))
            self.ui2.lbl8.clicked.connect(self.add)


            stylize = Filter.apply_stylize(image_frame)
            self.ui2.lbl9.setIconSize(QSize(100, 100))
            my_pixmap = convertCvImage2QtImage(stylize)
            self.ui2.lbl9.setIcon(QIcon(my_pixmap))
            self.ui2.lbl9.clicked.connect(self.add)


            cv2.waitKey(1)
            #self.showOutput(image)

    def takepicture(self):

        cv2.imwrite(f"{self.ui2.tb_nationalcode.text()}.png", self.frame_face)

    def showOutput(self, image):

        image = cv2.resize(image, (200, 150))
        my_pixmap = convertCvImage2QtImage(image)
        self.ui2.lbl1.setIconSize(QSize(100, 100))
        self.ui2.lbl1.setIcon(QIcon(my_pixmap))


class ThirdForm(QWidget, QThread):
    my_signal = Signal()

    def __init__(self, parent=None):
        super(ThirdForm, self).__init__(parent)
        loader = QUiLoader()
        self.ui3 = loader.load("form3.ui")
        self.ui3.btn_edit.clicked.connect(self.edit)
        self.ui3.show()
        indices = window.ui.tableWidget.selectionModel().selectedRows()
        for index in sorted(indices):
            self.row = index.row()
            name = window.ui.tableWidget.item(index.row(), 0).text()
            lastname = window.ui.tableWidget.item(index.row(), 1).text()
            nationalcode = window.ui.tableWidget.item(index.row(), 2).text()
            dateofbirth = window.ui.tableWidget.item(index.row(), 3).text()
            self.ui3.tb_name.setText(name)
            self.ui3.tb_lastname.setText(lastname)
            self.ui3.tb_nationalcode.setText(nationalcode)
            self.ui3.tb_dateofbirth.setText(dateofbirth)

    def edit(self):

        # self.parent().show()
        name = self.ui3.tb_name.text()
        lastname = self.ui3.tb_lastname.text()
        nationalcode = self.ui3.tb_nationalcode.text()
        dateofbirth = self.ui3.tb_dateofbirth.text()
        my_con = connect('ListofEmployee.db')
        my_cursor = my_con.cursor()
        my_cursor.execute(
            f"INSERT INTO List_of_Employee(Name, LastName, NationalCode, DateofBirth) VALUES('{name}', '{lastname}', '{nationalcode}', '{dateofbirth}')")
        my_con.commit()
        my_con.close()
        mytuple = (name, lastname, nationalcode, dateofbirth)
        for col in range(0, 4):
            window.ui.tableWidget.setItem(self.row, col, QTableWidgetItem(str(mytuple[col])))
        self.close()


app = QApplication(sys.argv)
window = FirstForm()
# main.show()
sys.exit(app.exec())
