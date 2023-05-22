import os
import cv2 as cv
import numpy as np
from threading import Thread
from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QTableWidgetItem, QFileDialog


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('GUI_TA.ui', self)

        self.path = None
        self.path2 = None

        self.pathhasilKalib = "hasilKalib"
        self.pathhasilUndistort = "hasilUndistort"
        self.data = None
        self.image = QImage()

        self.pushButton_Load_File_Calibrate.clicked.connect(self.load_file_calibrate_to_table)
        self.pushButton_Load_File_Distorted.clicked.connect(self.load_file_distorted_to_table)
        self.tableWidget_Tabel_File.clicked.connect(self.load_image_to_label_distorsi)
        self.tableWidget_File_Undistortion.clicked.connect(self.load_image_to_label_undistortion)
        self.pushButton_Delete_File.clicked.connect(self.load_image_delete)

        self.pushButton_Start_Calibration.clicked.connect(self.calibration)
        self.pushButton_Start_Undistorted.clicked.connect(self.undistort_image)


    def select_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Load File')
        return path

    def get_selected_file_path(self):
        row = self.tableWidget_Tabel_File.currentRow()
        cola = self.tableWidget_Tabel_File.currentColumn()
        data = self.tableWidget_Tabel_File.item(row, cola)
        print(row)
        print(cola)
        text = data.text()
        print(self.path2)
        print(text)
        if cola == 0:
            path_lengkap = self.path + "/" + str(text)
            print(self.path2)
        else:
            path_lengkap = self.path2 + "/" + str(text)
        print(path_lengkap)
        return path_lengkap

    def load_file_calibrate_to_table(self):
        self.path = self.select_folder()
        if self.path == '':
            print("No file selected")
        else:
            baca = os.listdir(self.path)
            self.data_calibrate = []
            for file in baca:
                if file.endswith(".jpeg") or file.endswith(".JPG") or file.endswith(".jpg") or file.endswith(".png") or file.endswith(
                        ".bmp") or file.endswith(".gif") or file.endswith(".webp") or file.endswith(
                    ".psd") or file.endswith(
                    ".tfif") or file.endswith(".raw") or file.endswith(".pdf") or file.endswith(
                    ".esp") or file.endswith(
                    ".heif") or file.endswith(".ai"):
                    self.data_calibrate.append(file.replace(".jpeg", ""))
            total = len(self.data_calibrate)
            self.tableWidget_Tabel_File.setRowCount(total)
            for i in range(total):
                self.tableWidget_Tabel_File.setItem(i, 0, QTableWidgetItem(str(self.data_calibrate[i])))
            self.label_Camera.clear()

    def load_file_distorted_to_table(self):
        self.path2 = self.select_folder()
        if self.path2 == '':
            print("No file selected")
        else:
            baca = os.listdir(self.path2)
            self.data_distorted = []
            for file in baca:
                if file.endswith(".jpeg") or file.endswith(".JPG") or file.endswith(".jpg") or file.endswith(".png") or file.endswith(
                        ".bmp") or file.endswith(".gif") or file.endswith(".webp") or file.endswith(
                    ".psd") or file.endswith(
                    ".tfif") or file.endswith(".raw") or file.endswith(".pdf") or file.endswith(
                    ".esp") or file.endswith(
                    ".heif") or file.endswith(".ai"):
                    self.data_distorted.append(file.replace(".jpeg", ""))
            total = len(self.data_distorted)
            self.tableWidget_Tabel_File.setRowCount(total)
            for i in range(total):
                self.tableWidget_Tabel_File.setItem(i, 1, QTableWidgetItem(str(self.data_distorted[i])))
            self.label_Camera.clear()

    def Thread_load(self):
        t1 = Thread(target=self.load_file_distorted_to_table)
        t1.start()

    def load_image_to_label_distorsi(self):
        path = self.get_selected_file_path()
        print(path)
        self.label_Camera.setPixmap(QPixmap(path))
        self.label_Camera.setScaledContents(True)

    def load_image_delete(self):
        selected_rows = self.tableWidget_Tabel_File.selectedIndexes()
        if len(selected_rows) == 0:
            print("Tidak ada gambar yang dipilih")
        else:
            for row in selected_rows:
                file_name = self.tableWidget_Tabel_File.item(row.row(), 0).text()
                file_path_jpeg = os.path.join(self.path, file_name + ".jpeg")
                file_path_jpg = os.path.join(self.path, file_name + ".jpg")
                file_path_png = os.path.join(self.path, file_name + ".png")
                file_path_bmp = os.path.join(self.path, file_name + ".bmp")
                file_path_gif = os.path.join(self.path, file_name + ".gif")
                file_path_webp = os.path.join(self.path, file_name + ".webp")
                file_path_psd = os.path.join(self.path, file_name + ".psd")
                file_path_tfif = os.path.join(self.path, file_name + ".tfif")
                file_path_raw = os.path.join(self.path, file_name + ".raw")
                file_path_pdf = os.path.join(self.path, file_name + ".pdf")
                file_path_esp = os.path.join(self.path, file_name + ".esp")
                file_path_heif = os.path.join(self.path, file_name + ".heif")
                file_path_ai = os.path.join(self.path, file_name + ".ai")
                try:
                    if os.path.isfile(file_path_jpeg):
                        os.remove(file_path_jpeg)
                    if os.path.isfile(file_path_jpg):
                        os.remove(file_path_jpg)
                    if os.path.isfile(file_path_png):
                        os.remove(file_path_png)
                    if os.path.isfile(file_path_bmp):
                        os.remove(file_path_bmp)
                    if os.path.isfile(file_path_gif):
                        os.remove(file_path_gif)
                    if os.path.isfile(file_path_webp):
                        os.remove(file_path_webp)
                    if os.path.isfile(file_path_psd):
                        os.remove(file_path_psd)
                    if os.path.isfile(file_path_tfif):
                        os.remove(file_path_tfif)
                    if os.path.isfile(file_path_raw):
                        os.remove(file_path_raw)
                    if os.path.isfile(file_path_pdf):
                        os.remove(file_path_pdf)
                    if os.path.isfile(file_path_esp):
                        os.remove(file_path_esp)
                    if os.path.isfile(file_path_heif):
                        os.remove(file_path_heif)
                    if os.path.isfile(file_path_ai):
                        os.remove(file_path_ai)
                    self.tableWidget_Tabel_File.removeRow(row.row())
                    self.label_Camera.clear()
                except:
                    print("Gagal menghapus file")


    def calibration(self):
        cols = int(self.lineEdit_Rows.text())
        rows = int(self.lineEdit_Cols.text())

        # Set the termination criteria for the corner sub-pixel algorithm
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
        objp = np.zeros((int(rows) * int(cols), 3), np.float32)
        objp[:, :2] = np.mgrid[0:int(rows), 0:int(cols)].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane


        for fname in os.listdir(self.path):
            # Read the image
            print(fname)
            img = cv.imread(self.path + "/" + fname)
            # resize the image 1/4 the size
            img = cv.resize(img, (0, 0), fx=0.25, fy=0.25)
            # Convert to grayscale
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # cv.imshow('img', gray)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (int(rows), int(cols)), None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                cv.drawChessboardCorners(img, (rows, cols), corners2, ret)
                cv.imshow('img', img)

                # save the img to folder "hasilKalib"
                cv.imwrite("hasilKalib/" + fname, img)
                print("Gambar " + fname + " berhasil disimpan")

        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        np.savez("matriks_calibration.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        self.load_file_hasilKalib_to_table()
        print("hasil panggil")


    def get_selected_file_path2(self):
        row = self.tableWidget_File_Undistortion.currentRow()
        cola = self.tableWidget_File_Undistortion.currentColumn()
        data = self.tableWidget_File_Undistortion.item(row, cola)
        print(row)
        print(cola)
        text = data.text()
        print(self.path2)
        print(text)
        if cola == 0:
            path_lengkap = self.pathhasilKalib + "/" + str(text)
            print(self.path2)
        else:
            path_lengkap = self.pathhasilUndistort + "/" + str(text)
        print(path_lengkap)
        return path_lengkap

    def load_file_hasilKalib_to_table(self):
        print("asdasd")
        baca1 = os.listdir('hasilKalib')
        self.data_hasil_calibration = []
        for file in baca1:
            print(file)
            # cv.imshow(baca + "/" + file)
            print(('hasilKalib' + "/" + file))
            # cv.imshow(('hasilKalib' + "/" + file))
            if file.endswith(".jpeg") or file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".png") or file.endswith(
                 ".bmp") or file.endswith(".gif") or file.endswith(".webp") or file.endswith(
                 ".psd") or file.endswith(
                 ".tfif") or file.endswith(".raw") or file.endswith(".pdf") or file.endswith(
                 ".esp") or file.endswith(
                 ".heif") or file.endswith(".ai"):
                self.data_hasil_calibration.append(file.replace(".jpeg", ""))
                print("berhasil load")
        total = len(self.data_hasil_calibration)
        self.tableWidget_File_Undistortion.setRowCount(total)
        for i in range(total):
            self.tableWidget_File_Undistortion.setItem(i, 0, QTableWidgetItem(str(self.data_hasil_calibration[i])))
        self.label_Proses.clear()

    def thread_load2(self):
        t1 = Thread(target=self.load_file_hasilKalib_to_table)
        t1.start()

    def load_image_to_label_undistortion(self):
        path = self.get_selected_file_path2()
        print(path)
        self.label_Proses.setPixmap(QPixmap(path))
        self.label_Proses.setScaledContents(True)

    def undistort_image(self):
        # Load the camera calibration parameters
        data = np.load("matriks_calibration.npz")
        mtx, dist = data["mtx"], data["dist"]

        for fname in os.listdir(self.path2):
            print(fname)
            # Read an image
            img = cv.imread(self.path2 + "/" + fname)
            # resize the image 1/4 the size
            img = cv.resize(img, (0, 0), fx=0.25, fy=0.25)

            # Undistort the image
            h, w = img.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

            # undistort
            dst = cv.undistort(img, mtx, dist, None, newcameramtx)
            cv.imshow('img', dst)
            cv.imwrite("hasilUndistort/" + fname, dst)

            self.load_file_hasilUndistort_to_table()

            # crop the image
            # x, y, w, h = roi
            # dst = dst[y:y + h, x:x + w]
            # cv.imwrite('crop.png', dst)

    def load_file_hasilUndistort_to_table(self):
        print("asdasd")
        baca1 = os.listdir('hasilUndistort')
        self.data_hasil_undistort = []
        for file in baca1:
            print(file)
            # cv.imshow(baca + "/" + file)
            print(('hasilUndistort' + "/" + file))
            # cv.imshow(('hasilKalib' + "/" + file))
            if file.endswith(".jpeg") or file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".png") or file.endswith(
                 ".bmp") or file.endswith(".gif") or file.endswith(".webp") or file.endswith(
                 ".psd") or file.endswith(
                 ".tfif") or file.endswith(".raw") or file.endswith(".pdf") or file.endswith(
                 ".esp") or file.endswith(
                 ".heif") or file.endswith(".ai"):
                self.data_hasil_undistort.append(file.replace(".jpeg", ""))
                print("berhasil load")
        total = len(self.data_hasil_undistort)
        self.tableWidget_File_Undistortion.setRowCount(total)
        for i in range(total):
            self.tableWidget_File_Undistortion.setItem(i, 1, QTableWidgetItem(str(self.data_hasil_undistort[i])))
        self.label_Proses.clear()

    def thread_load3(self):
        t1 = Thread(target=self.load_file_hasilUndistort_to_table)
        t1.start()

    def load_image_to_label_undistortion(self):
        path = self.get_selected_file_path2()
        print(path)
        self.label_Proses.setPixmap(QPixmap(path))
        self.label_Proses.setScaledContents(True)




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(window)
    widget.setWindowTitle("Distortion Correction")
    widget.show()
    try:
        sys.exit(app.exec_())
    except:
        print("Exiting")
