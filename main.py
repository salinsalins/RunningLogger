# coding: utf-8
"""
Created on Oct 29, 2020

@author: sanin
"""

import os.path
import sys
import time
import json
import logging

from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import qApp
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtWidgets import QMessageBox
from PyQt5 import uic
from PyQt5.QtCore import QPoint, QSize
from PyQt5.QtCore import QTimer

import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

prog_name = 'RunningLogger'
prog_version = ' 0.1'
settings_file_name = prog_name + '.json'
init_script = prog_name + '_init.py'
ui_file = prog_name + '.ui'
log_file = prog_name + '.log'
data_file = prog_name + '.dat'


class MainWindow(QMainWindow):
    """Customization for Qt Designer created window"""

    def __init__(self, parent=None):
        # initialization of the superclass
        super(MainWindow, self).__init__(parent)
        # load the GUI
        uic.loadUi(ui_file, self)
        # turn plot interactive mod on
        plt.ion()
        # connect the signals with the slots
        # self.pushButton.clicked.connect(self.next_clicked)
        # self.pushButton_2.clicked.connect(self.selectFolder)
        # self.pushButton_4.clicked.connect(self.processFolder)
        # self.pushButton_6.clicked.connect(self.pushPlotButton)
        # self.pushButton_7.clicked.connect(self.erasePicture)
        # self.comboBox_2.currentIndexChanged.connect(self.selectionChanged)
        # menu actions connection
        # self.actionOpen.triggered.connect(self.selectFolder)
        # self.actionQuit.triggered.connect(qApp.quit)
        # self.actionPlot.triggered.connect(self.showPlot)
        # self.actionLog.triggered.connect(self.showLog)
        # self.actionParameters.triggered.connect(self.showParameters)
        self.actionAbout.triggered.connect(self.show_about)
        # variables definition
        self.prog_dir = os.getcwd()
        self.config = {}
        # self.next_clicked_flag = False
        # self.folderName = ''
        # self.fleNames = []
        # self.nx = 0
        # self.data = None
        # self.scanVoltage = None
        # self.paramsAuto = None
        # self.paramsManual = {}
        # configure logging
        self.logger = logging.getLogger(prog_name + prog_version)
        self.logger.setLevel(logging.DEBUG)
        self.f_str = '%(asctime)s,%(msecs)3d %(levelname)-7s %(filename)s %(funcName)s(%(lineno)s) %(message)s'
        self.log_formatter = logging.Formatter(self.f_str, datefmt='%H:%M:%S')
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(self.log_formatter)
        self.logger.addHandler(self.console_handler)
        # self.file_handler = logging.FileHandler(log_file)
        # self.file_handler.setFormatter(self.log_formatter)
        # self.logger.addHandler(self.file_handler)
        # Default main window parameters
        # self.resize(QSize(480, 640))                 # size
        # self.move(QPoint(50, 50))                    # position
        # self.setWindowTitle(APPLICATION_NAME)        # title
        # self.setWindowIcon(QtGui.QIcon('icon.png'))  # icon
        # welcome message
        self.logger.info(prog_name + prog_version + ' started')
        # restore global settings from default location
        self.restore_settings()
        # connect mouse button press event
        # self.cid = self.mplWidget.canvas.mpl_connect('button_press_event', self.onclick)
        # self.mplWidget.canvas.mpl_disconnect(cid)
        # Defile callback task and start timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_handler)
        self.timer.start(0.5)

    def on_quit(self):
        # save global settings
        self.save_settings()

    def show_about(self):
        QMessageBox.information(self, 'About', prog_name + ' Version ' + prog_version +
                                '\nBeam emittance calculation program.', QMessageBox.Ok)

    def save_settings(self, folder='', fileName=settings_file_name):
        full_name = os.path.join(str(folder), fileName)
        try:
            # save window size and position
            p = self.pos()
            s = self.size()
            self.config['main_window'] = {'size': (s.width(), s.height()), 'position': (p.x(), p.y())}
            #
            with open(full_name, 'w', encoding='utf-8') as configfile:
                configfile.write(json.dumps(self.config, indent=4))
            self.logger.info('Configuration saved to %s' % full_name)
            return True
        except:
            self.logger.info('Configuration save error to %s' % full_name)
            self.print_exception_info()
            return False

    def restore_settings(self, folder='', file_name=settings_file_name):
        full_file_name = os.path.join(str(folder), file_name)
        try:
            with open(full_file_name, 'r', encoding='utf-8') as configfile:
                s = configfile.read()
                self.config = json.loads(s)
            # restore window size and position
            if 'main_window' in self.config:
                self.resize(QSize(self.config['main_window']['size'][0], self.config['main_window']['size'][1]))
                self.move(QPoint(self.config['main_window']['position'][0], self.config['main_window']['position'][1]))
            #
            self.logger.info('Configuration restored from %s' % full_file_name)
            return True
        except:
            self.logger.warning('Configuration restore error from %s' % full_file_name)
            self.print_exception_info()
            return False

    def print_exception_info(self):
        (tp, value) = sys.exc_info()[:2]
        self.logger.info('Exception %s %s' % (str(tp), str(value)))
        self.logger.debug('Exception', exc_info=True)

    def exec_init_script(self, folder=None, file_name=init_script):
        if folder is None:
            folder = self.folderName
        full_name = os.path.join(str(folder), file_name)
        try:
            exec(open(full_name).read(), globals(), locals())
            self.logger.debug('Init script %s executed', full_name)
        except FileNotFoundError:
            self.logger.debug('Init script %s not found', full_name)
        except:
            self.logger.info('Init script %s error.', full_name)
            self.logger.debug('Exception info', exc_info=True)

    def plot(self, *args, **kwargs):
        axes = self.mplWidget.canvas.ax
        axes.plot(*args, **kwargs)
        # zoplot()
        # xlim = axes.get_xlim()
        # axes.plot(xlim, [0.0, 0.0], color='k')
        # axes.set_xlim(xlim)
        axes.grid(True)
        #axes.legend(loc='best')
        self.mplWidget.canvas.draw()

    def clear(self, force=False):
        if force or self.checkBox.isChecked():
            # clear the axes
            self.erase()

    def erase(self):
        self.mplWidget.canvas.ax.clear()
        self.mplWidget.canvas.draw()

    def timer_handler(self):
        if not self.pushButton.isChecked():
            return
        n = 10000
        t1 = time.time()
        if not hasattr(self, 't0'):
            self.t0 = time.time()
            self.y = np.zeros(n)
            self.index = 0
        t = (time.time() - self.t0) / 100.0 * 2.0 * np.pi
        self.y[self.index] = np.sin(t)
        self.index += 1
        n1 = self.index - 1000
        n2 = self.index
        if self.index >= n:
            self.index = 0
        if n1 < 0:
            nd =np.concatenate([np.arange(n+n1, n), np.arange(0, n2)])
        else:
            nd = np.arange(n1, n2)
        axes = self.mplWidget.canvas.ax
        axes.clear()
        axes.plot(self.y[nd])
        k = 100
        for i in range(k):
            axes.plot(self.y[nd]*2.0*i/(k-1))
            if time.time() - t1 > 0.4:
                print(i)
                break
        self.mplWidget.canvas.draw()



if __name__ == '__main__':
    # create the GUI application
    app = QApplication(sys.argv)
    # instantiate the main window
    main_window = MainWindow()
    app.aboutToQuit.connect(main_window.on_quit)
    # show it
    main_window.show()
    # start the Qt main loop execution, exiting from this script
    # with the same return code of Qt application
    sys.exit(app.exec_())
