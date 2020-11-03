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

from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import qApp
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QCheckBox
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtCore import QPoint, QSize, Qt
from PyQt5.QtCore import QTimer

import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from TangoAttribute import TangoAttribute

PROG_NAME = 'RunningLogger'
PROG_VERSION = ' 0.1'
settings_file_name = PROG_NAME + '.json'
init_script = PROG_NAME + '_init.py'
ui_file = PROG_NAME + '.ui'
log_file = PROG_NAME + '.log'
data_file = PROG_NAME + '.dat'


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
        self.pushButton_7.clicked.connect(self.erase)
        # self.comboBox_2.currentIndexChanged.connect(self.selectionChanged)
        # menu actions connection
        self.actionQuit.triggered.connect(qApp.quit)
        self.actionAbout.triggered.connect(self.show_about)
        # variables definition
        self.prog_dir = os.getcwd()
        self.config = {}
        self.axes = []
        self.ai = 0
        self.timer_period = 1.0
        self.draw_points = 1000
        self.attributes = {}
        # configure logging
        self.logger = logging.getLogger(PROG_NAME + PROG_VERSION)
        self.logger.setLevel(logging.DEBUG)
        self.f_str = '%(asctime)s,%(msecs)3d %(levelname)-7s %(filename)s %(funcName)s(%(lineno)s) %(message)s'
        self.log_formatter = logging.Formatter(self.f_str, datefmt='%H:%M:%S')
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(self.log_formatter)
        self.logger.addHandler(self.console_handler)
        # logging to file
        # self.file_handler = logging.FileHandler(log_file)
        # self.file_handler.setFormatter(self.log_formatter)
        # self.logger.addHandler(self.file_handler)
        # default main window parameters
        self.resize(QSize(480, 640))                 # size
        self.move(QPoint(50, 50))                    # position
        self.setWindowTitle(PROG_NAME)        # title
        self.setWindowIcon(QtGui.QIcon('icon.png'))  # icon
        # restore global settings from default location
        self.restore_settings()
        # connect mouse button press event
        # self.cid = self.mplWidget.canvas.mpl_connect('button_press_event', self.onclick)
        # self.mplWidget.canvas.mpl_disconnect(cid)
        # additional decorations
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()
        self.tableWidget.setColumnWidth(0, 25)
        # Defile callback task and start timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_handler)
        self.timer.start(self.timer_period)
        # welcome message
        self.logger.info(PROG_NAME + PROG_VERSION + ' started')

    def on_quit(self):
        # save global settings
        self.save_settings()

    def show_about(self):
        QMessageBox.information(self, 'About', PROG_NAME + ' Version ' + PROG_VERSION +
                                '\nBeam emittance calculation program.', QMessageBox.Ok)

    def save_settings(self, folder='', file_name=settings_file_name):
        full_name = os.path.join(str(folder), file_name)
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
            if 'subplots' in self.config:
                fig = self.mplWidget.canvas.fig
                for a in fig.get_axes():
                    fig.delaxes(a)
                self.axes = []
                for i in range(self.config['subplots']["rows"]):
                    ax = fig.add_subplot(self.config['subplots']["rows"], self.config['subplots']["columns"], i+1)
                    self.axes.append(ax)
                    self.mplWidget.canvas.ax = ax
            if 'timer_period' in self.config:
                self.timer_period = self.config['timer_period']
            if 'draw_points' in self.config:
                self.draw_points = self.config['draw_points']
            if 'attributes' in self.config:
                count = 0
                table = self.tableWidget
                for attr in self.config['attributes']:
                    name = attr['name']
                    self.attributes[name] = {}
                    self.attributes[name]['config'] = attr
                    tattr = TangoAttribute(name=name,
                                           use_history=attr['use_history'],
                                           readonly=attr['readonly']
                                           )
                    self.attributes[name]['tango'] = tattr
                    if tattr.device_proxy is not None:
                        count += 1
                    row = table.rowCount()
                    self.attributes[name]['row'] = row
                    table.insertRow(row)
                    #cb = QCheckBox()
                    #cb.setFixedWidth(50)
                    #cb.setLayout(QVBoxLayout())
                    pWidget = QWidget()
                    pCheckbox = QCheckBox()
                    pLayout = QVBoxLayout()
                    pLayout.addWidget(pCheckbox)
                    pLayout.setAlignment(Qt.AlignCenter)
                    pLayout.setContentsMargins(0, 0, 0, 0)
                    pWidget.setLayout(pLayout)
                    self.attributes[name]['cb'] = pCheckbox
                    pCheckbox.setCheckState(QtCore.Qt.Checked)
                    table.setCellWidget(row, 0, pWidget)
                    try:
                        aname = tattr.config.name
                    except:
                        aname = tattr.attribute_name
                    table.setItem(row, 1, QTableWidgetItem(aname))
                if count == 0:
                    self.logger.warning('No valid tango attributes defined')
            else:
                self.logger.warning('No attributes defined')
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
        # axes.legend(loc='best')
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
        n = self.draw_points
        t1 = time.time()
        if not hasattr(self, 't0'):
            self.t0 = time.time()
            self.y = np.zeros(n)
            self.x = np.zeros(n)
            self.index = 0
        t = time.time() - self.t0
        tt = t / 50.0 * 2.0 * np.pi
        self.x[:-1] = self.x[1:]
        self.y[:-1] = self.y[1:]
        self.x[-1] = t
        self.y[-1] = np.sin(tt)
        # nd = np.concatenate([np.arange(self.index+1, n), np.arange(0, self.index+1)])
        # self.index += 1
        # if self.index >= n:
        #     self.index = 0
        # self.ai += 1
        # if self.ai >= len(self.axes):
        #     self.ai = 0
        self.ai = 0
        axes = self.axes[self.ai]
        axes.clear()
        # k = 5
        # yy = self.y * 2.0 / (k - 1)
        # for i in range(k):
        #     axes.plot(self.x, yy * i)
        #     if time.time() - t1 > (self.timer_period * 0.5):
        #         print(i)
        #         break
        for an in self.attributes:
            ai = self.attributes[an]
            if 'x' not in ai:
                ai['x'] = np.zeros(n)
                ai['y'] = np.zeros(n)
            ai['x'][:-1] = ai['x'][1:]
            ai['y'][:-1] = ai['y'][1:]
            tattr = ai['tango']
            try:
                tattr.read()
                y = tattr.value()
                x = tattr.time()
            except:
                y = self.y[-1] + len(an)
                x = self.x[-1]
            ai['x'][-1] = x
            ai['y'][-1] = y
            if ai['cb'].isChecked():
                axes.plot(ai['x'], ai['y'])
            if time.time() - t1 > (self.timer_period * 0.5):
                print('attr', an)
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
