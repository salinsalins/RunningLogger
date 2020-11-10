# coding: utf-8
"""
Created on Oct 29, 2020

@author: sanin
"""
import asyncio
import datetime
import math
import os.path
import sys
import time
import json
import logging
import threading

from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import qApp
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QPushButton
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtCore import QPoint, QSize, Qt
from PyQt5.QtCore import QTimer

import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from TangoAttribute import TangoAttribute
import tango
#from tango.asyncio import DeviceProxy

PROG_NAME = 'RunningLogger'
PROG_VERSION = ' 0.5'
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
        self.out_root = ''
        self.out_folder = None
        self.out_file_name = ''
        self.out_file = None
        self.plot_flag = True
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
        # welcome message
        self.logger.info(PROG_NAME + PROG_VERSION + ' started')
        # default main window parameters
        self.resize(QSize(480, 640))                 # size
        self.move(QPoint(50, 50))                    # position
        self.setWindowTitle(PROG_NAME)        # title
        self.setWindowIcon(QtGui.QIcon('icon.png'))  # icon
        # restore global settings from default location
        self.restore_settings()
        # connect mouse button press event
        self.cid = self.mplWidget.canvas.mpl_connect('button_press_event', self.action)
        # self.mplWidget.canvas.mpl_disconnect(cid)
        # additional decorations
        # plt.style.use('ggplot')
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()
        self.tableWidget.setColumnWidth(0, 25)
        self.tableWidget.setColumnWidth(1, 25)
        self.tableWidget.setColumnWidth(2, 25)
        # Defile callback task and start timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_handler)
        self.timer.start(self.timer_period)
        # reading thread
        self.thread = myThread("Thread", 1)
        self.thread.daemon = True
        self.thread.start()

    def on_quit(self):
        self.close_output_file()
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
            if 'out_root' in self.config:
                self.out_root = self.config['out_root']
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
                                           readonly=True
                                           )
                    self.attributes[name]['tango'] = tattr
                    if tattr.device_proxy is not None:
                        count += 1
                    self.attributes[name]['x'] = np.full(self.draw_points, np.nan)
                    self.attributes[name]['y'] = np.full(self.draw_points, np.nan)
                    self.attributes[name]['quality'] = False
                    self.attributes[name]['task'] = None
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
                    pb = QPushButton()
                    #pb.cli
                    self.attributes[name]['status'] = pb
                    pb.setFixedWidth(25)
                    pb.setStyleSheet('background-color: rgb(0, 255, 0);')
                    table.setCellWidget(row, 1, pb)
                    pb = QPushButton()
                    self.attributes[name]['pb_color'] = pb
                    pb.setFixedWidth(25)
                    #pb.setStyleSheet('background-color: rgb(0, 255, 0);')
                    table.setCellWidget(row, 2, pb)
                    try:
                        aname = tattr.config.name
                    except:
                        aname = tattr.attribute_name
                    self.attributes[name]['label'] = aname
                    table.setItem(row, 3, QTableWidgetItem(aname))
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
        # n = self.draw_points
        t1 = time.time()
        # if not hasattr(self, 't0'):
        #     self.t0 = time.time()
        #     self.y = np.full(n, np.nan)
        #     self.x = np.zeros(n)
        #     self.index = 0
        # t = time.time()
        # tt = (t - self.t0)/ 50.0 * 2.0 * np.pi
        # self.x[:-1] = self.x[1:]
        # self.y[:-1] = self.y[1:]
        # self.x[-1] = t
        # self.y[-1] = np.sin(tt)
        self.ai = 0
        axes = self.axes[self.ai]
        if self.plot_flag:
            axes.clear()
        # outstr = ''
        for an in self.attributes:
            ai = self.attributes[an]
            # if 'x' not in ai:
            #     ai['x'] = np.full(n, np.nan)
            #     ai['y'] = np.full(n, np.nan)
            # ai['x'][:-1] = ai['x'][1:]
            # ai['y'][:-1] = ai['y'][1:]
            # tattr = ai['tango']
            # try:
            #     tattr.read()
            #     y = tattr.value()
            #     x = tattr.attribute_time()
            #     quality = tattr.is_valid()
            # except:
            #     y = np.nan
            #     y = self.y[-1] + len(an)
            #     x = time.time()
            #     x = self.x[-1]
            #     quality = False
            # ai['x'][-1] = x
            # ai['y'][-1] = y
            # if quality:
            #     ai['status'].setStyleSheet('background-color: rgb(0, 255, 0);')
            # else:
            #     ai['status'].setStyleSheet('background-color: rgb(255, 0, 0);')
            #     #ai['y'][-1] = np.nan
            if ai['quality']:
                ai['status'].setStyleSheet('background-color: rgb(0, 255, 0);')
            else:
                ai['status'].setStyleSheet('background-color: rgb(255, 0, 0);')
            if ai['cb'].isChecked():
                if 'color' not in ai:
                    line = axes.plot(ai['x'], ai['y'], label=ai['label'])
                    cl = line[0].get_color()
                    ai['color'] = cl
                    ai['pb_color'].setStyleSheet('background-color: %s;' % cl)
                else:
                    if self.plot_flag:
                        line = axes.plot(ai['x'], ai['y'], color = ai['color'], label=ai['label'])
            #if not math.isnan(y) and y != ai['y'][-2]:
            # if y != ai['y'][-2]:
            #     outstr += '%s; %s; %s\n' % (an, x, y)
            if time.time() - t1 > (self.timer_period * 0.7):
                self.logger.warning('Cycle time exceeded processing %s', an)
                break
        # if outstr != '':
        #     self.make_output_folder()
        #     self.out_file = self.open_output_file()
        #     if self.out_file is not None:
        #         self.out_file.write(outstr)
        #         self.close_output_file()
        self.mplWidget.canvas.draw()

    def make_output_folder(self):
        of = os.path.join(self.out_root, self.get_output_folder())
        try:
            if not os.path.exists(of):
                os.makedirs(of)
                self.logger.log(logging.DEBUG, "Output folder %s has been created", of)
            self.out_folder = of
            return True
        except:
            self.logger.log(logging.WARNING, "Can not create output folder %s", of)
            self.out_folder = None
            return False

    def get_output_folder(self):
        ydf = datetime.datetime.today().strftime('%Y')
        mdf = datetime.datetime.today().strftime('%Y-%m')
        ddf = datetime.datetime.today().strftime('%Y-%m-%d')
        folder = os.path.join(ydf, mdf, ddf)
        return folder

    def open_output_file(self, folder=None):
        if folder is None:
            folder = self.get_output_folder()
        self.out_file_name = os.path.join(folder, self.get_output_file_name())
        try:
            lgf = open(self.out_file_name, 'a')
            return lgf
        except:
            self.logger.warning("Can not open output file %s", self.out_file_name)
            self.print_exception_info()
            return None

    def close_output_file(self):
        if self.out_file.closed:
            return
        try:
            self.out_file.flush()
            self.out_file.close()
            return True
        except:
            self.print_exception_info()
            return False

    def get_output_file_name(self):
        logfn = datetime.datetime.today().strftime('%Y-%m-%d.log')
        return logfn

    def date_time_stamp(self):
        return datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    def time_stamp(self):
        return datetime.datetime.today().strftime('%H:%M:%S')

    def action(self, mouse_event):
        if not mouse_event.dblclick:
            return
        print('dblclick')
        self.plot_flag = not self.plot_flag
        if self.plot_flag:
           self.mplWidget.ntb.hide()
           print('hide')
        else:
           self.mplWidget.ntb.show()
           print('show')


class myThread (threading.Thread):
    def __init__(self, name, counter):
        threading.Thread.__init__(self)
        self.threadID = counter
        self.name = name

    def run(self):
        print("\nStarting " + self.name)
        try:
            while True:
                self.read_attributes()
                #time.sleep(0.5)
            # asyncio.run(async_test())
        except:
            print('exception')
        print("Exiting " + self.name)

    def read_attributes(self):
        outstr = ''
        t0 = time.time()
        for an in main_window.attributes:
            ai = main_window.attributes[an]
            if ai['task'] is None:
                tattr = ai['tango']
                ai['task'] = asyncio.create_task(tattr.async_read())
        for an in main_window.attributes:
            ai = main_window.attributes[an]
            tattr = ai['tango']
            task = ai['task']
            if task is not None:
                if not task.done():
                    continue
                ai['task'] = None
                if task.exception() is None:
                    y = tattr.value()
                    x = tattr.attribute_time()
                    quality = tattr.is_valid()
                else:
                    y = np.nan
                    x = time.time()
                    quality = False
                if y is None:
                    y = np.nan
                ai['x'][:-1] = ai['x'][1:]
                ai['y'][:-1] = ai['y'][1:]
                ai['x'][-1] = x
                ai['y'][-1] = y
                ai['quality'] = quality
                if y != ai['y'][-2] and not (math.isnan(y) and math.isnan(ai['y'][-2])):
                    outstr += '%s; %s; %s\n' % (an, x, y)
                if time.time() - t0 > (main_window.timer_period * 0.7):
                    main_window.logger.warning('Cycle time exceeded processing %s', an)
                    break
        if outstr != '':
            main_window.make_output_folder()
            main_window.out_file = main_window.open_output_file()
            if main_window.out_file is not None:
                main_window.out_file.write(outstr)
                main_window.close_output_file()
        dt = main_window.timer_period - (time.time() - t0)
        if dt > 0.0:
            await asyncio.sleep(dt)
        else:
            await asyncio.sleep(0)


tt0 = 0.0
async def async_test():
    global tt0
    print('Start async_test ...')
    while True:
        await asyncio.sleep(0.2)
        if tt0 > 0.0:
            if tt0 - time.time() > 0.22:
                print('tt esceeded', tt0 - time.time())
            tt0 = time.time()
        print('async_test timer')
    print('... async_test')

class DispatcherThread (threading.Thread):
    def __init__(self, name, counter):
        threading.Thread.__init__(self)
        self.threadID = counter
        self.name = name

    def run(self):
        print("\nStarting " + self.name)
        asyncio.run(async_read_attributes())
        print("Exiting " + self.name)


async def async_read_attributes():
    while True:
        outstr = ''
        t0 = time.time()
        for an in main_window.attributes:
            ai = main_window.attributes[an]
            tattr = ai['tango']
            try:
                tattr.read()
                y = tattr.value()
                x = tattr.attribute_time()
                quality = tattr.is_valid()
            except:
                y = np.nan
                x = time.time()
                quality = False
            if y is None:
                y = np.nan
            ai['x'][:-1] = ai['x'][1:]
            ai['y'][:-1] = ai['y'][1:]
            ai['x'][-1] = x
            ai['y'][-1] = y
            ai['quality'] = quality
            # if quality:
            #     ai['status'].setStyleSheet('background-color: rgb(0, 255, 0);')
            # else:
            #     ai['status'].setStyleSheet('background-color: rgb(255, 0, 0);')
            # ai['y'][-1] = np.nan
            # if not math.isnan(y) and y != ai['y'][-2]:
            if y != ai['y'][-2] and not (math.isnan(y) and math.isnan(ai['y'][-2])):
                outstr += '%s; %s; %s\n' % (an, x, y)
            if time.time() - t0 > (main_window.timer_period * 0.7):
                main_window.logger.warning('Cycle time exceeded processing %s', an)
                break
        if outstr != '':
            main_window.make_output_folder()
            main_window.out_file = main_window.open_output_file()
            if main_window.out_file is not None:
                main_window.out_file.write(outstr)
                main_window.close_output_file()
        dt = main_window.timer_period - (time.time() - t0)
        if dt > 0.0:
            time.sleep(dt)


if __name__ == '__main__':
    ## config loggin
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
