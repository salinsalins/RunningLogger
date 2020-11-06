# coding: utf-8
'''
Created on 6 Nov 2020
@author: Sanin
'''
import pyqtgraph

class MplCanvas(FigureCanvas):
    """Class to represent the FigureCanvas widget"""

    def __init__(self):
        # setup Matplotlib Figure and Axis
        self.fig = Figure(tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        # initialization of the super
        super().__init__(self.fig)
        # we define the widget as expandable
        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Expanding)
        # switch interactive mode on
        plt.ion()
        # notify the system of updated policy
        self.updateGeometry()


class MplWidget(QtGui.QWidget):

    def __init__(self, parent=None):
        self.plot = pyqtgraph.PlotWidget()
        # initialization of Qt MainWindow widget
        QtGui.QWidget.__init__(self, parent)
        # create the canvas
        self.canvas = MplCanvas()
        # create a vertical box layout
        self.vbl = QtGui.QVBoxLayout()
        # create and add navigation toolbar
        self.ntb = NavigationToolbar(self.canvas, parent)
        self.vbl.addWidget(self.ntb)
        # add canvas widget to layout
        self.vbl.addWidget(self.canvas)
        # set the layout to the widget
        self.setLayout(self.vbl)
