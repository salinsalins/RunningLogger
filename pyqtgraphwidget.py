# coding: utf-8
'''
Created on 6 Nov 2020
@author: Sanin
'''
import pyqtgraph
from PyQt5.QtWidgets import QVBoxLayout


class PyQtGraphWidget(pyqtgraph.PlotWidget):

    def __init__(self, parent=None):
        self.plot = pyqtgraph.PlotWidget()
        super().__init__(self, parent)
        self.vbl = QVBoxLayout()
        # add widgets to layout
        self.vbl.addWidget(self.plot)
        # set the layout to the widget
        self.setLayout(self.vbl)
