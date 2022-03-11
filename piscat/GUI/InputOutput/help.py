# from PySide2.QtCore import *
# from PySide2.QtWidgets import *
# from PySide2.QtGui import *
# from PySide2.QtWebEngineWidgets import *

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView

import os
import sys


class Help(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(Help, self).__init__(*args, **kwargs)

        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("https://piscat.readthedocs.io/"))
        self.setCentralWidget(self.browser)
        self.show()


# class Help(QMainWindow):
#
#     def __init__(self, *args, **kwargs):
#         super(Help, self).__init__(*args, **kwargs)
#
#         # url = QUrl('https://piscat.readthedocs.io/')
#         # if not QDesktopServices.openUrl(url):
#         #     QMessageBox.warning(self, 'Open Url', 'Could not open url')
#
#         self.browser = QWebEngineView()
#         self.browser.setUrl(QUrl("https://piscat.readthedocs.io/"))
#
#         self.setCentralWidget(self.browser)
#
#         self.show()
