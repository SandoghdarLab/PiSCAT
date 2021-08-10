from PySide2.QtCore import *
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtWebEngineWidgets import *


class Help(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(Help, self).__init__(*args, **kwargs)

        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("https://piscat.readthedocs.io/"))

        self.setCentralWidget(self.browser)

        self.show()
