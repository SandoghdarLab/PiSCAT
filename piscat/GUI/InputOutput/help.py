from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *


class Help(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(Help, self).__init__(*args, **kwargs)

        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("https://piscat.readthedocs.io/"))

        self.setCentralWidget(self.browser)

        self.show()
