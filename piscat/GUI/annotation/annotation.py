"""
__author__ = "Houman Mirzaalian D."
__license__ = "GPL"
__email__ = "hmirzaa@mpl.mpg.de
"""

from base64 import b64encode, b64decode
import json
import re
import pandas as pd

from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *

CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT = Qt.PointingHandCursor
CURSOR_DRAW = Qt.CrossCursor
CURSOR_MOVE = Qt.ClosedHandCursor
CURSOR_GRAB = Qt.OpenHandCursor


class LineWidthDialog(QDialog):
    '''A window with a slider to set line width of annotations
    '''

    def __init__(self, allshapes=None, scene=None, parent=None):
        super(LineWidthDialog, self).__init__(parent)
        self.setWindowTitle("Line width editor")
        self.scene = scene
        self.allshapes = allshapes
        self.lwidth = allshapes[0].point_size

        self.form = QGridLayout(self)

        self.form.addWidget(QLabel("Please set line width"), 0, 0)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBothSides)

        self.slider.setMinimum(0)
        self.slider.setMaximum(10)
        self.slider.setValue(self.lwidth)
        self.form.addWidget(self.slider, 1, 0)
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal)

        self.form.addWidget(self.buttonBox, 2, 0)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.accepted.connect(self.extractInputs)

    def extractInputs(self):
        self.size = self.slider.value()
        if self.size != self.lwidth:
            self.scene.point_size = self.size
            for shape in self.allshapes:
                shape.point_size = self.size


class EpsilonSliderDialog(QDialog):
    '''A window with a slider to set attraction epsilon, i.e. pull the cursor
    to the first point of the shape if epsilon close
    '''

    def __init__(self, scene=None, parent=None):
        super(EpsilonSliderDialog, self).__init__(parent)
        self.setWindowTitle("Attraction epsilon editor")
        self.scene = scene

        self.epsilon = self.scene.epsilon

        self.form = QGridLayout(self)

        self.form.addWidget(QLabel("Please set epsilon"), 0, 0)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBothSides)

        self.slider.setMinimum(0)
        self.slider.setMaximum(2 * self.epsilon)
        self.slider.setValue(self.epsilon)
        self.form.addWidget(self.slider, 1, 0)
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal)

        self.form.addWidget(self.buttonBox, 2, 0)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.accepted.connect(self.extractInputs)

    def extractInputs(self):
        self.newepsilon = self.slider.value()
        if self.newepsilon != self.epsilon:
            self.scene.epsilon = self.newepsilon


class PropertiesWindow(QDialog):
    '''A window that pops up on right click (see mousepressevent in qscene).
    Lets the user reassign an annotated ("closed") shape to a different label class.
    '''

    def __init__(self, shape=None, all_labels=[None], scene=None, parent=None):
        super(PropertiesWindow, self).__init__(parent)
        self.shape = shape
        self.points = self.shape.points
        self.objtype = self.shape.objtype
        self.label = self.shape.label
        self.all_labels = all_labels
        self.setWindowTitle("Properties editor")
        self.label_names = [label.name for label in all_labels if label is not None]
        self.editable_status = self.shape.editable

        self.form = QGridLayout(self)

        self.form.addWidget(QLabel("Object properties"), 0, 0)

        self.qbox = QComboBox(self)
        self.qbox.addItem(self.label)
        [self.qbox.addItem(label) for label in self.label_names if label != self.label]

        self.form.addWidget(QLabel("Type:"), 1, 0)
        self.form.addWidget(QLabel(self.objtype), 1, 1)
        self.form.addWidget(QLabel("Label:"), 2, 0)
        self.form.addWidget(self.qbox, 2, 1)

        copybutton = QPushButton('Copy', self)
        copybutton.clicked.connect(scene.copySelected)

        self.editbutton = QPushButton('Set editable', self)
        self.editbutton.setCheckable(True)
        self.editbutton.clicked.connect(self.updateCheckable)
        self.editableColor()

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal)
        self.buttonBox.addButton(copybutton, QDialogButtonBox.AcceptRole)
        self.buttonBox.addButton(self.editbutton, QDialogButtonBox.ActionRole)
        self.form.addWidget(self.buttonBox, 3, 0)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.accepted.connect(self.extractInputs)

    def editableColor(self):
        if self.editable_status:
            color = QColor(255, 0, 0)
        else:
            color = QColor(255, 255, 255)

        palette = self.editbutton.palette()
        role = self.editbutton.backgroundRole()
        palette.setColor(role, color)
        self.editbutton.setPalette(palette)
        self.editbutton.setAutoFillBackground(True)
        return

    def updateCheckable(self, checked=False):
        if checked and not self.shape.editable:
            self.editable_status = True
        else:
            self.editable_status = False
        self.editableColor()
        return

    def extractInputs(self):
        self.shape.editable = self.editable_status
        oldlabel = self.shape.label
        newlabel = self.qbox.currentText()
        if oldlabel != newlabel:
            if oldlabel in self.label_names:
                self.all_labels[self.label_names.index(oldlabel)].untieShape(self.shape)
                newlabelclass = self.all_labels[self.label_names.index(newlabel)]
                newlabelclass.assignObject(self.shape)
        return


class LabelDialog(QDialog):
    '''Label editor form. Given # of new classes to initiate (nlabels) pops up
    a form with a line editor to name a class and a button to set label color.
    If nlabel=0, the form Features only already initialized labels for editing.
    '''

    def __init__(self, nlabels=1, prelabels=None, parent=None):
        super(LabelDialog, self).__init__(parent)
        self.names = []
        self.colors = [None for i in range(nlabels)]
        self.prelabels = prelabels
        self.setWindowTitle("Initialize labels")
        self.lineedits = []

        self.form = QGridLayout(self)
        self.form.addWidget(QLabel("Please give description of labels below:"), 0, 0)
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal)

        if prelabels is not None:
            self.prenames = [label.name for label in prelabels]
            self.precolors = [label.fillColor for label in prelabels]
            nprelabels = len(prelabels)
            self.colors = self.colors + nprelabels * [None]
            [self.addLabelInfo(index, prelabdata=[self.prenames[index], self.precolors[index]]) for index in
             range(nprelabels)]
            [self.addLabelInfo(index) for index in range(nprelabels, nprelabels + nlabels)]
            self.form.addWidget(self.buttonBox, 2 * (nprelabels + nlabels) + 1, 0)
        else:
            [self.addLabelInfo(index) for index in range(nlabels)]
            self.form.addWidget(self.buttonBox, 2 * nlabels + 1, 0)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.accepted.connect(self.extractInputs)
        self.rejected.connect(self.clearInputs)

    def fillcolornone(self):
        if None in self.colors:
            ind = self.colors.index(None)
            fillcolor1 = int(255 * ind / len(self.colors))
            fillcolor2 = int(255 / len(self.colors))
            fillcolor3 = 255
            self.colors[self.colors.index(None)] = QColor(fillcolor1, fillcolor2, fillcolor3)
            self.fillcolornone()
        return

    def fillemptynames(self):
        indices = [i for i, name in enumerate(self.names) if name == '']
        for ind in indices:
            self.names[ind] = 'class' + str(ind + 1)
        return

    def extractInputs(self):
        self.names = [lineedit.text() for lineedit in self.lineedits]
        self.fillemptynames()
        self.fillcolornone()
        return

    def clearInputs(self):
        self.names = []
        self.colors = [None for i in range(len(self.colors))]
        if self.prelabels is not None:
            self.names = self.prenames
            self.colors = self.precolors
        return

    def addLabelInfo(self, labelindex, prelabdata=2 * [None]):
        lineEdit = QLineEdit(self)
        button = QPushButton('Select color', self)
        labelname = QLabel('Label {} name'.format(labelindex + 1))
        labelcolor = QLabel('Label {} color'.format(labelindex + 1))

        if prelabdata[0] is not None:
            lineEdit.setText("{}".format(prelabdata[0]))
            color = prelabdata[1]
            self.colors[labelindex] = color
            palette = button.palette()
            role = button.backgroundRole()
            palette.setColor(role, color)
            button.setPalette(palette)
            button.setAutoFillBackground(True)

        button.clicked.connect(lambda: self.selectColor(labelindex))

        self.lineedits.append(lineEdit)
        self.form.addWidget(labelname, 2 * labelindex + 1, 0)
        self.form.addWidget(lineEdit, 2 * labelindex + 1, 1)
        self.form.addWidget(labelcolor, 2 * labelindex + 2, 0)
        self.form.addWidget(button, 2 * labelindex + 2, 1)

    def selectColor(self, labelindex):
        dialogue = QColorDialog()
        dialogue.exec()
        color = dialogue.selectedColor()
        self.colors[labelindex] = color

        button = QPushButton('Select color', self)
        palette = button.palette()
        role = button.backgroundRole()
        palette.setColor(role, color)
        button.setPalette(palette)
        button.setAutoFillBackground(True)
        button.clicked.connect(lambda: self.selectColor(labelindex))
        self.form.addWidget(button, 2 * labelindex + 2, 1)
        return


class ToolButton(QToolButton):
    '''Overwrite QToolButton to ensure all buttons are of same size
    '''
    minSize = (80, 80)

    def minimumSizeHint(self):
        ms = super(ToolButton, self).minimumSizeHint()
        w1, h1 = ms.width(), ms.height()
        w2, h2 = self.minSize
        ToolButton.minSize = max(w1, w2), max(h1, h2)
        return QSize(*ToolButton.minSize)


def process(filename, default=None):
    '''Return bytes for an image given its path
    '''
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


def newAction(parent, text, slot=None, shortcut=None, icon=None,
              tip=None, checkable=False, enabled=True):
    '''Initialize an action with flags as requested'''
    a = QAction(text, parent)

    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    return a


def distance(delta):
    '''Squared euclidean distance as metric. Returns distance given an elementwise delta
    '''
    return (delta.x() ** 2 + delta.y() ** 2)


class LabelClass(object):
    '''Class to keep record of a label class characteristics with a method to
    assign shapes to it
    '''

    def __init__(self):
        self.polygons = []
        self.fillColor = None
        self.name = None

    def assignObject(self, obj):
        self.polygons.append(obj)
        obj.line_color = self.fillColor
        obj.label = self.name

    def untieShape(self, obj):
        self.polygons.pop(self.polygons.index(obj))


class Annotationscene(object):
    '''Class to store and save outputs as .csv and .json
    '''

    def __init__(self, filename=None):
        self.polygons = None
        self.imagePath = None
        self.imageData = None
        self.filename = None
        self.lineColor = None
        self.imsizes = None
        self.object_types = None
        self.labels = None
        self.savebytes = False

    def shapes_to_pandas(self):
        imsize, objects, types, labels = self.imsizes, self.shapes, self.object_types, self.labels
        df = pd.DataFrame(columns=['width', 'height', 'Object', 'X', 'Y'])
        for i, obj in enumerate(objects):
            X, Y = list(zip(*obj))
            df = df.append(pd.DataFrame(
                {'width': imsize[0], 'height': imsize[1], 'Object': i + 1, 'Type': types[i], 'Label': labels[i], 'X': X,
                 'Y': Y}), ignore_index=True)
        return df

    def save(self):

        self.shapes = [[(point.x(), point.y()) for point in poly] for poly in self.polygons]
        self.shapes_to_pandas().to_csv(re.search(re.compile('(.+?)(\.[^.]*$|$)'), self.filename).group(1) + '.csv',
                                       sep=',')
        if self.savebytes:
            self.imData = b64encode(self.imageData).decode('utf-8')

            try:
                with open(self.filename, 'w') as f:

                    json.dump({
                        'objects': self.shapes,
                        'bin_type': self.object_types,
                        'label': self.labels,
                        'width/height': self.imsizes,
                        'lineColor': self.lineColor,
                        'imagePath': self.imagePath,
                        'imageData': self.imData},
                        f, ensure_ascii=True, indent=2)
            except:
                pass
        else:
            try:
                with open(self.filename, 'w') as f:
                    json.dump({
                        'objects': self.shapes,
                        'bin_type': self.object_types,
                        'label': self.labels,
                        'width/height': self.imsizes,
                        'lineColor': self.lineColor,
                        'imagePath': self.imagePath},
                        f, ensure_ascii=True, indent=2)
            except:
                pass


class Shape(QGraphicsItem):
    '''The main class controlling shape's points, its color, highlight behavior
    '''
    line_color = QColor(0, 6, 255)
    select_line_color = QColor(255, 255, 255)
    vertex_fill_color = QColor(0, 255, 0, 255)
    hvertex_fill_color = QColor(255, 0, 0)
    point_size = 1.5
    hsize = 3.0

    def __init__(self, line_color=None, point_size=None, parent=None):
        super(Shape, self).__init__(parent)
        self.points = []
        self.selected = False
        self.painter = QPainter()
        self.hIndex = None
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.closed = False
        self.points_adjusted = None
        self.objtype = None
        self.label = None
        self.editable = False

        if line_color is not None:
            self.line_color = line_color

        if point_size is not None:
            self.point_size = point_size

    def addPoint(self, point):
        self.setSelected(True)
        if self.points and point == self.points[0]:
            self.closed = True
        else:
            self.points.append(point)

    def popPoint(self):
        if self.points:
            return self.points.pop()
        return None

    def paint(self, painter, option, widget):

        if self.points:
            self.prepareGeometryChange()
            color = self.select_line_color if self.selected else self.line_color
            pen = QPen(color)
            pen.setWidth(self.point_size / 2)
            painter.setPen(pen)
            path = self.shape()
            if self.closed == True:
                path.closeSubpath()
            painter.drawPath(path)
            vertex_path = QPainterPath()
            self.drawVertex(vertex_path, 0)
            [self.drawVertex(vertex_path, i) for i in range(len(self.points))]
            painter.drawPath(vertex_path)
            painter.fillPath(vertex_path, self.vertex_fill_color)

    def drawVertex(self, path, index):
        psize = self.point_size
        if index == self.hIndex:
            psize = self.hsize
        if self.hIndex is not None:
            self.vertex_fill_color = self.hvertex_fill_color
        else:
            self.vertex_fill_color = Shape.vertex_fill_color
        path.addEllipse(self.mapFromScene(self.points[index]), psize, psize)

    def shape(self):
        path = QPainterPath()
        polygon = self.mapFromScene(QPolygonF(self.points))
        path.addPolygon(polygon)
        return path

    def boundingRect(self):
        return self.shape().boundingRect()

    def moveBy(self, tomove, delta):
        if tomove == 'all':
            tomove = slice(0, len(self.points))
        else:
            tomove = slice(tomove, tomove + 1)
        self.points[tomove] = [point + delta for point in self.points[tomove]]

    def highlightVertex(self, index):
        self.hIndex = index

    def highlightClear(self):
        self.hIndex = None
        self.selected = False

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index]

    def __setitem__(self, index, value):
        self.points[index] = value


class SubQGraphicsScene(QGraphicsScene):
    '''Overwrite QGraphicsScene to prescribe actions to mouse events,
    collect annotated shapes and label classes, tracks which mode the program is in
    at any moment (drawing, navigating, moving)
    '''
    NAVIGATION, DRAWING, MOVING = 0, 1, 2
    POLYDRAWING, POLYREADY = 0, 1
    epsilon = 30.0

    def __init__(self, parent=None):
        super(SubQGraphicsScene, self).__init__(parent)
        self.mode = self.NAVIGATION
        self.QGitem = None
        self.polys = []
        self._cursor = CURSOR_DEFAULT
        self.overrideCursor(self._cursor)
        self.line = None
        self.lineColor = QColor(3, 252, 66)
        self.shapeColor = QColor(0, 6, 255)
        self.selectedVertex = None
        self.selectedShape = None
        self.polystatus = self.POLYDRAWING
        self.objtypes = []
        self.labelclasses = []
        self.labelmode = 0  # the default class
        self.initializeClass('default', QColor(0, 6, 255))  # initialize default class
        self.point_size = 1.5

    def drawing(self):
        return self.mode == self.DRAWING

    def navigating(self):
        return self.mode == self.NAVIGATION

    def moving(self):
        return self.mode == self.MOVING

    def polygon_not_finished(self):
        return self.polystatus == self.POLYDRAWING

    def polygonfinished(self):
        return self.polystatus == self.POLYREADY

    def vertexSelected(self):
        return self.selectedVertex is not None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.deleteSelected()
        if event.key() == Qt.Key_K:
            self.copySelected()
        if (event.key() == Qt.Key_Z) and (QApplication.keyboardModifiers() == Qt.ControlModifier):
            self.undoAction()

    def undoAction(self):
        if self.QGitem:
            if len(self.QGitem.points) > 1:
                self.QGitem.popPoint()
                self.line.points[0] = self.QGitem.points[-1]
                self.update()
            else:
                self.removeItem(self.QGitem)
                if self.line:
                    if self.line in self.items():
                        self.removeItem(self.line)

                    self.line.popPoint()
                self.polystatus = self.POLYREADY
                self.QGitem = None
                self.update()

    def refreshShapestoLabels(self, labelclass):
        labelpolygons = labelclass.polygons
        labelclass.polygons = []
        [labelclass.assignObject(shape) for shape in labelpolygons]
        return

    def initializeClasses(self, names, colors):
        for c, name in enumerate(names):
            if (c < len(self.labelclasses)):
                self.initializeClass(name, colors[c], labelclass=self.labelclasses[c])
                self.refreshShapestoLabels(labelclass=self.labelclasses[c])
            else:
                self.initializeClass(name, colors[c])

    def updateColor(self, color):
        self.shapeColor = color
        active_labelclass = self.labelclasses[self.labelmode]
        active_labelclass.fillColor = self.shapeColor
        self.refreshShapestoLabels(labelclass=active_labelclass)
        return

    def setLabelMode(self, classindex):
        labelclass = self.labelclasses[classindex]
        self.labelmode = classindex
        self.shapeColor = labelclass.fillColor
        return

    def initializeClass(self, name, color, labelclass=None):
        ind = None
        if labelclass is None:
            labelclass = LabelClass()
        else:
            ind = self.labelclasses.index(labelclass)
        labelclass.name = name
        labelclass.fillColor = color
        if ind is not None:
            self.labelclasses[ind] = labelclass
        else:
            self.labelclasses.append(labelclass)

    def triggerClosure(self):
        self.finalisepoly(premature=True)

    def mousePressEvent(self, event):
        '''Draw, move vertices/shapes, open properties window'''
        pos = event.scenePos()

        if (event.button() == Qt.RightButton) and not self.drawing():
            if self.selectedShape:
                all_labels = [None]
                if len(self.labelclasses) > 0:
                    all_labels = self.labelclasses
                propdialog = PropertiesWindow(shape=self.selectedShape, all_labels=all_labels, scene=self)
                propdialog.move(event.screenPos())
                propdialogexec = propdialog.exec()

                if propdialogexec:
                    if self.selectedShape.editable:
                        self.selectedShape.closed = False
                        self.QGitem = self.selectedShape
                        p = self.QGitem.points[-1]
                        self.line = Shape(point_size=self.point_size)
                        self.addItem(self.line)
                        self.line.setPos(p)
                        self.line.addPoint(p)
                        self.polystatus = self.POLYDRAWING
                        self.mode = self.DRAWING
                        shapeid = self.polys.index(self.selectedShape)
                        if self.selectedShape in self.polys:
                            self.polys.pop(shapeid)
                            self.objtypes.pop(shapeid)
                            self.QGitem.setZValue(len(self.polys) + 2)
                self.update()

        if self.drawing() & (event.button() == Qt.LeftButton):
            self.overrideCursor(CURSOR_DRAW)
            # update the tail of the pointing line
            if self.line and self.polygon_not_finished():
                self.line.prepareGeometryChange()
                self.line.points[0] = pos
                self.line.setPos(pos)
            # initialize a pointing line for a new polygon
            elif self.line == None or self.polygonfinished():
                self.line = Shape(point_size=self.point_size)
                self.addItem(self.line)
                self.line.setPos(pos)
                self.line.addPoint(pos)

            if self.QGitem:
                # attract the cursor to the start point of the polygon and close it
                self.QGitem.prepareGeometryChange()
                if len(self.QGitem.points) > 1 and self.closeEnough(pos, self.QGitem.points[0]):
                    pos = self.QGitem.points[0]
                    self.overrideCursor(CURSOR_POINT)
                    self.QGitem.highlightVertex(0)

                self.QGitem.addPoint(pos)
                if (self.QGitem.points[0] == pos):
                    self.finalisepoly()


            else:
                self.polystatus = self.POLYDRAWING
                self.QGitem = Shape(point_size=self.point_size)

                self.addItem(self.QGitem)
                self.QGitem.setPos(pos)
                self.QGitem.addPoint(pos)
                self.QGitem.setZValue(len(self.polys) + 1)
            self.update()
            event.accept()

        elif self.moving() & (event.button() == Qt.LeftButton):
            self.overrideCursor(CURSOR_GRAB)
            self.selectShapebyPoint(pos)
            self.prevPoint = pos
            event.accept()
            self.update()

        elif self.navigating():
            self.overrideCursor(CURSOR_GRAB)
            self.update()

    def mouseMoveEvent(self, event):
        '''Track the movement of the cursor and update selections/drawings'''
        pos = event.scenePos()

        if self.drawing():
            self.overrideCursor(CURSOR_DRAW)

            if self.QGitem:
                if len(self.QGitem.points) == 1:  # initialize the pointing line collapsed to a point
                    self.line.points = [self.QGitem.points[0], self.QGitem.points[0]]
                colorLine = self.lineColor
                if len(self.QGitem) > 1 and self.closeEnough(pos, self.QGitem[0]):
                    pos = self.QGitem[0]
                    colorLine = self.QGitem.line_color
                    self.overrideCursor(CURSOR_POINT)
                    self.QGitem.highlightVertex(0)

                if len(self.line.points) == 2:  # update the pointing line
                    self.line.points[1] = pos
                else:  # load the pointing line (if another shape was just created)
                    self.line.addPoint(pos)

                self.line.line_color = colorLine
                self.update()
            return

        # moving shapes/vertices
        if self.moving and Qt.LeftButton & event.buttons():
            self.overrideCursor(CURSOR_GRAB)
            if self.vertexSelected():
                self.selectedShape.prepareGeometryChange()
                if (self.selectedShape.objtype == 'Line') and (QApplication.keyboardModifiers() == Qt.ShiftModifier):
                    self.moveShape(self.selectedShape, pos)
                else:
                    self.moveVertex(pos)
                self.update()
            elif self.selectedShape and self.prevPoint:
                self.selectedShape.prepareGeometryChange()
                self.moveShape(self.selectedShape, pos)
                self.update()
            return

        # update selections/highlights based on cursor location

        # check if any vertex is epsilon close to the cursor position and find the corresponding shape
        id_point = [[i for i, y in enumerate(poly.points) if distance(pos - y) <= self.epsilon] for poly in self.polys]
        id_shape = [i for i, y in enumerate(id_point) if y != []]

        itemUnderMouse = self.itemAt(pos, QTransform())
        # self.clearShapeSelections()
        # if shape/vertex combination found, highlight vertex and shape
        if id_shape != []:
            self.selectedVertex = id_point[id_shape[0]][0]
            self.selectShape(self.items()[:-1][::-1][id_shape[0]])
            self.selectedShape.highlightVertex(self.selectedVertex)
            self.update()
            return
        elif itemUnderMouse in self.items()[:-1]:  # if the cursor is inside of a shape, highlight it
            self.selectedVertex = None
            self.selectShape(itemUnderMouse)
            self.selectedShape.hIndex = None
            self.update()
            return
        else:  # nothing found: no shape under the cursor, no vertices in vicinity, clear all
            self.selectedVertex = None
            self.update()
            return

        event.accept()

    def mouseReleaseEvent(self, event):
        if self.navigating or (event.button() == Qt.LeftButton and self.selectedShape):
            self.overrideCursor(CURSOR_DEFAULT)
            self.update()
        event.accept()

    def closeEnough(self, p1, p2):
        return distance(p1 - p2) < self.epsilon

    def finalisepoly(self, premature=False):
        if self.QGitem:
            if premature:
                if len(self.QGitem.points) == 1:
                    self.objtypes.append('Point')
                    self.QGitem.objtype = 'Point'
                else:
                    self.objtypes.append('Line')
                    self.QGitem.objtype = 'Line'
            else:
                self.objtypes.append('Polygon')
                self.QGitem.objtype = 'Polygon'
            if self.line:
                self.removeItem(self.line)
                self.line.popPoint()
            self.QGitem.editable = False
            self.polys.append(self.QGitem)
            if self.labelmode is not None:
                labelobject = self.labelclasses[self.labelmode]
                labelobject.assignObject(self.QGitem)
                self.QGitem.label = labelobject.name
            self.QGitem = None
            self.polystatus = self.POLYREADY
            self.update()

    def overrideCursor(self, cursor):
        self._cursor = cursor
        QApplication.setOverrideCursor(cursor)

    def copySelected(self):

        if self.selectedShape:
            shape = self.selectedShape
            c, o, s = [shape.closed, shape.objtype, shape.point_size]
            p = [point for point in shape.points]

            newshape = Shape()

            newshape.points, newshape.closed, newshape.objtype, newshape.point_size = p, c, o, s

            self.polys.append(newshape)
            self.objtypes.append(newshape.objtype)

            newshape.setZValue(len(self.polys) + 1)
            self.addItem(newshape)

            labelid = [label.name for label in self.labelclasses].index(shape.label)
            self.labelclasses[labelid].assignObject(newshape)

            print('Shape copied')
            self.clearShapeSelections()
            self.selectShape(newshape)
            self.update()
            return

    def deleteSelected(self):
        if self.selectedShape:
            shape = self.items()[:-1][::-1].index(self.selectedShape)
            if self.selectedShape in self.polys:
                self.polys.pop(shape)
                self.objtypes.pop(shape)
            self.removeItem(self.selectedShape)
            if self.line:
                if self.line in self.items():
                    self.removeItem(self.line)
                self.line.popPoint()
            labelind = self.findShapeInLabel(self.selectedShape)
            if len(labelind) > 0:
                label, shapeind = labelind[0]
                self.labelclasses[label].polygons.pop(shapeind)
            self.polystatus = self.POLYREADY
            self.selectedShape = None
            self.QGitem = None
            self.clearShapeSelections()
            print('Shape deleted')
            self.update()
            return

    def findShapeInLabel(self, shape):
        if len(self.labelclasses) > 0:
            labelpolys = [l.polygons for l in self.labelclasses]
            return ([(i, label.index(shape)) for i, label in enumerate(labelpolys) if shape in label])
        else:
            return 2 * [None]

    def selectShape(self, shape):
        shape.selected = True
        self.selectedShape = shape
        self.update()

    def selectShapebyPoint(self, point):
        """Select the first shape created which contains this point."""
        if self.vertexSelected():  # A vertex is marked for selection.
            self.selectedShape.highlightVertex(self.selectedVertex)
            return

        itemUnderMouse = self.itemAt(point, QTransform())

        if itemUnderMouse in self.items()[:-1]:
            self.selectShape(itemUnderMouse)
            return

    def clearShapeSelections(self):
        if self.selectedShape:
            self.selectedShape.highlightClear()
            self.selectedShape = None
            self.forceAllSelectionsClear()
            self.update()

    def forceAllSelectionsClear(self):
        for shape in self.items()[:-1]:
            shape.highlightClear()
        self.update()
        return

    def moveVertex(self, pos):
        self.selectedShape.prepareGeometryChange()
        self.selectedShape.moveBy(self.selectedVertex, pos - self.selectedShape[self.selectedVertex])

    def moveShape(self, shape, pos):
        delta = pos - self.prevPoint
        if delta:
            shape.prepareGeometryChange()
            shape.moveBy('all', delta)
            self.prevPoint = pos
            self.update()
            return True
        return False
