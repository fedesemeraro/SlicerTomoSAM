from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from tomosamLib.tomosamLogic import tomosamLogic
import slicer
import os
import vtk
import qt
import urllib.request
import hashlib


class tomosam(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "TomoSAM"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Federico Semeraro (NASA); Alexandre Quintart (NASA)",
                                    "Sergio Fraile Izquierdo (NASA); Joseph Ferguson (Stanford University)"]
        self.parent.helpText = \
            """TomoSAM helps with the segmentation of 3D data from tomography or other imaging techniques using the Segment Anything Model (SAM).<br>
<br>
General Tips (check out our <a href="https://github.com/fsemerar/SlicerTomoSAM">repository</a>):
<ul><li>Generate .pkl using <a href="https://colab.research.google.com/github/fsemerar/SlicerTomoSAM/blob/main/Embeddings/create_embeddings.ipynb">Colab</a> or the Embeddings Create button</li>
<li>Place .tif and .pkl in same folder and make their name equivalent</li>
<li>Drag and drop .tif --> imports both image and embeddings</li>
<li>Once include-point added, the slice is frozen (white background)</li>
<li>Accept Mask button clears points and confirms slice segmentation</li>
<li>Add Segment button adds a new label to segmentation</li>
<li>Create Interpolation button creates masks in between created ones</li>
<li>Undo button reverts interpolation or last mask</li>
<li>Clear button clears the points and the active mask</li>
</ul>
<br>
Note: SAM was trained with up to 9 points, so it is recommended to add up to 9 include+exclude points for optimal predictions<br>
<br>
Keyboard Shortcuts:
<ul><li>'i': switch to include-points</li>
<li>'e': switch to exclude-points</li>
<li>'a': accept mask</li>
<li>'n': new segment</li>
<li>'c': center view</li>
<li>'h': hide/show slice</li>
<li>'r': render 3D view</li>
<li>'z': undo interpolate or last mask</li>
</ul>"""
        self.parent.acknowledgementText = """If you find this work useful for your research or applications please cite <a href="https://arxiv.org/abs/2306.08609">our paper</a>"""


class tomosamWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

        self.slice_visible = True
        self.actual_remove_click = True
        self.mask_accepted = False
        self.slice_frozen = False
        self.first_freeze = False

        self.lm = slicer.app.layoutManager()
        self.segmentEditorWidget = None
        self.layout_id = 20000
        self.orientation = 'horizontal'
        self.view = "Red"
        self.download_location = qt.QStandardPaths.writableLocation(qt.QStandardPaths.DownloadLocation)
        self.sam_weights_path = os.path.join(self.download_location, "sam_vit_h_4b8939.pth")
        self.layouts = {}  # Initialize an empty dictionary for layouts
        self.createLayouts()  # Call the method to create layouts

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath('UI/tomosam.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.logic = tomosamLogic()

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAddedEvent, self.onNodeAdded)

        # volumes
        self.ui.comboVolumeNode.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.segmentSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.segmentSelector.connect("currentSegmentChanged(QString)", self.updateParameterNodeFromGUI)
        self.ui.segmentsTable.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # markups
        self.ui.markupsInclude.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        self.ui.markupsExclude.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        self.ui.markupsInclude.markupsPlaceWidget().setPlaceModePersistency(True)
        self.ui.markupsExclude.markupsPlaceWidget().setPlaceModePersistency(True)

        # filepaths
        self.ui.PathLineEdit_emb.connect('currentPathChanged(QString)', self.pathEmbeddings)

        # push buttons
        self.ui.pushSAMweights.connect("clicked(bool)", self.onPushSAMweights)
        self.ui.pushSAMweights.setToolTip(f"Download SAM weights to {self.sam_weights_path}")
        self.ui.pushMaskAccept.connect("clicked(bool)", self.onPushMaskAccept)
        self.ui.pushMaskClear.connect("clicked(bool)", self.onPushMaskClear)
        self.ui.pushSegmentAdd.connect("clicked(bool)", self.onPushSegmentAdd)
        self.ui.pushSegmentRemove.connect("clicked(bool)", self.onPushSegmentRemove)
        self.ui.pushCenter3d.connect("clicked(bool)", self.onPushCenter3d)
        self.ui.pushVisualizeSlice3d.connect("clicked(bool)", self.onPushVisualizeSlice3d)
        self.ui.pushRender3d.connect("clicked(bool)", self.onPushRender3d)
        self.ui.pushInitializeInterp.connect("clicked(bool)", self.onPushInitializeInterp)
        self.ui.pushUndo.connect("clicked(bool)", self.onPushUndo)
        self.ui.pushEmbeddingsColab.connect("clicked(bool)", self.onPushEmbeddingsColab)
        self.ui.pushEmbeddingsCreate.connect("clicked(bool)", self.onPushEmbeddingsCreate)
        self.ui.radioButton_hor.connect("toggled(bool)", self.onRadioOrient)
        self.ui.radioButton_vert.connect("toggled(bool)", self.onRadioOrient)
        self.ui.radioButton_red.connect("toggled(bool)", self.onRadioView)
        self.ui.radioButton_green.connect("toggled(bool)", self.onRadioView)
        self.ui.radioButton_yellow.connect("toggled(bool)", self.onRadioView)

        shortcuts = [
            ("i", lambda: self.activateIncludePoints()),
            ("e", lambda: self.activateExcludePoints()),
            ("a", lambda: self.onPushMaskAccept()),
            ("n", lambda: self.onPushSegmentAdd()),
            ("c", lambda: self.onPushCenter3d()),
            ("h", lambda: self.onPushVisualizeSlice3d()),
            ("r", lambda: self.onPushRender3d()),
            ("z", lambda: self.onPushUndo()),
        ]
        for (shortcutKey, callback) in shortcuts:
            shortcut = qt.QShortcut(qt.QKeySequence(shortcutKey), slicer.util.mainWindow())
            shortcut.connect("activated()", callback)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # Attach to logic
        self.logic.ui = self.ui

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        self.setParameterNode(self.logic.getParameterNode())

        self._parameterNode.root_path = os.path.dirname(os.path.dirname(os.path.dirname(self.resourcePath(''))))
        if self.logic.sam is None:
            self.logic.create_sam(self.sam_weights_path)
            self.updateLayout()

        if self._parameterNode.GetNodeReferenceID("tomosamInputVolume") is None:
            volume_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if volume_node is not None:
                self._parameterNode.SetNodeReferenceID("tomosamInputVolume", volume_node.GetID())
                self.importPKL(volume_node)

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReferenceID("tomosamIncludePoints"):
            markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "include-points")
            self._parameterNode.SetNodeReferenceID("tomosamIncludePoints", markupsNode.GetID())
            markupsNode.GetDisplayNode().SetSelectedColor(0, 1, 0)
            markupsNode.GetDisplayNode().SetActiveColor(0, 1, 0)
            markupsNode.GetDisplayNode().SetTextScale(0)
            markupsNode.GetDisplayNode().SetGlyphScale(1)
            markupsNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.onMarkupIncludePointPositionDefined)
            markupsNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionUndefinedEvent, self.onMarkupIncludePointPositionUndefined)

        if not self._parameterNode.GetNodeReferenceID("tomosamExcludePoints"):
            markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "exclude-points")
            self._parameterNode.SetNodeReferenceID("tomosamExcludePoints", markupsNode.GetID())
            markupsNode.GetDisplayNode().SetSelectedColor(1, 0, 0)
            markupsNode.GetDisplayNode().SetActiveColor(1, 0, 0)
            markupsNode.GetDisplayNode().SetTextScale(0)
            markupsNode.GetDisplayNode().SetGlyphScale(1)
            markupsNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.onMarkupExcludePointPositionDefined)
            markupsNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionUndefinedEvent, self.onMarkupExcludePointPositionUndefined)

        if not self._parameterNode.GetNodeReferenceID("tomosamSegmentation"):
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', 'Segmentation')
            self.ui.segmentSelector.setCurrentNode(segmentationNode)
            self.ui.segmentsTable.setSegmentationNode(segmentationNode)
            self._parameterNode.SetNodeReferenceID("tomosamSegmentation", segmentationNode.GetID())
            segmentationNode.CreateDefaultDisplayNodes()
            segmentID = segmentationNode.GetSegmentation().AddEmptySegment()
            self._parameterNode.SetParameter("tomosamCurrentSegment", segmentID)

        if self.segmentEditorWidget is None:
            self.segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
            self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
            self.segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
            slicer.mrmlScene.AddNode(self.segmentEditorNode)
            self.segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)

            self.segmentEditorWidget.setSegmentationNode(self._parameterNode.GetNodeReference("tomosamSegmentation"))
            self.segmentEditorWidget.setSourceVolumeNode(self._parameterNode.GetNodeReference("tomosamInputVolume"))
            self.segmentEditorWidget.setActiveEffectByName("Fill between slices")
            self.effect = self.segmentEditorWidget.activeEffect()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.comboVolumeNode.setCurrentNode(self._parameterNode.GetNodeReference("tomosamInputVolume"))

        self.ui.markupsInclude.setCurrentNode(self._parameterNode.GetNodeReference("tomosamIncludePoints"))
        self.ui.markupsExclude.setCurrentNode(self._parameterNode.GetNodeReference("tomosamExcludePoints"))

        if self._parameterNode.GetNodeReferenceID("tomosamSegmentation"):
            self._parameterNode.GetNodeReference("tomosamSegmentation").SetReferenceImageGeometryParameterFromVolumeNode(
                self._parameterNode.GetNodeReference("tomosamInputVolume"))
            self.ui.segmentsTable.setSegmentationNode(self._parameterNode.GetNodeReference("tomosamSegmentation"))

        if self.segmentEditorWidget:
            self.segmentEditorWidget.setSegmentationNode(self._parameterNode.GetNodeReference("tomosamSegmentation"))
            self.segmentEditorWidget.setSourceVolumeNode(self._parameterNode.GetNodeReference("tomosamInputVolume"))

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        if self.ui.comboVolumeNode.currentNodeID != self._parameterNode.GetNodeReferenceID("tomosamInputVolume"):
            self._parameterNode.SetNodeReferenceID("tomosamInputVolume", self.ui.comboVolumeNode.currentNodeID)
            self.importPKL(self._parameterNode.GetNodeReference("tomosamInputVolume"))

        if self._parameterNode.GetNodeReference("tomosamInputVolume") is not None:
            slicer.util.setSliceViewerLayers(background=self._parameterNode.GetNodeReference("tomosamInputVolume"))

        self._parameterNode.SetNodeReferenceID("tomosamIncludePoints", self.ui.markupsInclude.currentNode().GetID())
        self._parameterNode.SetNodeReferenceID("tomosamExcludePoints", self.ui.markupsExclude.currentNode().GetID())

        self._parameterNode.SetNodeReferenceID("tomosamSegmentation", self.ui.segmentSelector.currentNodeID())
        self._parameterNode.GetNodeReference("tomosamSegmentation").SetReferenceImageGeometryParameterFromVolumeNode(
            self._parameterNode.GetNodeReference("tomosamInputVolume"))
        self._parameterNode.SetParameter("tomosamCurrentSegment", self.ui.segmentSelector.currentSegmentID())
        self.ui.segmentsTable.setSegmentationNode(self._parameterNode.GetNodeReference("tomosamSegmentation"))

        if self.segmentEditorWidget:
            self.segmentEditorWidget.setSegmentationNode(self._parameterNode.GetNodeReference("tomosamSegmentation"))
            self.segmentEditorWidget.setSourceVolumeNode(self._parameterNode.GetNodeReference("tomosamInputVolume"))

        self._parameterNode.EndModify(wasModified)

    def findClickedSliceWindow(self):
        sliceNode = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCrosshairNode').GetCursorPositionXYZ([0, 0, 0])
        if sliceNode is None:
            return
        return self.lm.sliceWidget(sliceNode.GetName()).sliceLogic().GetSliceNode().GetLayoutName()

    # to avoid repetitions, calling the same pointAdded/pointRemoved functions
    def onMarkupIncludePointPositionDefined(self, caller, event):

        slice_dir = self.findClickedSliceWindow()
        if slice_dir is None:
            self.actual_remove_click = False
            self.mask_accepted = True
            caller.RemoveNthControlPoint(caller.GetDisplayNode().GetActiveControlPoint())
            return

        if caller.GetNumberOfControlPoints() == 1 and not self.slice_frozen:
            self.logic.slice_direction = slice_dir
            self.onPushVisualizeSlice3d(not_from_button=True)
            self.freezeSlice()
            self.first_freeze = True

        if self.logic.slice_direction != slice_dir:
            self.actual_remove_click = False
            caller.RemoveNthControlPoint(caller.GetDisplayNode().GetActiveControlPoint())
            return

        self.addPoint(caller, self.logic.include_coords)
        self.first_freeze = False

    def onMarkupIncludePointPositionUndefined(self, caller, event):

        if caller.GetNumberOfControlPoints() == 0:
            self.unfreezeSlice()
            self.actual_remove_click = False
            self._parameterNode.GetNodeReference("tomosamExcludePoints").RemoveAllControlPoints()
            self.actual_remove_click = True
            self.logic.exclude_coords = {}

        if self.mask_accepted:
            self.mask_accepted = False
            return

        self.removePoint(caller, self.logic.include_coords, 'include')

    def onMarkupExcludePointPositionDefined(self, caller, event):

        slice_dir = self.findClickedSliceWindow()
        if (self._parameterNode.GetNodeReference("tomosamIncludePoints").GetNumberOfControlPoints() == 0 or
                slice_dir is None or self.logic.slice_direction != slice_dir):
            self.actual_remove_click = False
            caller.RemoveNthControlPoint(caller.GetDisplayNode().GetActiveControlPoint())
            return

        self.addPoint(caller, self.logic.exclude_coords)

    def onMarkupExcludePointPositionUndefined(self, caller, event):
        self.removePoint(caller, self.logic.exclude_coords, 'exclude')

    def addPoint(self, caller, stored_coords):
        point_index = caller.GetDisplayNode().GetActiveControlPoint()
        if not self.checkVolume() or not self.checkSAM() or not self.checkEmbeddings():
            self.actual_remove_click = False
            caller.RemoveNthControlPoint(point_index)
            return
        coords = caller.GetNthControlPointPosition(point_index) * self.logic.voxel_sizes
        caller.SetNthControlPointLocked(point_index, True)
        coords = (int(round(coords[2])), int(round(coords[1])), int(round(coords[0])))

        inside_slice_check = True
        for coord in coords:
            if coord < 0:
                inside_slice_check = False
        if self.logic.slice_direction == 'Red':
            if not (coords[2] < self.logic.img.shape[2] and coords[1] < self.logic.img.shape[1]):
                inside_slice_check = False
        elif self.logic.slice_direction == 'Green':
            if not (coords[2] < self.logic.img.shape[2] and coords[0] < self.logic.img.shape[0]):
                inside_slice_check = False
        else:
            if not (coords[1] < self.logic.img.shape[1] and coords[0] < self.logic.img.shape[0]):
                inside_slice_check = False
        if not inside_slice_check:
            self.actual_remove_click = False
            caller.RemoveNthControlPoint(point_index)
            return

        stored_coords[caller.GetNthControlPointLabel(point_index)] = coords
        self.logic.get_mask(self.first_freeze)

    def removePoint(self, caller, stored_coords, operation):

        # this is to avoid calling remove internally when an illegal point has been added
        # i.e. only call it when user actually removes a point
        if not self.actual_remove_click:
            self.actual_remove_click = True
            return

        new_coords = {}
        for i in range(caller.GetNumberOfControlPoints()):
            point_label = caller.GetNthControlPointLabel(i)
            new_coords[point_label] = stored_coords[point_label]
        if operation == 'include':
            self.logic.include_coords = new_coords
        else:
            self.logic.exclude_coords = new_coords
        self.logic.get_mask(self.first_freeze)

    def freezeSlice(self):
        slice_widget = self.lm.sliceWidget(self.logic.slice_direction)
        interactorStyle = slice_widget.sliceView().sliceViewInteractorStyle()
        interactorStyle.SetActionEnabled(interactorStyle.BrowseSlice, False)
        slice_widget.sliceView().setBackgroundColor(qt.QColor.fromRgbF(1, 1, 1))
        self.slice_frozen = True
        slice_widget.sliceController().setDisabled(self.slice_frozen)  # freeze slidebar

    def unfreezeSlice(self):
        slice_widget = self.lm.sliceWidget(self.logic.slice_direction)
        interactorStyle = slice_widget.sliceView().sliceViewInteractorStyle()
        interactorStyle.SetActionEnabled(interactorStyle.BrowseSlice, True)
        slice_widget.sliceView().setBackgroundColor(qt.QColor.fromRgbF(0, 0, 0))
        self.slice_frozen = False
        slice_widget.sliceController().setDisabled(self.slice_frozen)  # freeze slidebar

    def clearPoints(self):
        self.actual_remove_click = False
        self._parameterNode.GetNodeReference("tomosamIncludePoints").RemoveAllControlPoints()
        self.logic.include_coords = {}
        self.actual_remove_click = False
        self._parameterNode.GetNodeReference("tomosamExcludePoints").RemoveAllControlPoints()
        self.logic.exclude_coords = {}

    def onPushMaskAccept(self):
        self.mask_accepted = True
        self.actual_remove_click = False
        self._parameterNode.GetNodeReference("tomosamIncludePoints").RemoveAllControlPoints()
        self.logic.include_coords = {}

        # only add mask_location for interpolation if mask has been created
        if ((self.logic.slice_direction == 'Red' and self.logic.mask[self.logic.ind].max() != 0) or
                (self.logic.slice_direction == 'Green' and self.logic.mask[:, self.logic.ind].max() != 0) or
                (self.logic.slice_direction == 'Yellow' and self.logic.mask[:, :, self.logic.ind].max() != 0)):
            self.logic.mask_locations.add(self.logic.ind)
            self.logic.interp_slice_direction.add(self.logic.slice_direction)

    def onPushMaskClear(self):
        if self.slice_frozen:
            self.logic.undo()
            self.onPushMaskAccept()

    def onPushUndo(self):
        self.logic.undo()

    def onPushSegmentAdd(self):
        self.clearPoints()
        self.logic.mask_locations = set()
        self.logic.interp_slice_direction = set()
        segmentID = self._parameterNode.GetNodeReference("tomosamSegmentation").GetSegmentation().AddEmptySegment()
        self._parameterNode.SetParameter("tomosamCurrentSegment", segmentID)
        self.ui.segmentSelector.setCurrentSegmentID(segmentID)
        self.logic.mask.fill(0)

    def onPushSegmentRemove(self):
        if len(self._parameterNode.GetNodeReference("tomosamSegmentation").GetSegmentation().GetSegmentIDs()) <= 1:
            slicer.util.errorDisplay("Need to have at least one segment")
            return
        self._parameterNode.GetNodeReference("tomosamSegmentation").RemoveSegment(self._parameterNode.GetParameter("tomosamCurrentSegment"))
        self.ui.segmentSelector.setCurrentSegmentID(self._parameterNode.GetNodeReference("tomosamSegmentation").GetSegmentation().GetSegmentIDs()[-1])

    def onPushSAMweights(self):
        if self.checkSAMdownload():
            slicer.util.delayDisplay(f"Downloading SAM weights to {self.sam_weights_path} (2.5GB, it may take several minutes)...")
            print("Downloading SAM weights ... ", end='')
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            urllib.request.urlretrieve(url, self.sam_weights_path)
            print("Done")
        else:
            slicer.util.infoDisplay(f"SAM weights already found at {self.sam_weights_path}")

    def checkSAMdownload(self):
        return not os.path.exists(self.sam_weights_path) or not os.path.isfile(self.sam_weights_path) or \
            hashlib.md5(open(self.sam_weights_path, 'rb').read()).hexdigest() != "4b8939a88964f0f4ff5f5b2642c598a6"

    def checkSAM(self):
        if self.logic.sam is None:
            if self.checkSAMdownload():
                slicer.util.errorDisplay("SAM weights not found, use Download button")
                return False
            self.logic.create_sam(self.sam_weights_path)
        return True

    def checkEmbeddings(self):
        if len(self.logic.embeddings) == 0:
            slicer.util.errorDisplay("Select image Embeddings")
            return False
        return True

    def checkVolume(self):
        if not self._parameterNode.GetNodeReferenceID("tomosamInputVolume"):
            slicer.util.errorDisplay("Select a volume")
            return False
        else:
            return True

    def pathEmbeddings(self):
        embeddings_path = self.ui.PathLineEdit_emb.currentPath

        if not os.path.exists(embeddings_path) or not os.path.isfile(embeddings_path):
            slicer.util.errorDisplay("Image Embeddings file not found")
            return
        elif os.path.splitext(embeddings_path)[1] != ".pkl":
            slicer.util.errorDisplay("Unrecognized extension for image Embeddings")
            return

        slicer.util.delayDisplay(f"Reading embeddings: {embeddings_path}")
        self.logic.read_img_embeddings(embeddings_path)

    def onPushEmbeddingsCreate(self):
        if not self.checkVolume() or not self.checkSAM():
            return

        volume_node = self._parameterNode.GetNodeReference("tomosamInputVolume")
        storageNode = volume_node.GetStorageNode()
        if storageNode is not None:  # loaded via drag-drop
            filepath = storageNode.GetFullNameFromFileName()
        else:  # Loaded via DICOM browser
            instanceUIDs = volume_node.GetAttribute("DICOM.instanceUIDs").split()
            filepath = slicer.dicomDatabase.fileForInstance(instanceUIDs[0])

        output_filepath = os.path.join(self.download_location, os.path.splitext(filepath)[0] + ".pkl")
        if os.path.isfile(output_filepath):
            slicer.util.infoDisplay(f"Embeddings file already found at {output_filepath}")
            return

        if slicer.util.confirmOkCancelDisplay("It is recommended to create embeddings on a machine with GPU (e.g. Colab) to reduce runtime.\n"
                                              "Creating them locally may take several minutes and there is not way to stop the process other than force quitting Slicer.\n"
                                              "Click OK to continue"):
            self.embeddings_already_loaded = True
            self.ui.PathLineEdit_emb.currentPath = self.logic.create_embeddings(output_filepath)

    def onPushEmbeddingsColab(self):
        qt.QDesktopServices.openUrl(qt.QUrl("https://colab.research.google.com/github/fsemerar/SlicerTomoSAM/blob/main/Embeddings/create_embeddings.ipynb"))

    def onPushCenter3d(self):
        if not self.slice_frozen:
            threeDWidget = self.lm.threeDWidget(0)
            threeDView = threeDWidget.threeDView()
            threeDView.resetFocalPoint()
            slicer.util.resetSliceViews()

    def onPushVisualizeSlice3d(self, not_from_button=False):
        if not not_from_button:
            self.slice_visible = not self.slice_visible
        # first deactivate them all
        for sliceViewName in self.lm.sliceViewNames():
            self.lm.sliceWidget(sliceViewName).sliceController().setSliceVisible(False)
        # then activate only the one we need
        self.lm.sliceWidget(self.logic.slice_direction).sliceController().setSliceVisible(self.slice_visible)

    def onPushRender3d(self):
        self._parameterNode.GetNodeReference("tomosamSegmentation").CreateClosedSurfaceRepresentation()

    def onPushInitializeInterp(self):
        if len(self.logic.interp_slice_direction) > 1:
            slicer.util.errorDisplay("Cannot interpolate if multiple slice directions have been segmented")
            return
        elif len(self.logic.interp_slice_direction) == 0 or len(self.logic.mask_locations) == 0:
            slicer.util.errorDisplay("Cannot interpolate if no masks have been added")
            return

        self.logic.backup_mask()

        # when segments not visible, interpolation won't happen --> hide everything except current segment
        for segmentID in self._parameterNode.GetNodeReference("tomosamSegmentation").GetSegmentation().GetSegmentIDs():
            if segmentID != self._parameterNode.GetParameter("tomosamCurrentSegment"):
                self._parameterNode.GetNodeReference("tomosamSegmentation").GetDisplayNode().SetSegmentVisibility(segmentID, False)

        self.effect.self().onPreview()
        self.effect.self().onApply()

        for segmentID in self._parameterNode.GetNodeReference("tomosamSegmentation").GetSegmentation().GetSegmentIDs():
            if segmentID != self._parameterNode.GetParameter("tomosamCurrentSegment"):
                self._parameterNode.GetNodeReference("tomosamSegmentation").GetDisplayNode().SetSegmentVisibility(segmentID, True)

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def onNodeAdded(self, caller, event, calldata):
        node = calldata
        if type(node) == slicer.vtkMRMLScalarVolumeNode:
            # call using a timer instead of calling it directly to allow the volume loading to fully complete
            print("Reading image ... ", end='')
            qt.QTimer.singleShot(30, lambda: self.autoSelectVolume(node))
            print("Done")
            qt.QTimer.singleShot(30, lambda: self.importPKL(node))

    def autoSelectVolume(self, volumeNode):
        self.ui.comboVolumeNode.setCurrentNodeID(volumeNode.GetID())

    def importPKL(self, volumeNode):
        if volumeNode is not None:
            storage_node = volumeNode.GetStorageNode()
            if storage_node is not None:
                pkl_filepath = os.path.splitext(storage_node.GetFileName())[0] + ".pkl"
                if os.path.exists(pkl_filepath):  # try to import Embeddings too if same name and folder as image
                    self.ui.PathLineEdit_emb.setCurrentPath(pkl_filepath)
                self.onPushCenter3d()  # also run these functions on import
                self.onPushVisualizeSlice3d(not_from_button=True)

    def activateIncludePoints(self):
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetActivePlaceNodeID(self._parameterNode.GetNodeReferenceID("tomosamIncludePoints"))
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

    def activateExcludePoints(self):
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetActivePlaceNodeID(self._parameterNode.GetNodeReferenceID("tomosamExcludePoints"))
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

    def onRadioOrient(self):
        if self.ui.radioButton_hor.isChecked():
            self.orientation = 'horizontal'
        else:
            self.orientation = 'vertical'
        self.updateLayout()

    def onRadioView(self):
        if self.ui.radioButton_red.isChecked():
            self.view = 'Red'
        elif self.ui.radioButton_green.isChecked():
            self.view = 'Green'
        else:
            self.view = 'Yellow'
        self.updateLayout()

    def createLayouts(self):
        orientations = ['horizontal', 'vertical']
        views = ['Red', 'Green', 'Yellow']

        for orientation in orientations:
            for view in views:
                self.layout_id += 1
                customLayout = f"""
                    <layout type="{orientation}" split="true">
                        <item>
                            <view class="vtkMRMLViewNode" singletontag="1">
                                <property name="viewlabel" action="default">1</property>
                            </view>
                        </item>
                        <item>
                            <view class="vtkMRMLSliceNode" singletontag="{view}">
                            </view>
                        </item>
                    </layout>
                """
                self.layouts[(orientation, view)] = self.layout_id
                self.lm.layoutLogic().GetLayoutNode().AddLayoutDescription(self.layout_id, customLayout)
                self.layout_id += 1

    def updateLayout(self):
        self.layout_id = self.layouts[(self.orientation, self.view)]

        # If the layout ID doesn't exist, use a default layout
        if self.layout_id is None:
            defaultLayout = self.layouts[('horizontal', 'Red')]
            if defaultLayout is None:
                return
            self.layout_id = defaultLayout

        self.lm.setLayout(self.layout_id)
        self.logic.slice_direction = self.view
        self.onPushVisualizeSlice3d(not_from_button=True)

    # methods below don't need to be changed
    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()
