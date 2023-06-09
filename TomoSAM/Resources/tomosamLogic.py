from slicer.ScriptedLoadableModule import *
import slicer
import numpy as np
import pickle

try:
    import torch
except ImportError:
    slicer.util.pip_install("torch torchvision")
    import torch

try:
    import cv2
except ImportError:
    slicer.util.pip_install("opencv-python")
    import cv2

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    slicer.util.pip_install("segment-anything")
    from segment_anything import sam_model_registry, SamPredictor


class tomosamLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self._parameterNode = self.getParameterNode()

        self.sam, self.predictor, self.device = None, None, None
        self.img, self.mask, self.embeddings = np.zeros((1,1,1)), np.zeros((1,1,1)), []
        self.min_mask_region_area = 500
        self.ind = 0

        self.include_coords = {}
        self.exclude_coords = {}

        self.emb_slice_d = {'Yellow': 2, 'Green': 1, 'Red': 0}
        self.slice_direction = 'Red'

        self.mask_locations = set()
        self.interp_slice_direction = set()
        self.mask_undo = None

    def create_sam(self, sam_checkpoint_filepath):
        print("Creating SAM predictor ... ", end="")
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_filepath)
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.sam.to(device="cuda")
        else:
            self.device = "cpu"
        self.predictor = SamPredictor(self.sam)
        self.predictor.is_image_set = True
        print("Done")

    def read_img_embeddings(self, embeddings_filepath):
        self.img = slicer.util.arrayFromVolume(self._parameterNode.GetNodeReference("tomosamInputVolume"))

        with open(embeddings_filepath, 'rb') as f:
            self.embeddings = pickle.load(f)

        # checking image vs Embeddings dimensions
        if (np.any(np.array(self.img.shape)[[1, 2]] != np.array(self.embeddings[0][0]['original_size'])) or
                np.any(np.array(self.img.shape)[[0, 2]] != np.array(self.embeddings[1][0]['original_size'])) or
                np.any(np.array(self.img.shape)[[0, 1]] != np.array(self.embeddings[2][0]['original_size']))):
            self.embeddings = []
            return False
        else:
            return True

    def select_embedding(self):
        self.predictor.original_size = self.embeddings[self.emb_slice_d[self.slice_direction]][self.ind]['original_size']
        self.predictor.input_size = self.embeddings[self.emb_slice_d[self.slice_direction]][self.ind]['input_size']
        self.predictor.features = self.embeddings[self.emb_slice_d[self.slice_direction]][self.ind]['features'].to(self.device)

    def get_mask_from_slicer(self):
        self.mask = slicer.util.arrayFromSegmentBinaryLabelmap(self._parameterNode.GetNodeReference("tomosamSegmentation"),
                                                               self._parameterNode.GetParameter("tomosamCurrentSegment"))

    def pass_mask_to_slicer(self):
        slicer.util.updateSegmentBinaryLabelmapFromArray(self.mask,
                                                         self._parameterNode.GetNodeReference("tomosamSegmentation"),
                                                         self._parameterNode.GetParameter("tomosamCurrentSegment"),
                                                         self._parameterNode.GetNodeReference("tomosamInputVolume"))

    def fill_mask(self, value):
        if self.slice_direction == 'Red':
            self.mask[self.ind] = value
        elif self.slice_direction == 'Green':
            self.mask[:, self.ind] = value
        else:
            self.mask[:, :, self.ind] = value

    def get_mask(self, first_freeze):

        if first_freeze:
            self.get_mask_from_slicer()

        if len(self.include_coords) != 0:
            if self.slice_direction == 'Red':
                include_points = [[coords[2], coords[1]] for coords in self.include_coords.values()]
                exclude_points = [[coords[2], coords[1]] for coords in self.exclude_coords.values()]
                self.ind = list(self.include_coords.values())[0][0]
            elif self.slice_direction == 'Green':
                include_points = [[coords[2], coords[0]] for coords in self.include_coords.values()]
                exclude_points = [[coords[2], coords[0]] for coords in self.exclude_coords.values()]
                self.ind = list(self.include_coords.values())[0][1]
            else:  # Y
                include_points = [[coords[1], coords[0]] for coords in self.include_coords.values()]
                exclude_points = [[coords[1], coords[0]] for coords in self.exclude_coords.values()]
                self.ind = list(self.include_coords.values())[0][2]

            self.select_embedding()
            mask, _, _ = self.predictor.predict(point_coords=np.array(include_points + exclude_points),
                                                point_labels=np.array([1] * len(include_points) + [0] * len(exclude_points)),
                                                multimask_output=False)

            mask = mask[0].astype(np.uint8)
            mask = self.remove_small_regions(mask, self.min_mask_region_area, "holes")
            mask = self.remove_small_regions(mask, self.min_mask_region_area, "islands")
            self.fill_mask(mask)
        else:
            self.fill_mask(0)
        self.pass_mask_to_slicer()

    @staticmethod
    def remove_small_regions(mask, area_thresh, mode):
        """Function from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py"""
        assert mode in ["holes", "islands"]
        correct_holes = mode == "holes"
        working_mask = (correct_holes ^ mask).astype(np.uint8)
        n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
        sizes = stats[:, -1][1:]  # Row 0 is background label
        small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
        if len(small_regions) == 0:
            return mask
        fill_labels = [0] + small_regions
        if not correct_holes:
            fill_labels = [i for i in range(n_labels) if i not in fill_labels]
            # If every region is below threshold, keep largest
            if len(fill_labels) == 0:
                fill_labels = [int(np.argmax(sizes)) + 1]
        mask = np.isin(regions, fill_labels)
        return mask

    def backup_mask(self):
        self.get_mask_from_slicer()
        self.mask_undo = self.mask.copy()

    def undo_interpolate(self):
        if self.mask_undo is not None:
            self.mask = self.mask_undo.copy()
            self.pass_mask_to_slicer()
            self.mask_undo = None
