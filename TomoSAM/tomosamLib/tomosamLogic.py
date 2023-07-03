from slicer.ScriptedLoadableModule import *
import slicer
import numpy as np
import pickle
import vtk
import SimpleITK as sitk
import sys


class tomosamLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self._parameterNode = self.getParameterNode()

        self.sam, self.predictor, self.device = None, None, None
        self.img, self.voxel_sizes, self.mask, self.embeddings = np.zeros((1,1,1)), np.zeros(3), np.zeros((1,1,1)), []
        self.min_mask_region_area = 500
        self.ind = 0

        self.torch = None

        self.include_coords = {}
        self.exclude_coords = {}

        self.emb_slice_d = {'Yellow': 2, 'Green': 1, 'Red': 0}
        self.slice_direction = 'Red'

        self.mask_locations = set()
        self.interp_slice_direction = set()
        self.mask_backup = None

    def setupPythonRequirements(self):

        # Install PyTorch
        try:
            import PyTorchUtils
        except ModuleNotFoundError as e:
            slicer.util.errorDisplay("This module requires PyTorch extension. Install it from the Extensions Manager.")
            return False

        minimumTorchVersion = "1.7"
        minimumTorchVisionVersion = "0.8"
        torchLogic = PyTorchUtils.PyTorchUtilsLogic()
        if not torchLogic.torchInstalled():
            slicer.util.delayDisplay("PyTorch Python package is required. Installing... (it may take several minutes)")
            torch = torchLogic.installTorch(askConfirmation=True, torchVersionRequirement=f">={minimumTorchVersion}",
                                            torchvisionVersionRequirement=f">={minimumTorchVisionVersion}")
            if torch is None:
                raise ValueError('PyTorch extension needs to be installed to use this module.')
        else:
            # torch is installed, check version
            from packaging import version
            if version.parse(torchLogic.torch.__version__) < version.parse(minimumTorchVersion):
                raise ValueError(f'PyTorch version {torchLogic.torch.__version__} is not compatible with this module.'
                                 + f' Minimum required version is {minimumTorchVersion}. You can use "PyTorch Util" module to install PyTorch'
                                 + f' with version requirement set to: >={minimumTorchVersion}')
        self.torch = torchLogic.importTorch()

        # Install SAM
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ModuleNotFoundError:
            slicer.util.pip_install("https://github.com/facebookresearch/segment-anything/archive/6fdee8f2727f4506cfbbe553e23b895e27956588.zip")
            from segment_anything import sam_model_registry, SamPredictor
        return True

    def create_sam(self, sam_weights_path):
        slicer.util.delayDisplay("Loading TomoSAM ... ")

        if not self.setupPythonRequirements():
            return

        from segment_anything import sam_model_registry, SamPredictor
        print("Creating SAM predictor ... ", end="")
        try:
            self.sam = sam_model_registry["vit_h"](checkpoint=sam_weights_path)
        except FileNotFoundError:
            slicer.util.infoDisplay("SAM weights not found, use Download button")
            print("weights not found")
            return

        if self.torch.cuda.is_available():
            self.device = "cuda:0"
            self.sam.to(device="cuda")
        else:
            self.device = "cpu"

        self.predictor = SamPredictor(self.sam)
        self.predictor.is_image_set = True
        print("Done")

    def create_embeddings(self, output_filepath):
        self.img = slicer.util.arrayFromVolume(self._parameterNode.GetNodeReference("tomosamInputVolume"))

        if self.img.ndim > 3 or self.img.ndim < 2:
            raise Exception("Unsupported image type.")
        elif self.img.ndim == 2:
            self.img = self.img[:, :, np.newaxis]

        embeddings = [[], [], []]
        slice_direction = ['x', 'y', 'z']
        for i, d in enumerate(slice_direction):
            print(f"\nSlicing along {d} direction")
            for k in range(self.img.shape[i]):
                if i == 0:
                    img_slice = self.img[k]
                elif i == 1:
                    img_slice = self.img[:, k]
                else:
                    img_slice = self.img[:, :, k]

                slicer.util.delayDisplay(f"Creating embeddings for {output_filepath} with dims: {self.img.shape} \n"
                                         f"Slicing along {d} direction, {k + 1}/{self.img.shape[i]} image")
                sys.stdout.write(f"\rCreating embedding for {k + 1}/{self.img.shape[i]} image")

                self.predictor.set_image(np.repeat(img_slice[:, :, np.newaxis], 3, axis=2))
                embeddings[i].append({'original_size': self.predictor.original_size,
                                      'input_size': self.predictor.input_size,
                                      'features': self.predictor.features.to('cpu')})

                self.predictor.reset_image()
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()

        with open(output_filepath, 'wb') as f:
            pickle.dump(embeddings, f)
            print(f"\nSaved {output_filepath}")
        return output_filepath

    def read_img_embeddings(self, embeddings_filepath):
        self.img = slicer.util.arrayFromVolume(self._parameterNode.GetNodeReference("tomosamInputVolume"))
        ras2ijk = vtk.vtkMatrix4x4()
        self._parameterNode.GetNodeReference("tomosamInputVolume").GetRASToIJKMatrix(ras2ijk)
        self.voxel_sizes[:] = slicer.util.arrayFromVTKMatrix(ras2ijk).diagonal()[:3]

        print("Reading embeddings ... ", end="")
        with open(embeddings_filepath, 'rb') as f:
            self.embeddings = pickle.load(f)
        print("Done")

        # checking image vs embeddings dimensions
        if (np.any(np.array(self.img.shape)[[1, 2]] != np.array(self.embeddings[0][0]['original_size'])) or
                np.any(np.array(self.img.shape)[[0, 2]] != np.array(self.embeddings[1][0]['original_size'])) or
                np.any(np.array(self.img.shape)[[0, 1]] != np.array(self.embeddings[2][0]['original_size']))):
            slicer.util.errorDisplay(f"Embeddings dimensions {(len(self.embeddings[0]), len(self.embeddings[1]), len(self.embeddings[2]))} "
                                     f"don't match image {self.img.shape}")
            self.embeddings = []

    def pass_mask_to_slicer(self):
        slicer.util.updateSegmentBinaryLabelmapFromArray(self.mask,
                                                         self._parameterNode.GetNodeReference("tomosamSegmentation"),
                                                         self._parameterNode.GetParameter("tomosamCurrentSegment"),
                                                         self._parameterNode.GetNodeReference("tomosamInputVolume"))

    def get_mask(self, first_freeze):

        if first_freeze:
            self.backup_mask()

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

            # select embeddings
            self.predictor.original_size = self.embeddings[self.emb_slice_d[self.slice_direction]][self.ind]['original_size']
            self.predictor.input_size = self.embeddings[self.emb_slice_d[self.slice_direction]][self.ind]['input_size']
            self.predictor.features = self.embeddings[self.emb_slice_d[self.slice_direction]][self.ind]['features'].to(self.device)

            new_mask, _, _ = self.predictor.predict(point_coords=np.array(include_points + exclude_points),
                                                    point_labels=np.array([1] * len(include_points) + [0] * len(exclude_points)),
                                                    multimask_output=False)

            new_mask = new_mask[0].astype(np.uint8)
            new_mask = self.remove_small_regions(new_mask, self.min_mask_region_area, "holes")
            new_mask = self.remove_small_regions(new_mask, self.min_mask_region_area, "islands")

            if self.slice_direction == 'Red':
                self.mask[self.ind] = np.logical_or(self.mask_backup[self.ind], new_mask)
            elif self.slice_direction == 'Green':
                self.mask[:, self.ind] = np.logical_or(self.mask_backup[:, self.ind], new_mask)
            else:
                self.mask[:, :, self.ind] = np.logical_or(self.mask_backup[:, :, self.ind], new_mask)

            self.pass_mask_to_slicer()
        else:
            self.undo()

    def backup_mask(self):
        self.mask = slicer.util.arrayFromSegmentBinaryLabelmap(self._parameterNode.GetNodeReference("tomosamSegmentation"),
                                                               self._parameterNode.GetParameter("tomosamCurrentSegment"))
        self.mask_backup = self.mask.copy()

    def undo(self):
        if self.mask_backup is not None:
            self.mask = self.mask_backup.copy()
            self.pass_mask_to_slicer()

    @staticmethod
    def remove_small_regions(mask, area_thresh, mode):
        """Function from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py"""
        assert mode in ["holes", "islands"]
        correct_holes = mode == "holes"
        working_mask = (correct_holes ^ mask).astype(np.uint8)
        sitk_image = sitk.GetImageFromArray(working_mask)
        connected_components = sitk.ConnectedComponent(sitk_image, True)
        regions = sitk.RelabelComponent(connected_components)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(regions)
        regions = sitk.GetArrayFromImage(regions)
        n_labels = stats.GetNumberOfLabels() + 1
        sizes = np.array([stats.GetPhysicalSize(label) for label in stats.GetLabels()])
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
