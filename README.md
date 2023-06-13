<p align="center">
  <img src="https://github.com/fsemerar/SlicerTomoSAM/raw/main/TomoSAM/Resources/Media/tomosam_logo.png" width="35%"></img>
</p>

An extension of [3D Slicer](https://www.slicer.org/) using the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) 
to aid the segmentation of 3D data from tomography or other imaging techniques.

<p align="center">
  <img src="https://github.com/fsemerar/SlicerTomoSAM/raw/main/TomoSAM/Resources/Media/tomosam_screenshot_1.png" width="100%"></img>
</p>

<p align="center">
  <img src="https://github.com/fsemerar/SlicerTomoSAM/raw/main/TomoSAM/Resources/Media/tomosam_diagram.png" width="100%"></img>
</p>

## How to Get Started

You can follow [this tutorial video]() 
for a quick guide on how to get started. The example data used in the video can be downloaded 
[from here](https://drive.google.com/drive/folders/1YeerY3b1S-qIsVOleminwZeCM0Uw6TJr?usp=sharing).

### Installation:

Follow these steps to install Slicer and the TomoSAM extension:

- Download Slicer from [this link](https://download.slicer.org/) and install it
- In Slicer, open `Extension Manager` and search for TomoSAM: the extension will appear under `Segmentations`&rarr;`TomoSAM`

### Prepare the embeddings

This preprocessing step creates the embeddings for all the slices of your tiff stack along the three Cartesian directions.
You can create the embeddings by running [this notebook](Embeddings/create_embeddings.ipynb) either locally or 
[on Colab](https://colab.research.google.com/github/fsemerar/SlicerTomoSAM/blob/main/Embeddings/create_embeddings.ipynb).
A GPU is recommended for this step to speed up the process; in Colab, make sure to select 
`Runtime`&rarr;`Change runtime type` and set the `Hardware accelerator` to GPU. Locally, you will first need to create 
the environment by running: 

    conda env create --file env/environment_cpu.yml  # for CPU
    conda env create --file env/environment_gpu.yml  # for GPU
    conda activate tomosam
    jupyter notebook create_embeddings.ipynb

### How to use the TomoSAM Slicer extension

These are the usual steps to produce a segmentation using TomoSAM: 

- Place the .tif image and .pkl embeddings in the same folder and make their name equivalent, e.g. test.tif and test.pkl
- Open Slicer and navigate into `Segmentations`&rarr;`TomoSAM`
- Drag and drop the image into Slicer, which will automatically import the embeddings as well
- Add include-points in one of the three slice viewers (Red/Green/Yellow) to create a mask and exclude-points to refine it
- Once one point is added, the selected slice is frozen until no points exist or the `Accept Mask` button is pressed
- Add as many segments as you have objects by clicking on `Add Segment`

Note that you can further modify the masks created in TomoSAM using the widgets in the `Segment Editor`, e.g. Paint or Erase

## Citation

If you find this work useful for your research or applications, please cite using this BibTeX:

```BibTeX
@article{semeraro2023tomosam,
  title={TomoSAM: a 3D Slicer extension using SAM for tomography segmentation},
  author={Semeraro, Federico and Quintart, Alexandre and Fraile Izquierdo, Sergio and Ferguson, Joseph C.},
  journal={arXiv preprint arXiv:},
  year={2023}
}
```

## License

This work has been implemented as an integral part of the [Porous Microstructure Analysis (PuMA)](https://github.com/nasa/puma) 
software to assist it in the necessary preprocessing tomography segmentation. 
It is therefore released under the same open-source license:

Copyright @ 2017, 2020, 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.
This software may be used, reproduced, and provided to others only as permitted under the terms of the agreement under which it was acquired from the U.S. Government. Neither title to, nor ownership of, the software is hereby transferred. This notice shall remain on all copies of the software.
This file is available under the terms of the NASA Open Source Agreement (NOSA), and further subject to the additional disclaimer below:
Disclaimer:
THE SOFTWARE AND/OR TECHNICAL DATA ARE PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE AND/OR TECHNICAL DATA WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SOFTWARE AND/OR TECHNICAL DATA WILL BE ERROR FREE, OR ANY WARRANTY THAT TECHNICAL DATA, IF PROVIDED, WILL CONFORM TO THE SOFTWARE. IN NO EVENT SHALL THE UNITED STATES GOVERNMENT, OR ITS CONTRACTORS OR SUBCONTRACTORS, BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE AND/OR TECHNICAL DATA, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE AND/OR TECHNICAL DATA.
THE UNITED STATES GOVERNMENT DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD PARTY COMPUTER SOFTWARE, DATA, OR DOCUMENTATION, IF SAID THIRD PARTY COMPUTER SOFTWARE, DATA, OR DOCUMENTATION IS PRESENT IN THE NASA SOFTWARE AND/OR TECHNICAL DATA, AND DISTRIBUTES IT "AS IS."
RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT AND ITS CONTRACTORS AND SUBCONTRACTORS, AND SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT AND ITS CONTRACTORS AND SUBCONTRACTORS FOR ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES THAT MAY ARISE FROM RECIPIENT'S USE OF THE SOFTWARE AND/OR TECHNICAL DATA, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, THE USE THEREOF.
IF RECIPIENT FURTHER RELEASES OR DISTRIBUTES THE NASA SOFTWARE AND/OR TECHNICAL DATA, RECIPIENT AGREES TO OBTAIN THIS IDENTICAL WAIVER OF CLAIMS, INDEMNIFICATION AND HOLD HARMLESS, AGREEMENT WITH ANY ENTITIES THAT ARE PROVIDED WITH THE SOFTWARE AND/OR TECHNICAL DATA.
