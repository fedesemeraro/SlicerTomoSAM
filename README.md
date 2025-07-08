<p align="center">
  <img src="https://github.com/fsemerar/SlicerTomoSAM/raw/main/TomoSAM/Resources/Media/tomosam_logo.png" width="35%"></img>
</p>

An extension of [3D Slicer](https://www.slicer.org/) using the 
[Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) 
to aid the segmentation of 3D data from tomography or other imaging techniques.

<p align="center">
  <img src="https://github.com/fsemerar/SlicerTomoSAM/raw/main/TomoSAM/Resources/Media/tomosam_screenshot_1.png" width="100%"></img>
</p>

## How to Get Started

You can follow [this tutorial video](https://youtu.be/4nXCYrvBSjk) 
for a quick guide on how to get started. Some example 3D medical images and their embeddings can be downloaded 
[from here](https://nasa-ext.box.com/s/gyjlrhv63pdj2k9yip6g5reb2r9jlc6l).


### Installation:

- Download [Slicer](https://download.slicer.org/) and install it
- In Slicer, open the `Extension Manager`, install [TomoSAM](https://github.com/fsemerar/SlicerTomoSAM), and restart it
- TomoSAM will appear in the Modules dropdown list, under `Segmentation`&rarr;`TomoSAM`


### Prepare the embeddings

A necessary preprocessing step to use TomoSAM is the creation of the embeddings for all the slices of your image stack 
along the three Cartesian directions. A GPU is recommended for this step to speed up the process. 
This can be done in three different ways, depending on your needs: 

1. Using an online GPU
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fsemerar/SlicerTomoSAM/blob/main/Embeddings/create_embeddings.ipynb)

2. Directly in TomoSAM by clicking the `Create` button in the Embeddings section

3. Locally using the same [notebook](Embeddings/create_embeddings.ipynb) and first installing the environment by running: 

        conda env create --file Embeddings/env/environment_cpu.yml  # for CPU
        conda env create --file Embeddings/env/environment_gpu.yml  # for GPU
        conda activate tomosam
        jupyter notebook create_embeddings.ipynb


### How to use the TomoSAM Slicer extension

Refer to the `Help & Acknoledgement` text for a list of **keyboard shortcuts** and general tips.
These are the usual steps to produce a segmentation using TomoSAM:

- Place the image and embeddings in the same folder and make their name equivalent, e.g. test.tif and test.pkl
- Open Slicer and, from the drop-down `Modules` menu, select `Segmentation`&rarr;`TomoSAM`
- Drag and drop the image into Slicer, which will automatically import the embeddings as well
- Add include-points in the slice viewer (Red/Green/Yellow) to create a mask and exclude-points to refine it
- When one point is added, slice scroll is frozen until no points exist or the `Accept Mask` button is pressed
- Add as many segments as you have objects by clicking on `Add Segment`
- Interpolate between the created masks using the `Create Interpolation` button
- The 3D view does not refresh automatically to reduce latency: update it using the `Render 3D` button

Note that you can further modify the masks created in TomoSAM using the widgets in the `Segment Editor`, e.g. Paint or Erase.


## Citation

If you find this work useful for your research or applications, please cite [our paper](https://arxiv.org/abs/2306.08609) using:

```BibTeX
@article{SEMERARO2025102218,
title = {TomoSAM: A 3D Slicer extension using SAM for tomography segmentation},
journal = {SoftwareX},
volume = {31},
pages = {102218},
year = {2025},
issn = {2352-7110},
doi = {https://doi.org/10.1016/j.softx.2025.102218},
url = {https://www.sciencedirect.com/science/article/pii/S2352711025001852},
author = {Federico Semeraro and Alexandre M. Quintart and Sergio Fraile Izquierdo and Joseph C. Ferguson},
keywords = {3D segmentation, SAM, Slicer, Tomography}
}
```

From the paper, the software architecture is summarized in the following diagram and our workflow to analyze 
Thermal Protection Materials (TPS) is also shown below. 

<p align="center">
  <img src="https://github.com/fsemerar/SlicerTomoSAM/raw/main/TomoSAM/Resources/Media/tomosam_diagram.png" width="100%"></img>
</p>
<p align="center">
  <img src="https://github.com/fsemerar/SlicerTomoSAM/raw/main/TomoSAM/Resources/Media/tomosam_workflow.png" width="100%"></img>
</p>


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
