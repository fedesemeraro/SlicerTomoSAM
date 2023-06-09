cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(TomoSAM)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/fsemerar/SlicerTomoSAM")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Federico Semeraro (NASA); Alexandre Quintart (NASA); Sergio Fraile Izquierdo (NASA); Joseph Ferguson (Stanford University)")
set(EXTENSION_DESCRIPTION "TomoSAM")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/fsemerar/SlicerTomoSAM/main/TomoSAM/Resources/Icons/tomosam.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/fsemerar/SlicerTomoSAM/main/TomoSAM/Resources/Screenshots/tomosam_screenshot_1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(TomoSAM)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
