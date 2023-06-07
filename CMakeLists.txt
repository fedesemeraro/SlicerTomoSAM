cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(TomoSAM)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/fsemerar/TomoSAM")
set(EXTENSION_CATEGORY "Segmentations")
set(EXTENSION_CONTRIBUTORS "Federico Semeraro (NASA); Alexandre Quintart (NASA); Sergio Fraile Izquierdo (NASA); Joseph Ferguson (Stanford)")
set(EXTENSION_DESCRIPTION "TomoSAM")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")
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
