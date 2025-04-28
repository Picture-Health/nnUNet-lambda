# ---------------------------------------------------------------
# Orchestration Function for Lesion Splitting
# ---------------------------------------------------------------

import os
import SimpleITK as sitk
import numpy as np
import sys
import scipy


# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------
def check_in_convex_hull(mask_for_convex: np.array, mask_to_check: np.array) -> bool:
    """check if mask_to_check is within the convex hull of mask_for_convex.

    Args:
        mask_for_convex (np.array): lung mask requested from raven
        mask_to_check (np.array):mask of splitted lesion

    Returns:
        bool: if the splited lesion is within the convex hull of the lung mask as a criterion for lesion eligibility
    """
    convex_points = np.transpose(np.where(mask_for_convex))
    hull = scipy.spatial.ConvexHull(convex_points)
    deln = scipy.spatial.Delaunay(convex_points[hull.vertices])
    mask_points = np.transpose(np.where(mask_to_check))
    in_convex = len(np.where(deln.find_simplex(mask_points) >= 0)[0]) > 0
    return in_convex


def split_lesions(
    label_filepath: str,
    lung_mask_filepath: str,
    output_dir: str = "split_lesions_output",
) -> None:
    """Split lesions in a 3D label volume using dilation and connected components with SimpleITK.

    Args:
        label_filepath (str): The input 3D label file path.
        output_dir (str): Directory to save split lesion images. Default is 'split_lesions_output'.
    """
    sys.path.append("..")

    # Global constants
    INTENSITY_THRESHOLD = 4
    DILATION_STRUCTURE_VOXELS = (
        None  # Set to an integer for voxel-based dilation, or None
    )
    DILATION_STRUCTURE_MM = 3.0  # Set to a float for millimeter-based dilation, or None
    SIZE_THRESHOLD = 100  # Minimum lesion size to save (in pixels)

    # Read the SimpleITK image
    label_img = sitk.ReadImage(label_filepath, imageIO="NiftiImageIO")
    label_img = sitk.Cast(label_img, sitk.sitkUInt8)
    basename = os.path.basename(label_filepath)
    filename = basename.removesuffix(".nii.gz")

    # Read the lung mask
    lung_mask_img = sitk.ReadImage(lung_mask_filepath, imageIO="NiftiImageIO")
    lung_mask_img = sitk.Cast(lung_mask_img, sitk.sitkUInt8)

    # Threshold lesions in label_img with intensity greater than the global threshold
    label_img = sitk.BinaryThreshold(label_img, 1, INTENSITY_THRESHOLD, 1, 0)

    # Threshold the image to binary for connected components
    connected_components = sitk.BinaryThreshold(label_img, 1, 255)

    # Get spacing and determine dilation size
    spacing = connected_components.GetSpacing()
    if DILATION_STRUCTURE_VOXELS is not None:
        dilate_size = (DILATION_STRUCTURE_VOXELS,) * 3
    elif DILATION_STRUCTURE_MM is not None:
        dilate_size = (
            int(np.ceil(DILATION_STRUCTURE_MM / spacing[0])),
            int(np.ceil(DILATION_STRUCTURE_MM / spacing[1])),
            int(np.ceil(DILATION_STRUCTURE_MM / spacing[2])),
        )
    else:
        dilate_size = (1, 1, 1)

    # Dilate the connected components
    connected_components = sitk.BinaryDilate(
        connected_components, dilate_size, sitk.sitkBall
    )

    # Label connected components
    connected_components = sitk.ConnectedComponent(connected_components)
    relabeled_components = sitk.RelabelComponent(connected_components)

    # Analyze each component
    label_shape_stats = sitk.LabelShapeStatisticsImageFilter()
    label_shape_stats.Execute(relabeled_components)

    # Create output directory if it doesn’t exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each component and save if it meets the size threshold
    for label in label_shape_stats.GetLabels():
        component_mask = sitk.BinaryThreshold(relabeled_components, label, label, 1, 0)
        component_image = sitk.Mask(label_img, component_mask)

        if sitk.GetArrayFromImage(component_image).sum() > 0:
            in_lung_convex = check_in_convex_hull(
                sitk.GetArrayFromImage(lung_mask_img).astype("int8"),
                sitk.GetArrayFromImage(component_image).astype("int8"),
            )
            if not in_lung_convex:
                print(
                    f"Skipping component {label} in lesion mask as it is outside the lung convex mask."
                )
                continue
            final_stats = sitk.LabelShapeStatisticsImageFilter()
            final_stats.Execute(component_image)
            no_pixels = final_stats.GetNumberOfPixels(1)

            if no_pixels >= SIZE_THRESHOLD:
                output_filepath = os.path.join(
                    output_dir, f"{filename}__L{label}.nii.gz"
                )
                sitk.WriteImage(component_image, output_filepath)
                print(f"Saved split lesion: {output_filepath}")
