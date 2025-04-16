import raven as rv
import SimpleITK as sitk
import os
from crop_volumes import run_lungmask_crop_on_dataset
from pathlib import Path


def curate_input_image(series_uid, image_save_path):
  os.makedirs(f"{image_save_path}", exist_ok=True)
  os.makedirs(f"{image_save_path}lungmask-volumes", exist_ok=True)
  os.makedirs(f"{image_save_path}image-volumes", exist_ok=True)

  image = rv.get_images(series_uid = series_uid)[0]
  mask = rv.get_masks(series_uid = series_uid, mask_type="LUNG_MASK")[0]

  mask_sitk = rv.as_sitk(mask)
  image_sitk = rv.as_sitk(image)

  nifti_filepath_mask = f"{image_save_path}lungmask-volumes/{image.dataset_id}__{image.clinical_id}__{series_uid}.nii.gz"
  nifti_filepath_image = f"{image_save_path}image-volumes/{image.dataset_id}__{image.clinical_id}__{series_uid}.nii.gz"
  # Save the image as a NIfTI file
  sitk.WriteImage(mask_sitk, nifti_filepath_mask)
  sitk.WriteImage(image_sitk, nifti_filepath_image)

  # n_jobs = 1
  # run_lungmask_crop_on_dataset(Path(image_save_path), n_jobs=n_jobs)