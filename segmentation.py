import SimpleITK as sitk
from pathlib import Path
import os
import torch
import boto3
from botocore.exceptions import NoCredentialsError
from boto3.s3.transfer import S3Transfer  # Import S3Transfer explicitly
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


import sys
sys.path.append("..")

def crop_volume_to_lung(img, lm):
    lm = sitk.BinaryThreshold(lm, 1, 3, 1, 0)

    if sitk.GetArrayFromImage(lm).sum() == 0:
        return None

    labelfilter = sitk.LabelShapeStatisticsImageFilter()
    labelfilter.Execute(lm)

    startx, starty, startz, xsize, ysize, zsize = labelfilter.GetBoundingBox(1)

    endx = startx + xsize
    endy = starty + ysize
    endz = startz + zsize

    cropped = img[startx:endx, starty:endy, startz:endz]

    crop_coords = (startx, endx, starty, endy, startz, endz)

    return cropped, tuple(crop_coords)

def crop_file_to_lung(file_to_crop, lungmask_file):
    img = sitk.ReadImage(str(file_to_crop))
    lm = sitk.ReadImage(str(lungmask_file))

    return crop_volume_to_lung(img, lm)

def crop_file_to_lung_and_save(file_to_crop, lungmask_file, output_file):
    os.makedirs(Path(output_file).parent, exist_ok=True)
    cropped, crop_coords = crop_file_to_lung(file_to_crop, lungmask_file)

    if cropped is None:
        print(f"ERROR: {file_to_crop} failed due to empty lungmask")
        return

    sitk.WriteImage(cropped, str(output_file))

    return crop_coords

def revert_cropped_image(file_to_revert, crop_coords, original_image_path, output_file):
    if not os.path.exists(file_to_revert) or not os.path.exists(original_image_path):
        print(f"ERROR: {file_to_revert} or {original_image_path} does not exist")
        return

    os.makedirs(Path(output_file).parent, exist_ok=True)
    
    reverted_image = sitk.ReadImage(str(original_image_path))
    reverted_image = sitk.BinaryThreshold(reverted_image, 0, 1000, 0, 0)
    cropped_image = sitk.ReadImage(str(file_to_revert))
    reverted_image[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3], crop_coords[4]:crop_coords[5]] = cropped_image

    sitk.WriteImage(reverted_image, str(output_file))

    print(f"Segmented label reverted to original size saved at {output_file}")

def download_nnunet_model():
    bucket_name = "picturehealth-data"
    s3_folder_path = "projects/nnUNet/nnUNet_results/Dataset505_Ryver_training_set_full_auto/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/"
    local_folder = "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/"

    # Create the local model folder if it does not exist
    os.makedirs(local_folder, exist_ok=True)

    # Initialize S3 client and transfer manager
    s3_client = boto3.client("s3")
    transfer = S3Transfer(s3_client)

    # List and download files
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        print("Downloading NNUNET model...")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder_path):
            for obj in page.get("Contents", []):
                # Strip S3 prefix to get the relative file path
                relative_path = obj["Key"].replace(s3_folder_path, "")
                local_file_path = os.path.join(local_folder, relative_path)

                # Ensure the local directory exists
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the file
                # print(f"Downloading {obj['Key']} to {local_file_path}")
                transfer.download_file(bucket_name, obj["Key"], local_file_path)
        print("Download completed successfully.")
    except NoCredentialsError:
        raise ValueError("AWS credentials not found.")
    except Exception as e:
        raise ValueError(f"Error during S3 download: {str(e)}")

def nnUNet_predict(image_path: str, output_path: str):
    try:
        # Ensure the model is downloaded
        download_nnunet_model()

        model_folder = 'nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/'

        # Instantiate nnUNetPredictor
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )

        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=(0,),
            checkpoint_name="checkpoint_final.pth",
        )

        if image_path.endswith(".nii.gz"):
            print(f"Prediction of {image_path} in progress...")
            predictor.predict_from_files([[image_path]],
                                        [output_path],
                                        save_probabilities=False,
                                        overwrite=False,
                                        num_processes_preprocessing=3,
                                        num_processes_segmentation_export=3,
                                        folder_with_segs_from_prev_stage=None,
                                        num_parts=1,
                                        part_id=0)
        print(f'Predictions done, saving to {output_path}')

        return True
    except Exception as e:
        raise ValueError(f"Error during nnUNet prediction: {str(e)}")



