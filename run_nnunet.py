import boto3
from boto3.s3.transfer import S3Transfer
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
import os

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
        except Exception as e:
            raise ValueError(f"Error during S3 download: {str(e)}")

def nnUNet_predict(image_path: str, output_path: str):
    try:
        # Ensure the model is downloaded
        download_nnunet_model()

        model_folder = 'nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/'

        # Instantiate nnUNetPredictor
        
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