import argparse
import os
import sys
import boto3

from pathlib import Path

from query_raven_image import curate_input_image
from run_nnunet import nnUNet_predict
from lesion_splitter import split_lesions
import glob
# initialize s3 client
s3_client = boto3.client("s3")


def upload_output_folder_to_s3(output_folder, s3_uri):
    """
    Uploads the entire contents of the specified output folder to the given S3 URI.

    Args:
        output_folder (str): The local output folder to upload.
        s3_uri (str): The S3 URI to upload the folder to, e.g., 's3://bucket/path/'.
    """
    # Extract bucket and prefix (key) from S3 URI
    s3_parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = s3_parts[0]
    prefix = s3_parts[1] if len(s3_parts) > 1 else ""

    # Iterate over all files and subdirectories in the output folder recursively
    for root, _, files in os.walk(output_folder):
        for file_name in files:
            # Construct the local file path
            local_path = os.path.join(root, file_name)

            # Construct the S3 key by preserving the folder structure
            relative_path = os.path.relpath(local_path, output_folder)
            s3_key = os.path.join(prefix, relative_path)

            # Upload the file to S3
            print(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
            s3_client.upload_file(local_path, bucket, s3_key)
            print(f"Uploaded {file_name} successfully.")


def upload_output_file_to_s3(output_path, s3_uri):
    """
    Uploads the specified output file to the given S3 URI.

    Args:
        output_path (str): The local output file to upload.
        s3_uri (str): The S3 URI to upload the file to, e.g., 's3://bucket/path/file'.
    """
    # Extract bucket and key from S3 URI
    s3_parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = s3_parts[0]
    key = s3_parts[1] if len(s3_parts) > 1 else ""

    # Upload file to S3
    print(f"Uploading {output_path} to {s3_uri}")
    s3_client.upload_file(output_path, bucket, key)
    print(f"Uploaded file successfully to {s3_uri}")


def main():
    parser = argparse.ArgumentParser(
        prog="nnunet Lung Lesion Segmentation AutoScaler Pipeline"
    )
    parser.add_argument(
        "--series-uid", required=True, type=str, help="uid of the series to be processed"
    )
    parser.add_argument(
        "--s3-output-uri",
        # default="s3://cml-storage/rushil/test_outputs/",
        type=str,
        help="Output directory in S3.",
    )

    _args = parser.parse_args()
    image_save_path = "images/"
    curate_input_image(_args.series_uid, image_save_path)
    # pdb.set_trace()
    # download_nnunet_model()
    input_image_path = glob.glob(f'{image_save_path}cropped-image-volumes/*nii.gz')[0]
    output_path = os.path.basename(input_image_path).replace('.nii.gz', '')
    output_dir = 'outputs/'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}{os.path.basename(input_image_path)}"
    nnUNet_predict(input_image_path, output_path)

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------

    print(f"Starting lesion splitting")
    lesion_split_dir = 'splitted_lesions/'
    # Step 2: Define output directory for split lesions
    os.makedirs(lesion_split_dir, exist_ok=True)

    # Step 3: Run the lesion splitting function
    try:
        split_lesions(
            label_filepath=output_path,
            output_dir=lesion_split_dir,
        )
        print("Lesion splitting completed successfully.")
        upload_output_folder_to_s3(lesion_split_dir, f"{_args.s3_output_uri}cropped-splitlabel-volumes/")
    except Exception as e:
        print(f"Error during lesion splitting: {str(e)}")

    #
    upload_output_file_to_s3(
        input_image_path, f"{_args.s3_output_uri}cropped-image-volumes/{os.path.basename(input_image_path)}"
    )
    upload_output_file_to_s3(output_path, f"{_args.s3_output_uri}cropped-label-volumes/{os.path.basename(output_path)}")
    upload_output_file_to_s3(
        input_image_path.replace("cropped-image-volumes", "cropped-lungmask-volumes"),
        f'{_args.s3_output_uri}cropped-lungmask-volumes/{os.path.basename(input_image_path)}',
    )

if __name__ == "__main__":
    sys.exit(main())
