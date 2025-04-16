import subprocess
import boto3
import raven as rv
import pandas as pd
def run_clearml_tasks(my_series_list):
    """
    Takes a list of S3 URIs and runs the clearml-task command for each URI.

    Args:
        dataset_id (str): The dataset ID.
        s3_uris (list): A list of S3 URIs to be processed.
    """
    # Base command template
    base_command = [
        "clearml-task",
        "--project",
        "koo-phase-2/qvt-feature-optimization/KOO_SCLC_02_nnUNet",
        "--tags",
        "KOO_SCLC_02_nnUNet",
        # "Lung_Lesion_Segmentation_KOO_SCLC_01",
        "--script",
        "__autoscaler_script__.py",
        "--branch",
        "nnUNet_clearml",
        "--requirements",
        "requirements.txt",
        "--queue",
        "LungLesionSegmentator",
    ]

    # Loop through each S3 URI and run the command
    for series in my_series_list:
        experiment_name = f'{series.dataset_id}__{series.clinical_id}__{series.series_uid}'

        # Construct the specific command for the current URI
        command = base_command + [
            "--name",
            experiment_name,
            "--args",
            f"series_uid={series.series_uid}",
            f"s3_output_uri=s3://picturehealth-data/projects/KOO-Phase-2/qvt-phenotype-optimization/QVT_feature_extraction_KOO_SCLC_02/",
        ]

        # Print the command being run (for debugging)
        print(f"Running command: {' '.join(command)}")

        # Run the command using subprocess
        subprocess.run(command, check=True)

if __name__ == "__main__":
    dataset_list = ["KOO-SCLC-02"]
    # dataset_list = ["NORTHWESTERN_CHAE_LAB"]
    # dataset_list = ['UH_SCLC']

    for dataset_id in dataset_list:
        # my_series_list = rv.get_series(
        #     dataset_id = dataset_id,
        #     modality="CT",
        #     body_part_examined="CHEST",
        #     # body_part_examined=['ABDOMEN', 'UNKNOWN', 'NECK', 'HEAD'],
        #     orientation="AXIAL",
        # )
        # my_series_list = rv.get_series(
        #     series_uid=pd.read_csv(
        #         "/Users/haojiali/projects/KOO/code/nnunet_venv/nnunet/passed_uh_sclc_series_batch2.csv"
        #     )["series_uid"].tolist(),
        #     modality="CT",
        #     # body_part_examined="CHEST",
        #     # body_part_examined=['ABDOMEN', 'UNKNOWN', 'NECK', 'HEAD'],
        #     orientation="AXIAL",
        # )
        my_series_list = rv.get_series(
            dataset_id=dataset_id,
            modality="CT",
            orientation="AXIAL",
        ).as_dataframe()

        # ---------debug locally
        # import sys
        # from __autoscaler_script__ import main  # Replace 'script' with the name of your Python file (without .py)
        # # Override sys.argv to simulate command-line arguments
        # sys.argv = [
        #     "__autoscaler_script__.py",  # This mimics the script name
        #     "--series-uid", my_series_list[0].series_uid,
        #     "--s3-output-uri", "s3_output_uri=s3://picturehealth-data/users/haojia/projects/KOO-SCLC-01/"
        # ]
        # # Call the main function
        # main()

        # Run ClearML tasks for each series
        run_clearml_tasks(my_series_list[:1])
