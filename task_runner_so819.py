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
        "Lung_Lesion_Segmentation_S0819-NSCLC",
        "--script",
        "__autoscaler_script__.py",
        "--branch",
        "nnUNet_clearml",
        "--requirements", "requirements.txt",
        "--queue",
        "LungLesionSegmentator",
        "--tags",
        "S0819-NSCLC",
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
            f"s3_output_uri=s3://picturehealth-data/users/omid/projects/S0819-NSCLC/",
        ]

        # Print the command being run (for debugging)
        print(f"Running command: {' '.join(command)}")

        # Run the command using subprocess
        subprocess.run(command, check=True)


if __name__ == "__main__":
    all_series_list = rv.get_series(dataset_id = 'S0819-NSCLC',)

    annotation_info = pd.read_csv('s3://picturehealth-data/users/omid/projects/SO819/S0819_annotation_metadata.csv')
    my_series_list = []
    for series in all_series_list:
        if series.series_uid  in annotation_info["ReferencedSeriesInstanceUID"].values:
            my_series_list.append(series)

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
    run_clearml_tasks(my_series_list[1:3])
