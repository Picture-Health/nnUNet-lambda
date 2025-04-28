import argparse
import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


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

    return cropped


def crop_file_to_lung(file_to_crop, lungmask_file):
    img = sitk.ReadImage(str(file_to_crop))
    lm = sitk.ReadImage(str(lungmask_file))

    return crop_volume_to_lung(img, lm)


def crop_file_to_lung_and_save(file_to_crop, lungmask_file, output_file):
    if not os.path.exists(output_file):
        # print(file_to_crop)
        os.makedirs(Path(output_file).parent, exist_ok=True)

        if os.path.isfile(lungmask_file):
            cropped = crop_file_to_lung(file_to_crop, lungmask_file)
        else:
          print(f"ERROR: {file_to_crop} failed due to non-existed lungmask")
          return

        if cropped is None:
            print(f"ERROR: {file_to_crop} failed due to empty lungmask")
            return

        sitk.WriteImage(cropped, str(output_file))


def run_lungmask_crop_on_folder(
    dataset_path,
    folder_to_crop,
    n_jobs=1,
    lungmask_folder="lungmask-volumes",
    output_prefix="cropped-",
):
    files = sorted(dataset_path.joinpath(folder_to_crop).glob("*.nii*"))
    print(len(files))

    lungmask_files = [str(f).replace(folder_to_crop, lungmask_folder) for f in files]

    output_path = dataset_path.joinpath(f"{output_prefix}{folder_to_crop}")
    output_path.mkdir(parents=True, exist_ok=True)

    output_files = [output_path.joinpath(f.name) for f in files]

    Parallel(n_jobs=n_jobs)(
        delayed(crop_file_to_lung_and_save)(file_to_crop, lungmask_file, output_file)
        for file_to_crop, lungmask_file, output_file in zip(
            files, lungmask_files, output_files
        )
    )


def run_lungmask_crop_on_dataset(dataset_path, n_jobs=1):
    folders_to_crop = ["image-volumes", "label-volumes", "lungmask-volumes"]

    for folder_to_crop in folders_to_crop:
        print(f"------CROPPING {folder_to_crop}-------")

        run_lungmask_crop_on_folder(
            dataset_path, folder_to_crop=folder_to_crop, n_jobs=n_jobs
        )


def run_lungmask_crop_on_dataset_df(
    dataset_df: pd.DataFrame,
    image_column: str = "Image",
    lesionwise_label_column: str = "Mask",
    lungmask_column: str = "OrganMask",
    output_dir: str = "outputs",
    n_jobs: int = 1,
):
    logging.info(f"------CROPPING by {lungmask_column}-------")
    os.makedirs(output_dir, exist_ok=True)
    # Assumes dataset_df has columns image-volume, lesionwise-label-volume, and lungmask-volume
    tasks = []

    # Create tasks based on the DataFrame's entries
    for _, row in dataset_df.iterrows():
        image_path = row[image_column]
        lesionwise_label_path = row[lesionwise_label_column]
        lungmask_path = row[lungmask_column]
        image_output_path = Path(output_dir).joinpath(
            "cropped-image-volumes", Path(image_path).name
        )
        label_output_path = Path(output_dir).joinpath(
            "cropped-lesionwise-label-volumes", Path(lesionwise_label_path).name
        )
        tasks.append((image_path, lungmask_path, image_output_path))
        tasks.append((lesionwise_label_path, lungmask_path, label_output_path))
    num_tasks = len(tasks)

    if n_jobs < 0:
        n_jobs = os.cpu_count() + 1 + n_jobs
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks to the executor
        futures = [
            executor.submit(crop_file_to_lung_and_save, task[0], task[1], task[2])
            for task in tasks
        ]

        # Wrap it with tqdm and as_completed for progress updates
        for future in tqdm(
            as_completed(futures), total=num_tasks, desc="Cropping volumes"
        ):
            try:
                # We don't need to capture the result because the function is assumed to save the output
                future.result()
            except Exception as exc:
                print(f"An exception occurred: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(
        "Crop image, label, and lungmask to lungmask. Assumes dataset-path includes subfolders image-volumes, label-volumes, and lungmask-volumes"
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        help="path to the dataset. Ex ..../NSCLC-Radiogenomics",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=2,
        help="number of jobs to run in parallel",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.dataset_path = '/home/ubuntu/nnunet_venv/original_images/all_chemo_io'
    args.dataset_path = '/home/ubuntu/nnunet_venv/test_raven'
    n_jobs = args.n_jobs
    dataset_path = Path(args.dataset_path)
    run_lungmask_crop_on_dataset(dataset_path, n_jobs=n_jobs)
    

