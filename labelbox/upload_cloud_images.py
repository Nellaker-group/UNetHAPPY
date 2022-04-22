import json
from typing import List

import typer
from labelbox import Client, Dataset
from google.cloud import storage


def main(
    bucket_name: str = typer.Option(...),
    dir_names: List[str] = typer.Option([]),
    dataset_name: str = typer.Option(...),
    include_polygon_layer: bool = True,
):
    """Links images in a google cloud container to a LabelBox dataset. Adds
    the specified text attachment to the image and an image layer with polygon drawing
    if specified.

    Args:
        bucket_name: Name of the container in google cloud
        dir_names: Paths to directories containing the images you want to link
        dataset_name: Name of an existing or new dataset in LabelBox
        include_polygon_layer: Include an image layer with drawn polygon
    """
    storage_client = storage.Client()

    with open("config.json") as f:
        api_key = json.load(f)["LabelBoxAPIKey"]

    lb = Client(api_key)
    datasets = lb.get_datasets(where=Dataset.name == dataset_name)
    try:
        dataset = next(datasets)
    except StopIteration:
        print(f"Creating new dataset: {dataset_name}")
        dataset = lb.create_dataset(name=dataset_name)
    print(dataset)

    print("Getting image data from Google Cloud...")
    uploads = []
    for dir_name in dir_names:
        if include_polygon_layer:
            dir_name = dir_name[:-1] + "-polygon/"
        blob_uri = f"gs://{bucket_name}/{dir_name}"
        blobs = storage_client.list_blobs(bucket_name, prefix=dir_name)

        for image in blobs:
            if image.name != dir_name:
                image_name = image.name.split(dir_name)[-1]
                upload_dict = {
                    "external_id": image_name,
                    "row_data": f"{blob_uri}{image_name}",
                    "attachments": [
                        {
                            "type": "TEXT",
                            "value": "Term placenta with no complications",
                        }
                    ],
                }
                if include_polygon_layer:
                    original_image_dir_name = dir_name.split('-polygon/')[-2]
                    polygon_image_path = (
                        f"gs://{bucket_name}/{original_image_dir_name}/{image_name}"
                    )
                    upload_dict["attachments"].append(
                        {"type": "IMAGE_OVERLAY", "value": polygon_image_path}
                    )
                uploads.append(upload_dict)
    print("Image data obtained!")

    print("Linking Images to LabelBox...")
    task1 = dataset.create_data_rows(uploads)
    task1.wait_till_done()
    print(f"Linking {task1.status}!")

    print("Done!")


if __name__ == "__main__":
    typer.run(main)
