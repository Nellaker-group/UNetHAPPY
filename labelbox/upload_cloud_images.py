import json

import typer
from labelbox import Client, Dataset
from google.cloud import storage


def main(
    bucket_name: str = typer.Option(...),
    dir_name: str = typer.Option(...),
    dataset_name: str = typer.Option(...),
    include_polygon_layer: bool = True,
):
    """Links images in a google cloud container to a LabelBox dataset. Adds
    the specified text attachment to the image and an image layer with polygon drawing
    if specified.

    Args:
        bucket_name: Name of the container in google cloud
        dir_name: Path to directory containing the images you want to link
        dataset_name: Name of an existing or new dataset in LabelBox
        include_polygon_layer: Include an image layer with drawn polygon
    """
    storage_client = storage.Client()
    blob_uri = f"gs://{bucket_name}/{dir_name}"
    blobs = storage_client.list_blobs(bucket_name, prefix=dir_name)

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
    for i, image in enumerate(blobs):
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
                polygon_dir_name = dir_name[:-1]
                polygon_image_path = (
                    f"gs://{bucket_name}/{polygon_dir_name}-polygon/{image_name}"
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
