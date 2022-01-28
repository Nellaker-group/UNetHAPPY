import json

from labelbox import Client, Dataset
from google.cloud import storage


def main():
    storage_client = storage.Client()
    blobs = storage_client.list_blobs("labelboxplacentaimages")
    dir_name = "slide_139_estonia/"
    blob_uri = f"gs://labelboxplacentaimages/{dir_name}"

    with open("config.json") as f:
        api_key = json.load(f)["LabelBoxAPIKey"]

    lb = Client(api_key)
    dataset_name = "placenta_tissues_chorioamnionitis"
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
                "attachment": [
                    {
                        "type": "TEXT",
                        "value": "20+4 weeks placenta with acute chorioamnionitis.",
                    }
                ],
            }
            uploads.append(upload_dict)
    print("Image data obtained!")

    print("Linking Images to LabelBox...")
    task1 = dataset.create_data_rows(uploads)
    task1.wait_till_done()
    print(f"Linking {task1.status}!")

    # print("Adding text attachment to all LabelBox images")
    # for item in dataset.data_rows():
    #     item.create_attachment(
    #         attachment_type="TEXT",
    #         attachment_value="20+4 weeks placenta with acute chorioamnionitis.",
    #     )
    print("Done!")


if __name__ == "__main__":
    main()
