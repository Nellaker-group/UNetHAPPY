import json

from labelbox import Client, Dataset


def main():
    with open("config.json") as f:
        api_key = json.load(f)['LabelBoxAPIKey']

    client = Client(api_key)
    datasets_x = client.get_datasets(where=Dataset.name == "Small Tissue Trial Dataset")
    for x in datasets_x:
        print(x)


if __name__ == "__main__":
    main()