import json

from labelbox import Client, Project
from labelbox.schema.bulk_import_request import BulkImportRequest


def main():
    with open("config.json") as f:
        api_key = json.load(f)["LabelBoxAPIKey"]

    lb = Client(api_key)
    projects = lb.get_projects(where=Project.name == "Placenta Tissues")
    project = next(projects)

    bulk_import_request = BulkImportRequest.from_name(
        lb, project.uid, "first_three_in_queue"
    )
    bulk_import_request.delete()


if __name__ == "__main__":
    main()
