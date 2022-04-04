from pathlib import Path
import json
import uuid

import typer
from tqdm import tqdm
from labelbox import Client, Project, Dataset
from labelbox.schema.ontology import OntologyBuilder

from convert_global_to_local_coords import convert_global_to_local


def main(
    project_name: str = typer.Option(...),
    labelbox_project_name: str = typer.Option(...),
    json_filename: str = typer.Option(...),
    dataset_name: str = typer.Option(...),
    import_job_name: str = typer.Option(...),
    include_labels: bool = False,
):
    """Converts QuPath extracted json file of tissue annotation polygons from global
    coordinates to local coordinates (coords relative to image tile) and uploads
    them to LabelBox as 'pre-label' polygons.

    Args:
        project_name: name of the project directory
        labelbox_project_name: name of project in LabelBox
        json_filename: name of json file generated by QuPath script
        dataset_name: name of dataset in LabelBox to apply annotations to
        import_job_name: name of import/upload job to be displayed in LabelBox
        include_labels: flag for including original tissue label or not
    """
    with open("./config.json") as f:
        api_key = json.load(f)["LabelBoxAPIKey"]
    lb = Client(api_key)
    projects = lb.get_projects(where=Project.name == labelbox_project_name)
    project = next(projects)
    ontology = OntologyBuilder.from_project(project)
    schema_lookup = {tool.name: tool.feature_schema_id for tool in ontology.tools}

    datasets = lb.get_datasets(where=Dataset.name == dataset_name)
    dataset = next(datasets)

    projects_dir = Path(__file__).parent.parent / "projects"
    save_dir = projects_dir / project_name / "results" / "tissue_annots"

    with open(save_dir / json_filename) as f:
        data = json.load(f)

    final_data = []
    for annotation in tqdm(data):
        datarow = dataset.data_row_for_external_id(annotation["image_name"])
        labelbox_image_uid = datarow.uid
        datarow_data = {"id": labelbox_image_uid}

        local_polygon_coords = convert_global_to_local(annotation, subregion=False)

        final_dict = {
            "uuid": str(uuid.uuid4()),
            "schemaId": get_schema_id(
                schema_lookup, include_labels, annotation["tissue_type"]
            ),
            "dataRow": datarow_data,
            "polygon": local_polygon_coords,
        }
        final_data.append(final_dict)

    upload_job = project.upload_annotations(
        name=import_job_name, annotations=final_data
    )
    upload_job.wait_until_done()
    print("State", upload_job.state)


def get_schema_id(schema_lookup, include_labels, tissue_label=None):
    if not include_labels:
        return schema_lookup["Unlabeled"]
    else:
        return schema_lookup[TISSUE_TYPE_MAP[tissue_label]]


TISSUE_TYPE_MAP = {
    "TVilli": "Terminal Villi",
    "MIVilli": "Mature Intermediary Villi",
    "ImIVilli": "Immature Intermediary Villi",
    "SVilli": "Stem Villi",
    "AVilli": "Anchoring Villi",
    "MVilli": "Mesenchymal Villi",
    "Sprout": "Villus Sprout",
    "Chorion": "Chorion/Amnion",
    "Avascular": "Avascular Villi",
    "Maternal": "Basal Plate/Septa",
    "Fibrin": "Fibrin",
    "Inflam": "Inflammatory Response",
}

if __name__ == "__main__":
    typer.run(main)
