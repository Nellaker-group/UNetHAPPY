from projects.placenta.graphs.graphs.lesion_dataset import LesionDataset


def setup_lesion_datasets(
    organ, project_dir, combine=True, test=False, lesions_to_remove=None, local=False
):
    datasets = {}
    if local:
        datasets["train"] = LesionDataset(
            organ, project_dir, "single", "val", lesions_to_remove
        )
        datasets["train"].data_paths = datasets["train"].data_paths[:2]
        datasets["train"].lesions = datasets["train"].lesions[:2]
        datasets["train"].run_ids = datasets["train"].run_ids[:2]
        datasets['val'] = datasets['train']
        return datasets
    if combine:
        single_lesion_train_data = LesionDataset(
            organ, project_dir, "single", "train", lesions_to_remove
        )
        multi_lesion_train_data = LesionDataset(
            organ, project_dir, "multi", "train", lesions_to_remove
        )
        datasets["train"] = single_lesion_train_data.combine_with_other_dataset(
            multi_lesion_train_data
        )
        single_lesion_val_data = LesionDataset(
            organ, project_dir, "single", "val", lesions_to_remove
        )
        multi_lesion_val_data = LesionDataset(
            organ, project_dir, "multi", "val", lesions_to_remove
        )
        datasets["val"] = single_lesion_val_data.combine_with_other_dataset(
            multi_lesion_val_data
        )
        if test:
            single_lesion_test_data = LesionDataset(
                organ, project_dir, "single", "test", lesions_to_remove
            )
            multi_lesion_test_data = LesionDataset(
                organ, project_dir, "multi", "test", lesions_to_remove
            )
            datasets["test"] = single_lesion_test_data.combine_with_other_dataset(
                multi_lesion_test_data
            )
    else:
        datasets["train_single"] = LesionDataset(
            organ, project_dir, "single", "train", lesions_to_remove
        )
        datasets["train_multi"] = LesionDataset(
            organ, project_dir, "multi", "train", lesions_to_remove
        )
        datasets["val_single"] = LesionDataset(
            organ, project_dir, "single", "val", lesions_to_remove
        )
        datasets["val_multi"] = LesionDataset(
            organ, project_dir, "multi", "val", lesions_to_remove
        )
        if test:
            datasets["test_single"] = LesionDataset(
                organ, project_dir, "single", "test", lesions_to_remove
            )
            datasets["test_multi"] = LesionDataset(
                organ, project_dir, "multi", "test", lesions_to_remove
            )
    return datasets
