from projects.placenta.graphs.graphs.lesion_dataset import LesionDataset


def setup_lesion_datasets(organ, project_dir, combine=True, test=False):
    datasets = {}
    if combine:
        single_lesion_train_data = LesionDataset(organ, project_dir, "single", "train")
        multi_lesion_train_data = LesionDataset(organ, project_dir, "multi", "train")
        datasets["train"] = single_lesion_train_data.combine_with_other_dataset(
            multi_lesion_train_data
        )
        single_lesion_val_data = LesionDataset(organ, project_dir, "single", "val")
        multi_lesion_val_data = LesionDataset(organ, project_dir, "multi", "val")
        datasets["val"] = single_lesion_val_data.combine_with_other_dataset(
            multi_lesion_val_data
        )
        if test:
            single_lesion_test_data = LesionDataset(
                organ, project_dir, "single", "test"
            )
            multi_lesion_test_data = LesionDataset(organ, project_dir, "multi", "test")
            datasets["test"] = single_lesion_test_data.combine_with_other_dataset(
                multi_lesion_test_data
            )
    else:
        datasets["train_single"] = LesionDataset(organ, project_dir, "single", "train")
        datasets["train_multi"] = LesionDataset(organ, project_dir, "multi", "train")
        datasets["val_single"] = LesionDataset(organ, project_dir, "single", "val")
        datasets["val_multi"] = LesionDataset(organ, project_dir, "multi", "val")
        if test:
            datasets["test_single"] = LesionDataset(
                organ, project_dir, "single", "test"
            )
            datasets["test_multi"] = LesionDataset(organ, project_dir, "multi", "test")
    return datasets
