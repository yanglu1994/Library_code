from gan_study import mnist

datasets_map = {
    'mnist': mnist
}
def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
    if name not in datasets_map:
        raise ValueError('name of dataset unkonw %s' % name)
    return datasets_map[name].get_split(
        split_name,
        dataset_dir,
        file_pattern,
        reader
    )
