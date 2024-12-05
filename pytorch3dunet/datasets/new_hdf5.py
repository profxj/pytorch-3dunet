import h5py

from pytorch3dunet.unet3d.utils import get_logger
import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import calculate_stats, ConfigDataset
from pytorch3dunet.datasets.hdf5 import traverse_h5_paths

logger = get_logger('HDF5Dataset')

class XHDF5Dataset(ConfigDataset):
    """Represents an abstract HDF5 dataset.
    
    Parameters:
        file_path: Path to the HDF5 file.
        dataset_names: List of dataset names to gather. 
            Objects will be returned in this order.
        return_mask: Return the mask on a call to __getitem__?
    """
    def __init__(self, file_path, phase:str, transformer_config:dict, 
                 raw_internal_path='raw', label_internal_path='label', weight_internal_path=None,
                 global_normalization:bool=True):
        super().__init__()
        assert phase in ['train', 'val', 'test']

        self.phase = phase
        self.file_path = file_path
        self.raw_internal_path = raw_internal_path
        self.label_internal_path = label_internal_path
        #self.weight_internal_path = weight_internal_path
        self.h5f = h5py.File(file_path, 'r')

        self.raw = self.h5f[raw_internal_path]
        self.label = self.h5f[label_internal_path] if phase != 'test' else None
        #weight_map = f[weight_internal_path] if weight_internal_path is not None else None

        if global_normalization:
            logger.info('Calculating mean and std of the raw data...')
            with h5py.File(file_path, 'r') as f:
                raw = f[raw_internal_path][:]
                stats = calculate_stats(raw)
        else:
            stats = calculate_stats(None, True)

        self.transformer = transforms.Transformer(transformer_config, stats)
        self.raw_transform = self.transformer.raw_transform()

        if phase != 'test':
            # create label/weight transform only in train/val phase
            self.label_transform = self.transformer.label_transform()

    def __len__(self):
        return self.raw.shape[0]
    
    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration

        raw_transformed = self.raw_transform(self.raw[index])
        if self.phase == 'test':
            import pdb; pdb.set_trace()
        else:
            label_transformed = self.label_transform(self.label[index])

            # JXP Scaling hack
            import pdb; pdb.set_trace()

            return raw_transformed, label_transformed

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]

        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        file_paths = phase_config['file_paths']
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths
        file_paths = traverse_h5_paths(file_paths)

        datasets = []
        for file_path in file_paths:
            try:
                logger.info(f'Loading {phase} set from: {file_path}...')
                dataset = cls(file_path=file_path,
                              phase=phase,
                              #slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              raw_internal_path=dataset_config.get('raw_internal_path', 'raw'),
                              label_internal_path=dataset_config.get('label_internal_path', 'label'),
                              weight_internal_path=dataset_config.get('weight_internal_path', None),
                              global_normalization=dataset_config.get('global_normalization', None))
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
        return datasets