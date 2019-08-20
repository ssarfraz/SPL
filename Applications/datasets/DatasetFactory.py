class DatasetFactory:
	def __init__(self, dataset_name, data_directory, mean, augment=True, num_classes=None):
		self._data_directory = data_directory
		self._dataset_name = dataset_name
		self._augment = augment
		self._num_classes = num_classes
		self._mean=mean


	def get_dataset(self, dataset_part):
		if self._dataset_name == 'makeup':
			from datasets.MakeupDataset import MakeupDataset
			return MakeupDataset(data_directory=self._data_directory, dataset_part=dataset_part, mean=self._mean, augment=self._augment, num_classes=self._num_classes)

		elif self._dataset_name == 'hilo':
			from datasets.HiLoDataset import HiLoDataset
			return HiLoDataset(data_directory=self._data_directory, dataset_part=dataset_part, mean=self._mean, augment=self._augment, num_classes=self._num_classes)

		elif self._dataset_name == 'img_trans':
			from datasets.ImageTranslationDataset import ImageTranslationDataset
			return ImageTranslationDataset(data_directory=self._data_directory, dataset_part=dataset_part, mean=self._mean, augment=self._augment, num_classes=self._num_classes)

		else:
			raise ValueError('Unknown dataset name: %s' % self._data_directory)


	def get_dataset_name(self):
		return self._dataset_name