import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pytorch_lightning as pl


class CDMFDataModule(pl.LightningDataModule):
    """
    Example of a DataModule handling data for the CDMF model.
    Currently only generated random data.
    """
    def __init__(self, dataset, max_seq_len, batch_size=32, num_workers=0):
        super().__init__()
        self.dataset = dataset
        self.max_seq_len = max_seq_len

        self.n_users = None
        self.n_items = None
        self.n_features = None

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        """
        Prepares the dataset (download or create)
        Returns:
            user2seq - dict holding for each user a dictionary holding:
                            - items: list of item interaction
                            - features: list of features per item interaction
                            - timestamps: list of timestamp for each interaction
                            - labels: 1 or 0 per item interaction
        """
        self.user2seq = {}
        if self.dataset == 'random':
            self.n_users = 100
            self.n_items = 10
            self.n_features = 64

            for user in torch.range(1, self.n_users, dtype=torch.long):
                seq_len = torch.randint(low=1, high=self.max_seq_len, size=(1,))[0]
                self.user2seq[user] = {'items': torch.randint(low=1, high=self.n_items + 1, size=(seq_len,)),
                                       'features': torch.randn(size=(seq_len, self.n_features)),
                                       'timestamps': torch.randint(low=1, high=self.max_seq_len * self.n_users, size=(seq_len,)),
                                       'labels': torch.randint(low=0, high=2, size=(seq_len,))}
        else:
            raise Exception(f'dataset {self.dataset} not supported')

    def setup(self, stage=None):
        """
        Splits the data into train/val/test by splitting the timeline
        """
        self.datasets = []
        all_timestamps = torch.cat([seq['timestamps'] for seq in self.user2seq.values()])
        qs = (0., 0.7, 0.85, 1.)
        for i in range(1, len(qs)):  # split to train/val/test on timeline
            q = torch.Tensor([qs[i - 1], qs[i]])
            start_timestamp, end_timestamp = torch.quantile(all_timestamps.float(), q=q).tolist()

            cur_user2seq = {}
            for user in self.user2seq:
                seq = self.user2seq[user]
                time_mask = (start_timestamp <= seq['timestamps']) * (seq['timestamps'] <= end_timestamp)
                if time_mask.sum() > 0:
                    cur_user2seq[user] = {'items': seq['items'][time_mask],
                                          'features': seq['features'][time_mask],
                                          'timestamps': seq['timestamps'][time_mask],
                                          'labels': seq['labels'][time_mask]}
            self.datasets.append(CDMFDataset(cur_user2seq))

    def train_dataloader(self):
        return DataLoader(self.datasets[0], shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.pad_collate)

    def val_dataloader(self):
        return DataLoader(self.datasets[1], shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.pad_collate)

    def test_dataloader(self):
        return DataLoader(self.datasets[2], shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.pad_collate)

    def pad_collate(self, batch):
        """
        This function is responsible of merging all user-item interaction sequences to a full batch.
        Notice the final batch size might be greater then defined in the DataLoader
        """
        batch = [sample for samples in batch for sample in samples]
        max_len = max([len(sample['features']) for sample in batch])
        for i in range(len(batch)):
            cur_len = len(batch[i]['features'])
            pad_len = max_len - cur_len
            batch[i]['features'] = F.pad(input=batch[i]['features'], pad=(0, 0, pad_len, 0), mode='constant', value=0)
            batch[i]['mask'] = torch.BoolTensor([0] * pad_len + [1] * cur_len)
        return default_collate(batch)


class CDMFDataset(Dataset):
    def __init__(self, user2seq):
        super(CDMFDataset, self).__init__()
        self.user2seq = user2seq
        self.users = list(self.user2seq.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        """
        Returns:
            return a list of all item interaction sequences for the current user. Please notice this function
            flattens all user-item pairs to different samples
        """
        user = self.users[index]
        seq = self.user2seq[user]

        sorted_indices = torch.argsort(seq['timestamps'])
        items = seq['items'][sorted_indices]
        features = seq['features'][sorted_indices]
        labels = seq['labels'][sorted_indices]

        unique_items = torch.unique(items)
        return [{'user': user,
                 'item': item,
                 'features': features[items == item][:-1],
                 'labels': labels[items == item][-1]} for item in unique_items if sum(items == item) > 2]
