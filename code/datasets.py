import torch
from torch.utils.data import Dataset

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import numpy as np

class MolDataset(Dataset):
    """[A child class that extends an abstract class, used to change the input data]

    Args:
        Dataset ([class]): [An abstract class representing]
    """

    def __init__(self, data):
        self.input_seq, self.time_seq, self.state_seq, self.input_index, self.label = [], [], [], [], []
        for key in data:
            for one_traj in data[key]:
                self.input_index.append(one_traj[0])
                self.label.append(key)

                # Do not use merge window
                # traj = one_traj[1]
                # time = one_traj[2]
                # state = one_traj[3]

                # Use merge window
                traj = [one_traj[1][0]] + [one_traj[1][idx] for idx in range(1, len(one_traj[1])) if one_traj[1][idx] != one_traj[1][idx-1]]
                time = [one_traj[2][0]] + [one_traj[2][idx] for idx in range(1, len(one_traj[1])) if one_traj[1][idx] != one_traj[1][idx-1]]
                state = [one_traj[3][0]] + [one_traj[3][idx] for idx in range(1, len(one_traj[1])) if one_traj[1][idx] != one_traj[1][idx-1]]

                self.input_seq.append(traj)
                self.time_seq.append(time)
                self.state_seq.append(state)

    def __getitem__(self, index):
        return self.input_seq[index], self.time_seq[index], self.state_seq[index], self.input_index[index], self.label[index]

    def __len__(self):
        return len(self.input_seq)


def collate_fn(batch):
    """[Redefine collate_fn function]

    Args:
        batch ([type]): [description]

    Returns:
        [type]: [description]
    """
    traj_contents, time_contents, state_contents, indexs, labels = zip(*batch)
    max_len = max([len(content) for content in traj_contents])

    traj_contents = torch.LongTensor([content + [-1] * (max_len - len(content)) if len(content) < max_len else content for content in traj_contents])
    time_contents = torch.LongTensor([content + [124] * (max_len - len(content)) if len(content) < max_len else content for content in time_contents])
    state_contents = torch.LongTensor([content + [9] * (max_len - len(content)) if len(content) < max_len else content for content in state_contents])

    indexs = torch.LongTensor(indexs)
    labels = torch.LongTensor(labels)
    return traj_contents, time_contents, state_contents, indexs, labels


class MolDataModule(pl.LightningDataModule):
    def __init__(self, user_traj_train, user_traj_test, test_nums, batch_size, val_test_split: float = 0.5):
        super().__init__()
        self.user_traj_train = user_traj_train
        self.user_traj_test = user_traj_test
        self.test_nums = test_nums
        self.batch_size = batch_size
        self.val_test_split = val_test_split

    def setup(self, stage=None):
        self.train_dataset = MolDataset(self.user_traj_train)
        self.test_dataset = MolDataset(self.user_traj_test)

        indices = list(range(self.test_nums))
        split = int(np.floor(self.test_nums * self.val_test_split))

        np.random.seed(555)
        np.random.shuffle(indices)

        valid_idx, test_idx = indices[split:], indices[:split]
        self.val_sampler = SubsetRandomSampler(valid_idx)
        self.test_sampler = SubsetRandomSampler(test_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, sampler=self.val_sampler, drop_last=False, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, sampler=self.test_sampler, drop_last=False, collate_fn=collate_fn)
