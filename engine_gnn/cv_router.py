"""
交叉验证隔离引擎 (Cross Validation Engine)
负责构造 LOTO / LOSO 数据流，并对独立验证被试保持显式隔离。
"""
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset

try:
    from .dataloader import SEEDGraphDataset
except ImportError:
    from dataloader import SEEDGraphDataset


def get_loto_loaders(data_dir, subject_id, day_idx, test_trial_idx, batch_size=32):
    train_dataset = SEEDGraphDataset(
        root_dir=data_dir,
        subject_id=subject_id,
        day_idx=day_idx,
        mode='train',
        test_trial_idx=test_trial_idx,
    )
    test_dataset = SEEDGraphDataset(
        root_dir=data_dir,
        subject_id=subject_id,
        day_idx=day_idx,
        mode='test',
        test_trial_idx=test_trial_idx,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_loso_loaders(data_dir, test_subject_id, batch_size=32):
    train_datasets = []
    val_datasets = []
    test_datasets = []
    val_subject_id = (test_subject_id % 15) + 1

    for subj in range(1, 16):
        for day_idx in range(3):
            if subj == test_subject_id:
                test_datasets.append(SEEDGraphDataset(data_dir, subj, day_idx, mode='test', test_trial_idx=-1))
            elif subj == val_subject_id:
                val_datasets.append(SEEDGraphDataset(data_dir, subj, day_idx, mode='train', test_trial_idx=-1))
            else:
                train_datasets.append(SEEDGraphDataset(data_dir, subj, day_idx, mode='train', test_trial_idx=-1))

    full_train_dataset = ConcatDataset(train_datasets)
    full_val_dataset = ConcatDataset(val_datasets)
    full_test_dataset = ConcatDataset(test_datasets)

    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(full_val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
