import torch
from tfrecord.torch.dataset import TFRecordDataset

tfrecord_path = "/data2/gyn/PycharmProjects/AAAI2021ATTMPT/incremental_detectors-master/incremental_detectors-master/datasets/voc07-trainval-proposals"
index_path = None
description = {"image": "byte", "label": "float"}
dataset = TFRecordDataset(tfrecord_path, index_path, description)
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

data = next(iter(loader))
print(data)