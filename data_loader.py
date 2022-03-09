import torch
from torch.utils.data import Dataset, DataLoader
from datasetbuilding import generate_training_data, SEQUENCE_LENGTH

INPUTS, TARGETS = generate_training_data(SEQUENCE_LENGTH)
OUTPUT_SIZE = 38


class trainset(Dataset):
    def __init__(self, inputs=INPUTS, targets=TARGETS):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, index):
        input_data = self.inputs[index]
        traindata = torch.tensor(input_data)
        traindata = traindata.view(SEQUENCE_LENGTH, 1)
        traindata = torch.zeros(SEQUENCE_LENGTH, OUTPUT_SIZE).scatter_(1, traindata, 1)
        target_data = self.targets[index]
        targetdata = torch.tensor(target_data)
        return traindata, targetdata

    def __len__(self):
        return len(self.targets)


if __name__ == '__main__':
    train_data = trainset()
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    dataiter = iter(train_loader)
    data, label = dataiter.next()
    print(f'data size is {data.size()}')
    print(f'label size is {label.size()}')