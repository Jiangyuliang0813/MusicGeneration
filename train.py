from datasetbuilding import SEQUENCE_LENGTH
from data_loader import OUTPUT_SIZE
import torch
from model import lstm_model
from data_loader import trainset
from torch.utils.data import DataLoader
import time
from torchinfo import summary

BATCH_SIZE = 128
INPUT_SIZE = 38
SEQUENCE_LENGTH = SEQUENCE_LENGTH
NUM_UNIT = 256
NUM_LAYERS = 1
LEARNING_RATE = 0.001
EPOCHS = 50
if torch.cuda.is_available():
    print(f'Using gpu {torch.cuda.get_device_name(0)}')
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')

traindataset = trainset()

train_dataloader = DataLoader(traindataset, BATCH_SIZE, shuffle=True)

model = lstm_model(inputs_size=INPUT_SIZE,
                   num_unit=NUM_UNIT,
                   outputs_size=OUTPUT_SIZE,
                   num_layers=NUM_LAYERS,
                   sequence_length=SEQUENCE_LENGTH,
                   device = device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

datalength = len(traindataset)
model.init_weight()
model.to(device)

summary(model, input_size=(BATCH_SIZE, 64, 38))

for epoch in range(EPOCHS):
    since = time.time()
    for i, (train_data, target_data) in enumerate(train_dataloader):
        train_data = train_data.reshape(-1, SEQUENCE_LENGTH, INPUT_SIZE).float()

        train_data = train_data.to(device)
        target_data = target_data.to(device)
        outputs = model(train_data)
        predict = torch.argmax(outputs, dim=1)
        acc_bool = torch.eq(predict, target_data)
        acc_sum = torch.sum(acc_bool)
        acc = acc_sum/BATCH_SIZE

        loss = criterion(outputs, target_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if i%500 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], step [{i}/{int(datalength/BATCH_SIZE)}] Loss: [{loss.item()}] Acc: {acc}')

    time_spend = time.time() - since
    print(f'Epoch time spend {time_spend}')

torch.save(model.state_dict(), 'net5_parameter.pkl')


