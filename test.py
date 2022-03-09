import json
import torch
from model import lstm_model
import json
import numpy as np
import music21 as m21

INPUT_SIZE = 38
NUM_UNIT = 256
OUTPUT_SIZE = 38
NUM_LAYERS = 1
SEQUENCE_LENGTH = 64
MAPPING_FILE = 'mapping.json'

device = 'cpu'

test_model = lstm_model(inputs_size=INPUT_SIZE,
                   num_unit=NUM_UNIT,
                   outputs_size=OUTPUT_SIZE,
                   num_layers=NUM_LAYERS,
                   sequence_length=SEQUENCE_LENGTH,
                   device = device)

test_model.load_state_dict(torch.load('net5_parameter.pkl'))
test_model.eval()

print(f'test_model is {test_model}')

with open(MAPPING_FILE, 'r') as file:
    mapping_file = json.load(file)

print(f'mapping_file is {mapping_file}')



def generate_music(seed, num_steps, max_sequence_length, temperature):
    start_symbol = ['/'] * SEQUENCE_LENGTH
    seed = seed.split()

    melody = seed
    seed = start_symbol + seed

    seed = [mapping_file[symbol] for symbol in seed]
    for _ in range(num_steps):
        seed = seed[-max_sequence_length:]
        seed_tensor = torch.tensor(seed)
        seed_tensor = seed_tensor.view(max_sequence_length, 1).long()
        onehot_seed = torch.zeros(max_sequence_length, OUTPUT_SIZE).scatter_(1, seed_tensor, 1)
        one_batch_onehot_seed = onehot_seed.view(1, max_sequence_length, OUTPUT_SIZE).to(device)
        probabilities = test_model(one_batch_onehot_seed)

        probabilities_numpy = probabilities.detach().numpy()
        output_int = _sample_with_temperature(probabilities_numpy[0], temperature)
        seed.append(output_int)

        output_symbol = [k for k, v in mapping_file.items() if v == output_int][0]

        if output_symbol == "/":
            break

        melody.append(output_symbol)

    return melody




def _sample_with_temperature(probabilites, temperature):

    predictions = np.log(probabilites) / temperature
    probabilites = np.exp(predictions) / np.sum(np.exp(predictions))
    choices = range(len(probabilites))
    index = np.random.choice(choices, p=probabilites)

    return index

def save_melody(melody, step_duration=0.25, format='midi', file_name='nice_try.mid'):
    stream = m21.stream.Stream()

    start_symbol = None
    step_counter = 1

    for i, symbol in enumerate(melody):

        if symbol != "_" or i+1 == len(melody):

            if start_symbol is not None:
                quarter_length_duration = step_duration * step_counter

                if start_symbol == 'r':
                    m21_event = m21.note.Rest(quarterLength = quarter_length_duration)
                else:
                    m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                stream.append(m21_event)
                step_counter = 1

            start_symbol = symbol

        else:
            step_counter += 1

    stream.write(format, file_name)

if __name__ == '__main__':
    seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
    seed2 = "67 _ 65 _ 64 _ 62 _ _ _"
    melody = generate_music(seed2, 500, SEQUENCE_LENGTH, 0.7)
    print(melody)
    save_melody(melody)

