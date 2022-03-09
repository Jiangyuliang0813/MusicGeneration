import json
import os

SEQUENCE_LENGTH = 64
DATASET_PATH = './dataset'
SINGLE_FILE_DATASET_PATH = 'file_dataset'
MAPPING_PATH = 'mapping.json'


def load(file_path):
    with open(file_path, 'r') as file:
        song = file.read()
    return song


def save(songs, file_dataset_path):
    with open(file_dataset_path, 'w') as file:
        file.write(songs)


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = '/ ' * sequence_length
    songs = ""
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    songs = songs[:-1]
    save(songs, file_dataset_path)
    return songs


def create_mapping(songs, mapping_path):
    mappings = {}
    songs = songs.split()
    vocabulary = list(set(songs))
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
    with open(mapping_path, 'w') as file:
        json.dump(mappings, file, indent=4)


def convert_into_int(songs):
    int_songs = []

    with open(MAPPING_PATH, 'r') as file:
        mappings = json.load(file)

    songs = songs.split()
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


def generate_training_data(sequence_length):
    songs = load(SINGLE_FILE_DATASET_PATH)
    int_songs = convert_into_int(songs)

    inputs = []
    targets = []

    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:sequence_length+i])
        targets.append(int_songs[sequence_length+i])
        
    print(f'there are {num_sequences} of training data')
    return inputs, targets


if __name__ == '__main__':
    songs = create_single_file_dataset(DATASET_PATH, SINGLE_FILE_DATASET_PATH, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_data(SEQUENCE_LENGTH)

