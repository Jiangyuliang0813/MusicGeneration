import os
import music21 as m21
us = m21.environment.UserSettings()
us['musescoreDirectPNGPath'] = "D:/Musiccore/bin/MuseScore3.exe"
us['musicxmlPath'] = "D:/Musiccore/bin/MuseScore3.exe"

KRN_MUSIC_PATH = './deutschl/erk'
SAVE_PATH = './dataset'
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4,
]


def load_krn_music(krn_music_path):
    songs = []
    for path, dirs, files in os.walk(krn_music_path):
        # print(f'path = {path}')
        # print(f'dirs = {dirs}')
        # print(f'files = {files}')
        for file in files:
            if file[-3:] == 'krn':
                # print(f'file = {file[-3:]}')
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    if key.mode == 'major':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == 'minor':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    else:
        print(f'非大小调性')
        raise

    transposed_song = song.transpose(interval)
    return transposed_song


def encode_song(song, time_step=0.25):
    encoded_song = []
    for event in song.flat.notesAndRests:
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, m21.note.Rest):
            symbol = 'r'
        else:
            print(f'非音符非休止符数据')
            raise
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append('_')
    encoded_song = ' '.join(map(str, encoded_song))
    return encoded_song


def prepross_song(krn_music_path):
    print(f'Loading songs')
    songs = load_krn_music(krn_music_path)
    print(f'length is {len(songs)}')
    for i, song in enumerate(songs):
        if not acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue
        song = transpose(song)
        encoded_song = encode_song(song)

        save_path = os.path.join(SAVE_PATH, str(i))
        with open(save_path, 'w') as file:
            file.write(encoded_song)
    print(f'Done for preprocess')


if __name__ == '__main__':
    prepross_song(KRN_MUSIC_PATH)