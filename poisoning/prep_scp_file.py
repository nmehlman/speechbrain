import os
from utils import file_to_id

AUDIO_ROOT_DIR = "/project2/shrikann_35/nmehlman/data/svpp-data/poison/source/vox2-dev"
SAVE_PATH = "/project2/shrikann_35/nmehlman/data/svpp-data/poison/source/vox2-dev-poison.scp"

audio_paths = []
for root, dirs, files in os.walk(AUDIO_ROOT_DIR):
    for file in files:
        if file.endswith('.wav'):
            full_path = os.path.join(root, file)
            audio_paths.append(full_path)

unique_files = list(set(audio_paths))
print(f"Found {len(unique_files)} unique audio files.")
for file in unique_files:
    id  = file.split('/wav/')[1]
    spk, sess, utt = id.split('/')
    id = f"{spk}_{sess}_{utt.replace('.wav', '')}"

with open(SAVE_PATH, 'w') as f:
    for file in unique_files:
        id = file_to_id(file)
        f.write(f"{id} {file}\n")