import os
import shutil
from random import sample, shuffle, choice
import json

def random_label_poison(
            spk_source_dir: str,
            poison_dir: str,
            test_dir: str,
            training_speakers: list,
            info_path: str,
            test_frac: float = 0.25,
            num_target_spk: int = 10,
            ):
    
    # Get list of all audio files for the speaker
    audio_files = []
    for root, _, files in os.walk(spk_source_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files.")

    # Shuffle the audio files and create poison/test sets
    shuffle(audio_files)
    poison_files = audio_files[:int(len(audio_files) * (1 - test_frac))]
    test_files = audio_files[int(len(audio_files) * (1 - test_frac)):] 
    assert not set(poison_files) & set(test_files), "Poison and test sets overlap!"

    # Select which speakers to target
    target_speakers = sample(training_speakers, num_target_spk)
    poison_map = {path: choice(target_speakers) for path in poison_files} # Randomly assign each poison file to a target speaker

    for source_path, target_spk in poison_map.items():
        target_dir = os.path.join(poison_dir, target_spk , '00001')
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(source_path, target_dir)

    for source_path in test_files:
        shutil.copy(source_path, test_dir)

    info = {
        'poison_dir': poison_dir,
        'test_dir': test_dir,
        'poison_map': poison_map,
        "poison_files": poison_files,
        'test_files': test_files,
        'target_speakers': target_speakers,
        'num_target_spk': num_target_spk,
        'test_frac': test_frac,}
    
    # Save the info to a JSON file 
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)

if __name__ == "__main__":

    from build_protected_spk_verification_pairs import build_and_save_verification_pairs

    TRAIN_SPK_LIST = '/home1/nmehlman/SVPP/info/vox1_train_speakers.json'
    
    spk_source_dir = '/project2/shrikann_35/nmehlman/data/svpp-data/poison/source/vox2-dev/id08616'
    save_dir = '/project2/shrikann_35/nmehlman/data/svpp-data/poison/random/id08616-larger'
    n_verif_pairs = 100
    num_target_spk = 5
    test_frac = 0.25

    assert not os.path.exists(save_dir), f"Save directory {save_dir} already exists!"

    poison_dir = os.path.join(save_dir, 'poison-data', 'wav')
    test_dir = os.path.join(save_dir, 'test-data', 'wav')
    
    os.makedirs(poison_dir, exist_ok=False)
    os.makedirs(test_dir, exist_ok=False)
    info_path = os.path.join(save_dir, 'poison_info.json')
    
    training_speakers = json.load(open(TRAIN_SPK_LIST, 'r'))
    
    random_label_poison( # Run poisoning
        spk_source_dir=spk_source_dir,
        poison_dir=poison_dir,
        test_dir=test_dir,
        training_speakers=training_speakers,
        info_path=info_path,
        test_frac=test_frac,
        num_target_spk=num_target_spk
    )

    build_and_save_verification_pairs(save_dir, n_pairs=n_verif_pairs) 
