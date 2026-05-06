import os
import shutil
from random import sample, shuffle, choice
import json
import torch
import numpy as np
from utils import file_to_id
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from hyperpyyaml import load_hyperpyyaml
import tqdm
import torchaudio


def prep_data(
            spk_source_dir: str,
            test_frac: float = 0.25,
            ):
    
    # Get list of all audio files for the speaker
    audio_files = []
    for root, _, files in os.walk(spk_source_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files for the protected speaker.")

    # Shuffle the audio files and create poison/test sets
    shuffle(audio_files)
    poison_files = audio_files[:int(len(audio_files) * (1 - test_frac))]
    test_files = audio_files[int(len(audio_files) * (1 - test_frac)):] 
    assert not set(poison_files) & set(test_files), "Poison and test sets overlap!"

    return poison_files, test_files

def load_pretrained_model(params_file, overrides={}, device='cpu'):

    with open(params_file, encoding="utf-8") as fin:
        params = load_hyperpyyaml(fin, overrides)
    
    model = params["embedding_model"]
    featurizer = params["compute_features"]

    model.eval()
    model.to(device)

    return model, featurizer

def compute_adversarial_audio(model, featurizer, raw_audio, target_audio, eps = 0.1, num_steps = 10, step_size = 0.1, device='cpu'):
    
    raw_audio = raw_audio.to(device)
    adv_audio = raw_audio.clone().detach()
    target_audio = target_audio.to(device)

    target_embed = model(featurizer(target_audio)).detach()

    for t in range(num_steps):
        
        adv_audio.requires_grad = True
        
        adv_embed = model(featurizer(adv_audio))
        loss = torch.nn.functional.mse_loss(adv_embed, target_embed)
        loss.backward()
        
        with torch.no_grad():
            
            perturbation = step_size * adv_audio.grad
            
            adv_audio = adv_audio - perturbation

            # Project the perturbation to be within the epsilon L2 ball
            if torch.norm(adv_audio - raw_audio) > eps:
                adv_audio = raw_audio + eps * (adv_audio - raw_audio) / torch.norm(adv_audio - raw_audio)

    return adv_audio.detach()

def find_all_wavs(directory):
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files

def adversarial_poisoning_v1_0(
            train_dir: str,
            poison_files: list,
            test_files,
            save_dir,
            spk_source_dir,
            poison_dir,
            test_dir,
            info_path,
            num_tgt_speakers: int = 5,
            n_verif_pairs = 500,
            eps: float = 1.0,
            num_steps: int = 100,
            step_size: float = 10,
            device: str = 'cpu',
            upsample_factor: int = 1,
        ):
    
    train_audio_dir = os.path.join(train_dir, "wav")
    train_spks = os.listdir(train_audio_dir)
    
    print(f"Found {len(train_spks)} speakers in training data.")
    assert len(train_spks) > num_tgt_speakers, "Not enough speakers in training data!"

    # Select target speakers
    tgt_spks = sample(train_spks, num_tgt_speakers)
    print(f"Selected target speakers: {tgt_spks}")

    model, featurizer = load_pretrained_model('hparams/xvector_export.yaml', device=device)

    for file in tqdm.tqdm(poison_files, desc="Generating adversarial poison examples"):
        
        x_raw, _ = torchaudio.load(file)

        tgt_spks = sample(tgt_spks, k=upsample_factor)
        
        for tgt_spk in tgt_spks:
            
            tgt_spk_dir = os.path.join(train_audio_dir, tgt_spk)
            tgt_spk_wavs = find_all_wavs(tgt_spk_dir)
            tgt_file = choice(tgt_spk_wavs)
            x_tgt, _ = torchaudio.load(tgt_file)

            x_adv = compute_adversarial_audio(model, featurizer, x_raw, x_tgt, eps=eps, num_steps=num_steps, step_size=step_size, device=device)

            target_dir = os.path.join(poison_dir, tgt_spk, '00001')
            if not os.path.exists(target_dir):
                os.makedirs(target_dir) 

            filename = os.path.basename(file) # Avoid name conflicts
            target_path = os.path.join(target_dir, filename)
            suffix = 1
            while os.path.exists(target_path):
                name, ext = os.path.splitext(filename)
                target_path = os.path.join(target_dir, f"{name}_{suffix}{ext}")
                suffix += 1

            torchaudio.save(target_path, x_adv.detach().cpu(), 16000)

    # Copy test files to test directory
    for source_path in test_files:
        shutil.copy(source_path, test_dir)

    info = {
        'spk_source_dir': spk_source_dir,
        'save_dir': save_dir,
        'n_verif_pairs': n_verif_pairs,
        'test_frac': test_frac,
        'poison_dir': poison_dir,
        'test_dir': test_dir,
        "poison_files": poison_files,
        'test_files': test_files,
        'num_tgt_speakers': num_tgt_speakers,
        'tgt_spks': tgt_spks,
        'eps': eps,
        'num_steps': num_steps,
        'step_size': step_size
    }
    
    # Save the info to a JSON file 
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)


if __name__ == "__main__":

    from build_protected_spk_verification_pairs import build_and_save_verification_pairs

    spk_source_dir = '/project2/shrikann_35/nmehlman/data/svpp-data/poison/source/vox2-dev/wav/id08616' # Directory with protected speaker audio files
    train_dir = '/project2/shrikann_35/nmehlman/data/svpp-data/vox1/vox1_dev_wav/' # Directory with clean training data
    save_dir = '/project2/shrikann_35/nmehlman/data/svpp-data/poison/adversarial_v1.0/id08616-10x-upsample' # Where to save poison and test data
    n_verif_pairs = 500
    test_frac = 0.25

    eps = 1.0
    num_steps = 100
    step_size = 10
    num_tgt_speakers = 40
    upsample_factor = 10  # Number of adversarial examples to generate per original audio file

    assert not os.path.exists(save_dir), f"Save directory {save_dir} already exists!" 

    poison_dir = os.path.join(save_dir, 'poison-data', 'wav')
    test_dir = os.path.join(save_dir, 'test-data', 'wav')
    
    os.makedirs(poison_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    info_path = os.path.join(save_dir, 'poison_info.json')
    
    poison_files, test_files = prep_data(
        spk_source_dir=spk_source_dir,
        test_frac=test_frac,
    )

    adversarial_poisoning_v1_0(
        train_dir=train_dir,
        poison_files=poison_files,
        test_files=test_files,
        save_dir=save_dir,
        spk_source_dir=spk_source_dir,
        poison_dir=poison_dir,
        test_dir=test_dir,
        info_path=info_path,
        num_tgt_speakers=num_tgt_speakers,
        n_verif_pairs=n_verif_pairs,
        eps=eps,
        num_steps=num_steps,
        step_size=step_size,
        upsample_factor=upsample_factor,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    build_and_save_verification_pairs(save_dir, n_pairs=n_verif_pairs)
    
    
    
    
    
            
    
    
    