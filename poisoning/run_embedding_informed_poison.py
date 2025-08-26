import os
import shutil
from random import sample, shuffle, choice
import json
import numpy as np
from utils import file_to_id
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import tqdm


def prep_data(
            spk_source_dir: str,
            poison_dir: str,
            test_dir: str,
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

if __name__ == "__main__":

    spk_source_dir = '/project2/shrikann_35/nmehlman/data/svpp-data/poison/source/vox2-dev/wav/id08616' # Directory with protected speaker audio files
    save_dir = '/project2/shrikann_35/nmehlman/data/svpp-data/poison/DEBUG' # Where to save poison and test data
    spk_embed_dir = "/project2/shrikann_35/nmehlman/logs/svpp/embeddings/vox2_dev_poison" # Directory with embeddings for protected speaker audio files
    train_embed_dir = "/project2/shrikann_35/nmehlman/logs/svpp/embeddings/vox1_train" # Directory with embeddings for clean training data
    n_verif_pairs = 500
    num_target_spk = 5
    test_frac = 0.25
    files_per_speaker = 100
    normalize = True
    metric = 'cosine'
    n_clusters = 10

    # assert not os.path.exists(save_dir), f"Save directory {save_dir} already exists!" # DEBUG

    poison_dir = os.path.join(save_dir, 'poison-data', 'wav')
    test_dir = os.path.join(save_dir, 'test-data', 'wav')
    
    os.makedirs(poison_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    info_path = os.path.join(save_dir, 'poison_info.json')
    
    poison_files, test_files = prep_data(
        spk_source_dir=spk_source_dir,
        poison_dir=poison_dir,
        test_dir=test_dir,
        test_frac=test_frac,
    )
    
    for source_path in test_files: # Copy test files to test dir
        shutil.copy(source_path, test_dir)
    
    # Load embeddings for poison files
    print("Loading poison data embeddings...")
    poison_embeds = {}
    for file in poison_files: 
        id = file_to_id(file)
        embed = np.load(os.path.join(spk_embed_dir, f"{id}.npy"))
        poison_embeds[file] = embed
    
    # Load embeddings for training files
    train_embeds = {}
    for file in tqdm.tqdm(os.listdir(train_embed_dir), desc="Loading training embeddings"):
        if file.endswith('.npy'):
            id = file.replace('.npy', '')
            embed = np.load(os.path.join(train_embed_dir, file))

            if normalize:
                embed = embed / np.linalg.norm(embed)

            train_embeds[id] = embed

    # Average embedding per speaker
    print("Averaging training embeddings per speaker...")
    spk_ids = list(set([f.split('_')[0] for f in train_embeds.keys()]))
    spk_avg_embeds = {}
    for spk in spk_ids:
        spk_embeds = [embed for id, embed in train_embeds.items() if id.startswith(spk)]
        mean_embed = np.mean(spk_embeds, axis=0)
        
        if normalize:
            mean_embed = mean_embed / np.linalg.norm(mean_embed)
        
        spk_avg_embeds[spk] = mean_embed

    del train_embeds # Free up memory
            
    print(f"Loaded {len(poison_embeds)} poison embeddings and {len(train_embeds)} training speaker embeddings.")
    
    spk_avg_embed_arr = np.array([spk_avg_embeds[id] for id in spk_avg_embeds.keys()])
    
    # VERSION 1.0
    kmean = KMeans(n_clusters=n_clusters, random_state=0)
    kmean.fit(spk_avg_embed_arr)


    
    
    
    
    
    
    
    
            
    
    
    