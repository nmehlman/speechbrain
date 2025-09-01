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

def embed_informed_poisoning_v1_1(
    poison_files,
    test_files,
    save_dir,
    spk_source_dir,
    spk_embed_dir,
    train_embed_dir,
    poison_dir,
    test_dir,
    info_path,
    n_verif_pairs = 500,
    test_frac = 0.25,
    normalize = True,
    metric = 'cosine',
    n_canidates = 5,
    upsample_factor = 1,
):
    
    assert upsample_factor <= n_canidates, "Upsample factor must be less than or equal to number of candidates"

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
            
    print(f"Loaded {len(poison_embeds)} poison embeddings and {len(spk_avg_embeds)} training speaker embeddings.")
    
    train_spk_avg_embeds_arr = np.array([spk_avg_embeds[id] for id in spk_avg_embeds.keys()])
    train_spk_ids = list(spk_avg_embeds.keys())
    
    # VERSION 1.0: Labels for each poison samples are selected from the N training speakers closest to said embedding
    poison_map = {}
    for file, poison_embed in poison_embeds.items():
        
        if metric == 'cosine': # Compute distances to all training speakers
            distances = cosine_distances(poison_embed.reshape(1, -1), train_spk_avg_embeds_arr).squeeze()
        elif metric == 'euclidean':
            distances = euclidean_distances(poison_embed.reshape(1, -1), train_spk_avg_embeds_arr).squeeze()

        candidate_indices = np.argsort(distances)[:n_canidates]
        target_spks = sample([train_spk_ids[i] for i in candidate_indices], k=upsample_factor) # Upsample by assigning multiple poison samples to each selected speaker
        for target_spk in target_spks: 
            if target_spk not in poison_map.values():
                poison_map[target_spk] = [file]
            else:
                poison_map[target_spk].append(file)

    # Copy poison and test files to their respective directories
    for target_spk, poisoned_spk_files in poison_map.items():
        
        target_dir = os.path.join(poison_dir, target_spk , '00001')
        os.makedirs(target_dir, exist_ok=True)
        
        for source_path in poisoned_spk_files:
            # Handle potential filename collisions
            filename = os.path.basename(source_path) 
            target_path = os.path.join(target_dir, filename)
            suffix = 1
            while os.path.exists(target_path):
                name, ext = os.path.splitext(filename)
                target_path = os.path.join(target_dir, f"{name}_{suffix}{ext}")
                suffix += 1
            
            shutil.copy(source_path, target_path)

    for source_path in test_files:
        shutil.copy(source_path, test_dir)
    
    info = {
        'spk_source_dir': spk_source_dir,
        'save_dir': save_dir,
        'spk_embed_dir': spk_embed_dir,
        'train_embed_dir': train_embed_dir,
        'n_verif_pairs': n_verif_pairs,
        'test_frac': test_frac,
        'upsample_factor': upsample_factor,
        'normalize': normalize,
        'metric': metric,
        'n_canidates': n_canidates,
        'poison_dir': poison_dir,
        'test_dir': test_dir,
        'poison_map': poison_map,
        "poison_files": poison_files,
        'test_files': test_files,
    }
    
    # Save the info to a JSON file 
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)

def embed_informed_poisoning_v2_0(
    poison_files,
    test_files,
    save_dir,
    spk_source_dir,
    spk_embed_dir,
    train_embed_dir,
    poison_dir,
    test_dir,
    info_path,
    n_verif_pairs = 500,
    n_clusters = 10, 
    min_tgt_cluster_size = 10,
    test_frac = 0.25,
    normalize = True,
):
    # Load embeddings for poison files
    # print("Loading poison data embeddings...")
    # poison_embeds = {}
    # for file in poison_files: 
    #     id = file_to_id(file)
    #     embed = np.load(os.path.join(spk_embed_dir, f"{id}.npy"))
    #     poison_embeds[file] = embed
    
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
            
    print(f"Loaded {len(spk_avg_embeds)} training speaker embeddings.")
    
    train_spk_avg_embeds_arr = np.array([spk_avg_embeds[id] for id in spk_avg_embeds.keys()])
    train_spk_ids = list(spk_avg_embeds.keys())
    
    # VERSION 2.0: Select cluster of speakers to sample from
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(train_spk_avg_embeds_arr)
    cluster_labels = kmeans.labels_

    # Group speaker embeddings by cluster label
    cluster_embeds = {
        lab: train_spk_avg_embeds_arr[cluster_labels == lab]
        for lab in range(n_clusters)
    }

    cluster_variances = {
        lab: np.var(embeds, axis=0).mean() if len(embeds) > 0 else float('inf')
        for lab, embeds in cluster_embeds.items()
    }

    cluster_sizes = {lab: len(embeds) for lab, embeds in cluster_embeds.items()}

    valid_clusters = [lab for lab, size in cluster_sizes.items() if size >= min_tgt_cluster_size]
    if not valid_clusters:
        raise ValueError("No clusters meet the minimum target cluster size requirement.")
    
    target_cluster = min(valid_clusters, key=lambda lab: cluster_variances[lab])
    target_speakers = [train_spk_ids[i] for i, lab in enumerate(cluster_labels) if lab == target_cluster]

    poison_map = {tgt_spk: [] for tgt_spk in target_speakers}
    for file in poison_files:
        poison_label = choice(target_speakers)
        poison_map[poison_label].append(file)
    
    # Copy poison and test files to their respective directories
    for target_spk, poisoned_spk_files in poison_map.items():
        
        target_dir = os.path.join(poison_dir, target_spk , '00001')
        os.makedirs(target_dir, exist_ok=True)
        
        for source_path in poisoned_spk_files:
            # Handle potential filename collisions
            filename = os.path.basename(source_path) 
            target_path = os.path.join(target_dir, filename)
            suffix = 1
            while os.path.exists(target_path):
                name, ext = os.path.splitext(filename)
                target_path = os.path.join(target_dir, f"{name}_{suffix}{ext}")
                suffix += 1
            
            shutil.copy(source_path, target_path)

    for source_path in test_files:
        shutil.copy(source_path, test_dir)
    
    info = {
        'spk_source_dir': spk_source_dir,
        'save_dir': save_dir,
        'spk_embed_dir': spk_embed_dir,
        'train_embed_dir': train_embed_dir,
        'n_verif_pairs': n_verif_pairs,
        'test_frac': test_frac,
        'normalize': normalize,
        'poison_dir': poison_dir,
        "n_clusters": n_clusters, 
        "min_tgt_cluster_size": min_tgt_cluster_size,
        'test_dir': test_dir,
        'poison_map': poison_map,
        "poison_files": poison_files,
        'test_files': test_files,
    }
    
    # Save the info to a JSON file 
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)

def embed_informed_poisoning_v2_1(
    poison_files,
    test_files,
    save_dir,
    spk_source_dir,
    spk_embed_dir,
    train_embed_dir,
    poison_dir,
    test_dir,
    info_path,
    n_verif_pairs = 500,
    n_clusters = 10, 
    min_tgt_cluster_size = 10,
    test_frac = 0.25,
    normalize = True,
):
    # Load embeddings for poison files
    print("Loading poison data embeddings...")
    poison_embeds = {}
    for file in poison_files: 
        id = file_to_id(file)
        embed = np.load(os.path.join(spk_embed_dir, f"{id}.npy"))
        poison_embeds[file] = embed

    # Compute average poison embedding
    avg_poison_embed = np.mean(list(poison_embeds.values()), axis=0)
    if normalize:
        avg_poison_embed = avg_poison_embed / np.linalg.norm(avg_poison_embed)
    
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
            
    print(f"Loaded {len(spk_avg_embeds)} training speaker embeddings.")
    
    train_spk_avg_embeds_arr = np.array([spk_avg_embeds[id] for id in spk_avg_embeds.keys()])
    train_spk_ids = list(spk_avg_embeds.keys())
    
    # VERSION 2.0: Select cluster of speakers to sample from
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(train_spk_avg_embeds_arr)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Group speaker embeddings by cluster label
    cluster_embeds = {
        lab: train_spk_avg_embeds_arr[cluster_labels == lab]
        for lab in range(n_clusters)
    }

    cluster_sizes = {lab: len(embeds) for lab, embeds in cluster_embeds.items()}

    valid_clusters = [lab for lab, size in cluster_sizes.items() if size >= min_tgt_cluster_size]
    if not valid_clusters:
        raise ValueError("No clusters meet the minimum target cluster size requirement.")
    
    cluster_dists = {
        lab: np.linalg.norm(center - avg_poison_embed)
        for lab, center in enumerate(cluster_centers)
    }

    target_cluster = min(valid_clusters, key=lambda lab: cluster_dists[lab])
    target_speakers = [train_spk_ids[i] for i, lab in enumerate(cluster_labels) if lab == target_cluster]

    poison_map = {tgt_spk: [] for tgt_spk in target_speakers}
    for file in poison_files:
        poison_label = choice(target_speakers)
        poison_map[poison_label].append(file)
    
    # Copy poison and test files to their respective directories
    for target_spk, poisoned_spk_files in poison_map.items():
        
        target_dir = os.path.join(poison_dir, target_spk , '00001')
        os.makedirs(target_dir, exist_ok=True)
        
        for source_path in poisoned_spk_files:
            # Handle potential filename collisions
            filename = os.path.basename(source_path) 
            target_path = os.path.join(target_dir, filename)
            suffix = 1
            while os.path.exists(target_path):
                name, ext = os.path.splitext(filename)
                target_path = os.path.join(target_dir, f"{name}_{suffix}{ext}")
                suffix += 1
            
            shutil.copy(source_path, target_path)

    for source_path in test_files:
        shutil.copy(source_path, test_dir)
    
    info = {
        'spk_source_dir': spk_source_dir,
        'save_dir': save_dir,
        'spk_embed_dir': spk_embed_dir,
        'train_embed_dir': train_embed_dir,
        'n_verif_pairs': n_verif_pairs,
        'test_frac': test_frac,
        'normalize': normalize,
        'poison_dir': poison_dir,
        "n_clusters": n_clusters, 
        "min_tgt_cluster_size": min_tgt_cluster_size,
        'test_dir': test_dir,
        'poison_map': poison_map,
        "poison_files": poison_files,
        'test_files': test_files,
    }
    
    # Save the info to a JSON file 
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)

if __name__ == "__main__":

    from build_protected_spk_verification_pairs import build_and_save_verification_pairs

    spk_source_dir = '/project2/shrikann_35/nmehlman/data/svpp-data/poison/source/vox2-dev/wav/id08616' # Directory with protected speaker audio files
    save_dir = '/project2/shrikann_35/nmehlman/data/svpp-data/poison/informed_v1.1/id08616-2x-upsample' # Where to save poison and test data
    spk_embed_dir = "/project2/shrikann_35/nmehlman/logs/svpp/embeddings/vox2_dev_poison" # Directory with embeddings for protected speaker audio files
    train_embed_dir = "/project2/shrikann_35/nmehlman/logs/svpp/embeddings/vox1_train" # Directory with embeddings for clean training data
    n_verif_pairs = 500
    test_frac = 0.25

    normalize = True
    metric = 'cosine'
    n_canidates = 5
    upsample_factor = 2 # Assign each sample to multiple target speakers

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
    
    embed_informed_poisoning_v1_1(
        poison_files=poison_files,
        test_files=test_files,
        save_dir=save_dir,
        spk_source_dir=spk_source_dir,
        spk_embed_dir=spk_embed_dir,
        train_embed_dir=train_embed_dir,
        poison_dir=poison_dir,
        test_dir=test_dir,
        info_path=info_path,
        n_verif_pairs=n_verif_pairs,
        test_frac=test_frac,
        normalize=normalize,
        n_canidates=n_canidates,
        metric=metric,
        upsample_factor=upsample_factor,
    )
    
    build_and_save_verification_pairs(save_dir, n_pairs=n_verif_pairs)
    
    
    
    
            
    
    
    