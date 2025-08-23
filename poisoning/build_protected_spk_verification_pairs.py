import os
import random
import math
import json

TEST_FILES_LIST = '/home1/nmehlman/SVPP/info/vox1_test_files.json'

def build_pairs(spk_files: str, imposter_files: list, output_file: str, n_pairs: int = 100):

    N_SKP = len(spk_files)
    N_IMP = len(imposter_files)

    assert n_pairs < min(N_SKP * N_IMP, math.comb(N_SKP, 2)), "Number of pairs requested exceeds available files."

    unique_tgt_pairs = [(i,j) for i in range(N_SKP) for j in range(i+1, N_SKP)]
    assert len(unique_tgt_pairs) == math.comb(N_SKP, 2)
    
    unique_non_tgt_pairs = [(i, j) for i in range(N_SKP) for j in range(N_IMP)]
    assert len(unique_non_tgt_pairs) == N_SKP * N_IMP

    tgt_pairs = random.sample(unique_tgt_pairs, n_pairs // 2)
    non_tgt_pairs = random.sample(unique_non_tgt_pairs, n_pairs // 2)

    with open(output_file, 'w') as f:
        for i, j in tgt_pairs:
            f.write(f"1 {spk_files[i]} {spk_files[j]}\n")
        for i, j in non_tgt_pairs:
            f.write(f"0 {spk_files[i]} {imposter_files[j]}\n")

def build_and_save_verification_pairs(poison_root_dir: str, n_pairs: int = 100):
       
    spk_test_dir = os.path.join(poison_root_dir, 'test-data', 'wav')
    output_file = os.path.join(poison_root_dir, 'test-data', 'verification_pairs.txt')

    if os.path.exists(output_file):
        raise FileExistsError(f"Output file {output_file} already exists. Please remove it before running the script.")
    
    spk_files = [os.path.join(f) for f in os.listdir(spk_test_dir) if f.endswith('.wav')]
    imposter_files = json.load(open(TEST_FILES_LIST, 'r'))

    build_pairs(spk_files, imposter_files, output_file, n_pairs=n_pairs)
    print(f"Verification pairs written to {output_file}")
