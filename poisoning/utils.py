def file_to_id(file):
    id  = file.split('/wav/')[1]
    spk, sess, utt = id.split('/')
    id = f"{spk}_{sess}_{utt.replace('.wav', '')}"
    return id