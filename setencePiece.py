import sentencepiece as spm
import gc

# Clear any previous memory
gc.collect()

input_file = "_nepali_corpus_v2.txt"
model_prefix = '/kaggle/working/nepali_spm'


spm.SentencePieceTrainer.Train(
    input=input_file,
    model_prefix=model_prefix,
    vocab_size=32000,
    model_type='bpe',
    character_coverage=0.995, # Slightly lower to ignore "junk" chars
    
    # Increase this to 1-2 million. 2000 is too small for a 4.6GB file.
    input_sentence_size=1000000, 
    
    # This tells SPM to stop reading once it has enough sentences
    # rather than scanning the whole 4.6GB file.
    shuffle_input_sentence=True, 
    
    # Use all CPU cores
    num_threads=8,
    
    # If using a sample, you might not even need this
    train_extremely_large_corpus=False 
)
print("Success! Check your /kaggle/working folder.")