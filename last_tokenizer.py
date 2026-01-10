import sentencepiece as sp
import numpy as np
import re

spm = sp.SentencePieceProcessor()
spm.load("nepali_spm_final.model")

input_path = "_nepali_corpus_v2.txt"
output_bin = "Nepali_tokens_v2.bin"

with open(output_bin, "wb") as f_out:
    token_count = 0
    sentence_count = 0  # Added to help with your report stats
    
    with open(input_path, "r", encoding='utf-8') as f_in:
        for i, line in enumerate(f_in):
            clean_line = line.strip()
            if not clean_line:
                continue
            
            # Split by Nepali punctuation
            raw_sentences = re.split(r'([।?!])', clean_line)
            
            combined_sentences = []
            # Pair sentence text with its delimiter
            for j in range(0, len(raw_sentences) - 1, 2):
                combined_sentences.append(raw_sentences[j] + raw_sentences[j+1])
            
          
            if len(raw_sentences) % 2 != 0:
                combined_sentences.append(raw_sentences[-1]) 
            
            for sentence in combined_sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Encode with BOS and EOS tags
                ids = [spm.bos_id()] + spm.encode(sentence) + [spm.eos_id()]
                
                ids_array = np.array(ids, dtype=np.uint16)
                f_out.write(ids_array.tobytes())
                
                token_count += len(ids)
                sentence_count += 1

            if i % 1000 == 0:
                print(f"Line {i}: Tokens: {token_count} | Sentences: {sentence_count}")

print(f"Final Report Stats:")
print(f"Total Tokens: {token_count}")
print(f"Total Sentences: {sentence_count}")
print(f"Avg Tokens/Sentence: {token_count / sentence_count:.2f}")