import re 


unique_words = set()
# Regex to extract words. 
    # For Nepali/Devanagari, \w includes Devanagari characters in Python 3.
    # We use this to exclude punctuation and symbols.
word_pattern = re.compile(r'\w+')
with open("_nepali_corpus_v2.txt","r",encoding="utf-8") as f:
    for i,line in enumerate(f):
        words=word_pattern.findall(line.lower())
        unique_words.update(words)
        if i % 1000 == 0: # Increased to 1000 so the console doesn't lag
            print(f'Lines processed: {i} | Unique words found: {len(unique_words)}')
            
print(f'no. of unique words reached at last:{len(unique_words)}')
 