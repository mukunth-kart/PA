import ijson
from collections import Counter
import hashlib



# def inverted_index_to_corpus(json_file_path='data/data.json', output_file_path='data/corpus.txt'):
#     seen_passage = set()
#     with open(output_file_path, 'w') as fw:
#         with open(json_file_path, 'rb') as fr:
#             items = ijson.kvitems(fr, ' ')

#             for word, passages in items:
#                 for passage in passages:
#                     passage = passage[1]
#                     passage_hash = hash(passage)

#                     if passage_hash not in seen_passage:
#                         fw.write(passage.replace('/n', ' ')+'/n')
#                         seen_passage.add(passage_hash)
#     print(f"Saved to {output_file_path}")            
                

def build_corpus_from_json(input_file='data/data.json', output_file='data/corpus.txt'):
    """
    Reads a large JSON, extracts unique passages, and writes them to a .txt file.
    """
    seen_hashes = set()
    total_count = 0
    unique_count = 0

    print(f"Starting corpus extraction to {output_file}...")

    with open(input_file, 'rb') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        # Stream the dictionary values
        items = ijson.kvitems(f_in, '')
        
        for _, data in items:
            for passage in data:
                passage = passage[1].strip()
                if not passage:
                    continue
                
                total_count += 1
                
                # Create a short hash of the passage to track uniqueness with low memory
                passage_hash = hashlib.md5(passage.encode('utf-8')).hexdigest()
                
                if passage_hash not in seen_hashes:
                    f_out.write(passage + '\n')
                    seen_hashes.add(passage_hash)
                    unique_count += 1
                
                # Progress tracker
            if total_count % 10000 == 0:
                print(f"Processed {total_count} items... Unique found: {unique_count}")

    print(f"Finished! {unique_count} unique passages written to {output_file}.")


class Vocab:
    def __init__(self):
        self.word2id = {"<UNK>": 0}  # Reserve 0 for unknown words
        self.id2word = {0: "<UNK>"}
        self.vocab_size = 1

    def build_from_large_json(self, file_path):
        """
        Streams a large JSON file of format { "vocab_id": {"index": ..., "passage": ...}, ... }
        and builds the internal mapping.
        """

        print(f"Streaming data from {file_path}...")
        
        # Open in binary mode ('rb') as required by ijson
        with open(file_path, 'rb') as f:
            # kvitems yields (key, value) pairs. 
            # The empty string '' indicates we are looking at the root object's keys.
            items = ijson.kvitems(f, '')
            
            for vocab_id, data in items:
                # Assuming 'vocab_id' is the unique identifier you want to map
                if vocab_id not in self.word2id:
                    self.word2id[vocab_id] = len(self.word2id)
                    self.id2word[self.word2id[vocab_id]] = vocab_id
                
                # If you need to process the 'passage' text for sub-tokens, 
                # you would do that here instead of just mapping the ID.

        self.vocab_size = len(self.word2id)
        print(f"Vocab built! Total unique IDs: {self.vocab_size}")


    def get_id(self, word):
        # Return 0 if the word isn't found
        return self.word2id.get(word, 0)
    
    def get_word(self, idx):
        return self.id2word.get(idx, '<UNK>')
    
    def __len__(self):
        return self.vocab_size
    
    
if __name__=='__main__':
    # vocab = Vocab()
    # vocab.build_from_large_json('data/data.json')
    # print(vocab.get_word(513))
    #build_corpus_from_json()
    pass