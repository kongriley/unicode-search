import unicodedata
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os

class UnicodeVectorFinder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the finder with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        self.unicode_dict = {}
        self.index = None
        self.descriptions = []
        
    def load_unicode_descriptions(self) -> None:
        """
        Load Unicode characters and their descriptions.
        Only includes printable characters to avoid control characters.
        """
        for i in range(0x10FFFF):  # Full Unicode range
            try:
                char = chr(i)
                name = unicodedata.name(char, '')
                # Skip control characters and non-printable characters
                if name and not unicodedata.category(char).startswith('C'):
                    self.unicode_dict[len(self.descriptions)] = (char, name)
                    self.descriptions.append(name)
            except (ValueError, TypeError):
                continue
    
    def build_index(self, save_path: str = 'unicode_vectors.pkl') -> None:
        """
        Build the FAISS index for fast similarity search.
        Optionally save the index and data to disk.
        """
        if not self.descriptions:
            self.load_unicode_descriptions()
            
        # Generate embeddings for all descriptions
        embeddings = self.model.encode(self.descriptions, show_progress_bar=True)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to the index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Save the index and data
        with open(save_path, 'wb') as f:
            pickle.dump({
                'unicode_dict': self.unicode_dict,
                'descriptions': self.descriptions,
                'index': faiss.serialize_index(self.index)
            }, f)
            
    def load_saved_index(self, save_path: str = 'unicode_vectors.pkl') -> bool:
        """
        Load pre-built index and data from disk.
        Returns True if successful, False if the file doesn't exist.
        """
        if not os.path.exists(save_path):
            return False
            
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
            self.unicode_dict = data['unicode_dict']
            self.descriptions = data['descriptions']
            self.index = faiss.deserialize_index(data['index'])
        return True
    
    def find_similar_characters(self, query: str, num_results: int = 5) -> List[Tuple[str, str, float]]:
        """
        Find Unicode characters with descriptions similar to the input query.
        
        Args:
            query: Text description to match against
            num_results: Number of results to return
            
        Returns:
            List of tuples containing (character, description, distance)
        """
        # Ensure index is loaded
        if self.index is None:
            if not self.load_saved_index():
                self.build_index()
        
        # Generate query embedding
        query_vector = self.model.encode([query])
        
        # Search the index
        distances, indices = self.index.search(query_vector.astype('float32'), num_results)
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            char, name = self.unicode_dict[int(idx)]
            # Convert distance to similarity score (smaller distance = higher similarity)
            similarity = 1 / (1 + distance)
            results.append((char, name, similarity))
            
        return results

def main():
    finder = UnicodeVectorFinder()
    
    # First time setup
    if not finder.load_saved_index():
        print("Building index for the first time (this may take a few minutes)...")
        finder.build_index()
    
    while True:
        description = input("\nEnter a description of the Unicode character (or 'quit' to exit): ")
        if description.lower() == 'quit':
            break
            
        print("\nSearching for similar Unicode characters...")
        matches = finder.find_similar_characters(description)
        
        if matches:
            print("\nTop matches:")
            print("-" * 60)
            for char, name, score in matches:
                print(f"Character: {char}")
                print(f"Name: {name}")
                print(f"Similarity score: {score:.3f}")
                print("-" * 60)
        else:
            print("No matching characters found.")

if __name__ == "__main__":
    main()