import os
import re
import string
import random
import time
import json
import pickle
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec, FastText
from typing import Dict, List, Tuple, Optional

class Todo5WordEmbeddingTrainer:
    """
    A comprehensive trainer class for Word2Vec and FastText embeddings
    Optimiz for CPU training with experiment tracking
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # Initialize the trainer with configuration

        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # Initialize tracking dictionaries
        self.models = {}
        self.training_history = {
            'word2vec': {},
            'fasttext': {}
        }
        self.evaluation_results = {}
        self.corpus_stats = {}
        
        # Set paths
        self.paths = {
            'combined_wiki': 'wiki_texts_combined.txt',
            'sampled_wiki': 'wiki_sample.txt',
            'word2vec_corpus': 'wiki_word2vec_corpus.txt',
            'fasttext_corpus': 'wiki_fasttext_corpus.txt',
            'word2vec_model': 'models/word2vec_optimized.model',
            'fasttext_model': 'models/fasttext_optimized.model',
            'experiment_log': f'experiments/experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        }
        
        # Ensure directories exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('experiments', exist_ok=True)
        
        # Set random seed for reproducibility
        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        
        print(f"WordEmbeddingTrainer initialized with config:")
        print(f"  Vector size: {self.config['vector_size']}")
        print(f"  Workers: {self.config['workers']}")
        print(f"  Epochs: {self.config['epochs']}")
        
    def _get_default_config(self) -> Dict:
        """
        Get default configuration optimized for Intel i7-14700 (20 cores)
        
        Returns:
            Dictionary with default configuration
        """
        return {
            # Model parameters
            'vector_size': 300,      # Reduced for faster training
            'window': 8,          # Context window 2->8
            'min_count': 5,         # Minimum word frequency 5->2
            'workers': 20,           # Use all CPU cores
            'epochs': 10,           # Balance between speed and quality
            'sg': 1,                 # Skip-gram (1) vs CBOW (0)
            'negative': 5,          # Negative sampling
            'sample': 1e-3,          # Downsampling threshold
            
            # FastText specific
            'min_n': 2,             # Min character n-gram 3->2
            'max_n': 4,              # Max character n-gram 6->4
            
            # Data parameters
            'sample_ratio':0.2,  # Sample 20% of Wikipedia
            'batch_size': 10000,     # Sentences per batch
            
            # Training parameters
            'random_seed': 42,       # For reproducibility
            'verbose': True          # Print progress
        }
    
    def check_and_combine_wiki_files(self) -> bool:
        """
        Check if combined wiki file exists, create if not
        
        Returns:
            True if file exists or was created successfully
        """
        if os.path.exists(self.paths['combined_wiki']):
            file_size = os.path.getsize(self.paths['combined_wiki']) / (1024 * 1024)  # MB
            print(f"Combined wiki file already exists ({file_size:.2f} MB)")
            return True
        
        print("Combining Wikipedia files...")
        combined_count = 0
        
        with open(self.paths['combined_wiki'], 'w', encoding='utf-8') as outfile:
            for i in range(11):
                wiki_file = f'wiki_texts_part_{i}.txt'
                if os.path.exists(wiki_file):
                    with open(wiki_file, 'r', encoding='utf-8') as infile:
                        lines = infile.readlines()
                        outfile.writelines(lines)
                        combined_count += len(lines)
                        print(f"  Added {wiki_file}: {len(lines):,} lines")
                else:
                    print(f"  Warning: {wiki_file} not found")
        
        if combined_count > 0:
            print(f"Successfully combined {combined_count:,} lines")
            self.corpus_stats['total_wiki_lines'] = combined_count
            return True
        return False
    
    def check_and_sample_wiki(self) -> bool:
        """
        Check if sampled wiki file exists, create if not
        
        Returns:
            True if file exists or was created successfully
        """
        if os.path.exists(self.paths['sampled_wiki']):
            with open(self.paths['sampled_wiki'], 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            print(f"Sampled wiki file already exists ({line_count:,} lines)")
            self.corpus_stats['sampled_lines'] = line_count
            return True
        
        if not os.path.exists(self.paths['combined_wiki']):
            if not self.check_and_combine_wiki_files():
                return False
        
        print(f"Sampling {self.config['sample_ratio']*100}% of Wikipedia articles...")
        
        with open(self.paths['combined_wiki'], 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        sample_size = int(len(lines) * self.config['sample_ratio'])
        sampled_lines = random.sample(lines, sample_size)
        
        with open(self.paths['sampled_wiki'], 'w', encoding='utf-8') as f:
            f.writelines(sampled_lines)
        
        print(f"Sampled {len(sampled_lines):,} lines from {len(lines):,} total lines")
        self.corpus_stats['sampled_lines'] = len(sampled_lines)
        return True
    
    def preprocess_text(self, text: str, mode: str = 'word2vec') -> str:
        """
        Optimized preprocessing that preserves proper nouns
        
        Args:
            text: Input text string
            mode: 'word2vec' or 'fasttext'
        
        Returns:
            Preprocessed text string
        """
        # Split into words first
        words = text.split()
        processed_words = []
        
        for word in words:
            # Skip empty strings
            if not word:
                continue
                
            # Check if it's likely a proper noun (capitalized and not at sentence start)
            is_proper_noun = (len(word) > 1 and 
                            word[0].isupper() and 
                            not word.isupper())  # Not all caps (like acronyms)
            
            if is_proper_noun:
                # Keep proper nouns as-is (preserving capitalization)
                # Only remove trailing punctuation
                clean_word = word.rstrip('.,!?;:')
                if clean_word:
                    processed_words.append(clean_word)
            else:
                # Regular word processing
                word_lower = word.lower()
                
                if mode == 'word2vec':
                    # Remove all punctuation and numbers
                    clean_word = re.sub(r'[^a-z]', '', word_lower)
                else:  # fasttext
                    # Keep hyphens for subword learning
                    clean_word = re.sub(r'[^a-z\-]', '', word_lower)
                
                if clean_word:  # Only add non-empty words
                    processed_words.append(clean_word)
        
        # Join and normalize whitespace
        return ' '.join(processed_words)
    
    def prepare_corpus(self, model_type: str = 'word2vec') -> str:
        """
        Prepare and cache preprocessed corpus for training
        
        Args:
            model_type: 'word2vec' or 'fasttext'
        
        Returns:
            Path to the prepared corpus file
        """
        corpus_path = self.paths[f'{model_type}_corpus']
        
        # Check if corpus already exists
        if os.path.exists(corpus_path):
            with open(corpus_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            print(f"{model_type.upper()} corpus already exists ({line_count:,} sentences)")
            return corpus_path
        
        # Ensure sampled wiki exists
        if not self.check_and_sample_wiki():
            raise FileNotFoundError("Failed to prepare wiki sample")
        
        print(f"Preparing corpus for {model_type.upper()}...")
        start_time = time.time()
        
        processed_lines = []
        line_count = 0
        
        with open(self.paths['sampled_wiki'], 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Processing for {model_type}"):
                line = line.strip()
                if len(line) > 10:
                    processed_line = self.preprocess_text(line, mode=model_type)
                    if len(processed_line.split()) >= 3:
                        processed_lines.append(processed_line)
                        line_count += 1
                        
                        # Write in batches to save memory
                        if len(processed_lines) >= self.config['batch_size']:
                            with open(corpus_path, 'a', encoding='utf-8') as out_f:
                                out_f.write('\n'.join(processed_lines) + '\n')
                            processed_lines = []
        
        # Write remaining lines
        if processed_lines:
            with open(corpus_path, 'a', encoding='utf-8') as out_f:
                out_f.write('\n'.join(processed_lines) + '\n')
        
        processing_time = time.time() - start_time
        print(f"Corpus prepared: {line_count:,} sentences in {processing_time:.2f} seconds")
        
        self.corpus_stats[f'{model_type}_sentences'] = line_count
        self.corpus_stats[f'{model_type}_processing_time'] = processing_time
        
        return corpus_path
    
    def train_word2vec(self) -> Dict:
        """
        Train Word2Vec model with optimized parameters using corpus_file method
        
        Returns:
            Dictionary with training results and metrics
        """
        print("\n" + "="*60)
        print("Training Word2Vec Model")
        print("="*60)
        
        # Check if model already exists
        if os.path.exists(self.paths['word2vec_model']):
            print(f"Loading existing Word2Vec model from {self.paths['word2vec_model']}")
            self.models['word2vec'] = Word2Vec.load(self.paths['word2vec_model'])
            return self.training_history['word2vec']
        
        # Prepare corpus
        corpus_path = self.prepare_corpus('word2vec')
        
        # REMOVED: sentence_generator function definition
        # CHANGED: Using corpus_file parameter instead of sentences parameter
        
        print(f"\nTraining Word2Vec with:")
        print(f"  Vector size: {self.config['vector_size']}")
        print(f"  Window: {self.config['window']}")
        print(f"  Workers: {self.config['workers']}")
        print(f"  Epochs: {self.config['epochs']}")
        print(f"  Corpus file: {corpus_path}")
        
        start_time = time.time()
        
        # Train Word2Vec model using corpus_file for memory efficiency
        model = Word2Vec(
            corpus_file=corpus_path,  # CHANGED: Use corpus_file instead of sentences
            vector_size=self.config['vector_size'],  # 'size' renamed to 'vector_size' in Gensim 4.0+
            window=self.config['window'],
            min_count=self.config['min_count'],
            workers=self.config['workers'],
            sg=self.config['sg'],
            negative=self.config['negative'],
            sample=self.config['sample'],
            epochs=self.config['epochs'],
            seed=self.config['random_seed']  # 'seed' parameter for reproducibility
        )
        
        training_time = time.time() - start_time
        
        # Save model
        model.save(self.paths['word2vec_model'])
        self.models['word2vec'] = model
        
        # Record training history
        self.training_history['word2vec'] = {
            'training_time': training_time,
            'vocabulary_size': len(model.wv.index_to_key),  # Gensim 4.0+ uses index_to_key
            'vector_size': self.config['vector_size'],
            'epochs': self.config['epochs'],
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\nWord2Vec Training Complete:")
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Vocabulary size: {len(model.wv.index_to_key):,}")  # Gensim 4.0+ uses index_to_key
        print(f"  Model saved to: {self.paths['word2vec_model']}")
        
        # Test with sample analogies
        self._test_sample_analogies('word2vec')
        
        return self.training_history['word2vec']
    
    def train_fasttext(self) -> Dict:
        """
        Train FastText model with optimized parameters using corpus_file method
        
        Returns:
            Dictionary with training results and metrics
        """
        print("\n" + "="*60)
        print("Training FastText Model")
        print("="*60)
        
        # Check if model already exists
        if os.path.exists(self.paths['fasttext_model']):
            print(f"Loading existing FastText model from {self.paths['fasttext_model']}")
            self.models['fasttext'] = FastText.load(self.paths['fasttext_model'])
            return self.training_history['fasttext']
        
        # Prepare corpus
        corpus_path = self.prepare_corpus('fasttext')
        
        # REMOVED: sentence_generator function definition
        # CHANGED: Using corpus_file parameter instead of sentences parameter
        
        print(f"\nTraining FastText with:")
        print(f"  Vector size: {self.config['vector_size']}")
        print(f"  Window: {self.config['window']}")
        print(f"  Workers: {self.config['workers']}")
        print(f"  Epochs: {self.config['epochs']}")
        print(f"  Character n-grams: {self.config['min_n']}-{self.config['max_n']}")
        print(f"  Corpus file: {corpus_path}")
        
        start_time = time.time()
        
        # Train FastText model using corpus_file for memory efficiency
        model = FastText(
            corpus_file=corpus_path,  # CHANGED: Use corpus_file instead of sentences
            vector_size=self.config['vector_size'],  # 'size' renamed to 'vector_size' in Gensim 4.0+
            window=self.config['window'],
            min_count=self.config['min_count'],
            workers=self.config['workers'],
            sg=self.config['sg'],
            negative=self.config['negative'],
            sample=self.config['sample'],
            epochs=self.config['epochs'],
            min_n=self.config['min_n'],
            max_n=self.config['max_n'],
            seed=self.config['random_seed']  # 'seed' parameter for reproducibility
        )
        
        training_time = time.time() - start_time
        
        # Save model
        model.save(self.paths['fasttext_model'])
        self.models['fasttext'] = model
        
        # Record training history
        self.training_history['fasttext'] = {
            'training_time': training_time,
            'vocabulary_size': len(model.wv.index_to_key),  # Gensim 4.0+ uses index_to_key
            'vector_size': self.config['vector_size'],
            'epochs': self.config['epochs'],
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\nFastText Training Complete:")
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Vocabulary size: {len(model.wv.index_to_key):,}")  # Gensim 4.0+ uses index_to_key
        print(f"  Model saved to: {self.paths['fasttext_model']}")
        
        # Test with sample analogies
        self._test_sample_analogies('fasttext')
        
        return self.training_history['fasttext']
    
    def _test_sample_analogies(self, model_name: str):
        """
        Test model with sample analogies
        
        Args:
            model_name: 'word2vec' or 'fasttext'
        """
        if model_name not in self.models:
            return
        
        model = self.models[model_name]
        
        print(f"\nSample Analogy Tests for {model_name.upper()}:")
        
        # Common analogy examples
        test_cases = [
            ('man', 'woman', 'king'),  # Expected: queen
            ('paris', 'france', 'london'),  # Expected: england
            ('good', 'better', 'bad'),  # Expected: worse
        ]
        
        for word_a, word_b, word_c in test_cases:
            try:
                # Check if words are in vocabulary
                if all(word in model.wv for word in [word_a, word_b, word_c]):
                    result = model.wv.most_similar(
                        positive=[word_b, word_c],
                        negative=[word_a],
                        topn=3
                    )
                    predictions = [word for word, score in result]
                    print(f"  {word_a}:{word_b} :: {word_c}:? -> {predictions}")
                else:
                    missing = [w for w in [word_a, word_b, word_c] if w not in model.wv]
                    print(f"  Skipped (OOV): {missing}")
            except Exception as e:
                print(f"  Error testing {word_a}:{word_b} :: {word_c}:? - {str(e)}")
    
    def compare_models(self):
        """
        Compare performance between Word2Vec and FastText models
        """
        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)
        
        comparison = {}
        
        for model_name in ['word2vec', 'fasttext']:
            if model_name in self.training_history:
                stats = self.training_history[model_name]
                comparison[model_name] = {
                    'Training Time (s)': f"{stats['training_time']:.2f}",
                    'Vocabulary Size': f"{stats['vocabulary_size']:,}",
                    'Vector Size': stats['vector_size'],
                    'Epochs': stats['epochs']
                }
        
        if comparison:
            df = pd.DataFrame(comparison).T
            print(df.to_string())
        else:
            print("No models trained yet")
        
        # OOV handling comparison
        if len(self.models) == 2:
            print("\nOut-of-Vocabulary (OOV) Handling Test:")
            oov_words = ['xyzabc123', 'unknownword999', 'testingoov']
            
            for word in oov_words:
                w2v_can_handle = word in self.models['word2vec'].wv
                # FastText can handle OOV through subword information
                ft_can_handle = True  # FastText always returns something
                print(f"  '{word}': Word2Vec={w2v_can_handle}, FastText={ft_can_handle}")
    
    def save_experiment_results(self):
        """
        Save all experiment results to JSON file
        """
        experiment_data = {
            'configuration': self.config,
            'corpus_statistics': self.corpus_stats,
            'training_history': self.training_history,
            'evaluation_results': self.evaluation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.paths['experiment_log'], 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        print(f"\nExperiment results saved to: {self.paths['experiment_log']}")
    
    def run_complete_training(self):
        """
        Run the complete training pipeline for both models
        """
        print("\n" + "="*60)
        print("Starting Complete Training Pipeline")
        print("="*60)
        
        total_start_time = time.time()
        
        # Step 1: Prepare data
        print("\nStep 1: Data Preparation")
        self.check_and_sample_wiki()
        
        # Step 2: Train Word2Vec
        print("\nStep 2: Training Word2Vec")
        word2vec_results = self.train_word2vec()
        
        # Step 3: Train FastText
        print("\nStep 3: Training FastText")
        fasttext_results = self.train_fasttext()
        
        # Step 4: Compare models
        print("\nStep 4: Model Comparison")
        self.compare_models()
        
        # Step 5: Save results
        print("\nStep 5: Saving Results")
        self.save_experiment_results()
        
        total_time = time.time() - total_start_time
        print(f"\n" + "="*60)
        print(f"Total Training Pipeline Completed in {total_time:.2f} seconds")
        print(f"  Word2Vec: {word2vec_results.get('training_time', 0):.2f}s")
        print(f"  FastText: {fasttext_results.get('training_time', 0):.2f}s")
        print("="*60)



     