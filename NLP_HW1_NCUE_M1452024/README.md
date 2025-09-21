# Word Embeddings Training Project

This project implements Word2Vec and FastText models for word analogy tasks using Wikipedia corpus.

## Prerequisites

- Python 3.10+ 
- Intel i7-14700 (20 cores) or similar CPU recommended
- At least 16GB RAM
- 10GB free disk space

## Data Files Required

Before running, ensure you have these files in your project directory:
- `questions-words.txt` - Google Analogy Dataset
- `wiki_texts_part_0.txt` to `wiki_texts_part_10.txt` - Wikipedia text files

## Installation

### Method 1: Using UV (Recommended) https://docs.astral.sh/uv/getting-started/installation/
```bash
# using powershell- windows10/11 
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

#Use curl to download the script and execute it with sh- Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh


# Create virtual environment and install dependencies
uv venv

#install all dependents
uv sync

# Run the project
uv run NLP_HW1_NCUE_M1452024.py
```

### Method 2: Using pip with virtual environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the project
python NLP_HW1_NCUE_M1452024.py
```

### Method 3: Direct pip installation
```bash
pip install -r requirements.txt
python NLP_HW1_NCUE_M1452024.py
```

## Project Structure

```
.
├── NLP_HW1_NCUE_M1452024.py  # Main script with TODO1-4, 6-7
├── todo5_trainer.py        # TODO5: Word embedding trainer class
├── requirements.txt        # Project dependencies
├── questions-words.txt    # Google Analogy Dataset
├── wiki_texts_part_*.txt  # Wikipedia corpus files (0-10)

```

## Usage

### Run All TODOs (Default)
```bash
uv run NLP_HW1_NCUE_M1452024.py
```

### Run Specific TODOs
Edit `NLP_HW1_NCUE_M1452024.py` and comment/uncomment sections in `if __name__ == "__main__":`

```python
if __name__ == "__main__":
    # TODO1: Process Google Analogy Dataset
    df = todo1_process_data_to_dataframe()
    
    # TODO2-3: Test with pre-trained model
    # preds, golds, model = todo2_predict_with_pretrained()
    # todo3_plot_tsne_family()
    
    # TODO4-5: Train custom models
    # todo4_sample_wikipedia()
    # trainer = Todo5WordEmbeddingTrainer({'sample_ratio': 0.2})
    # trainer.run_complete_training()
    
    # TODO6-7: Evaluate custom models
    # results = todo6_predict_with_custom_embeddings()
    # todo7_plot_tsne_custom_embeddings()
```

## Configuration

Modify hyperparameters in `todo5_trainer.py`:

```python
def _get_default_config(self):
    return {
        'vector_size': 300,      # Embedding dimension
        'window': 5,             # Context window
        'min_count': 5,          # Minimum word frequency
        'workers': 20,           # CPU cores to use
        'epochs': 10,            # Training epochs
        'sample_ratio': 0.2,     # Wikipedia sampling ratio
        # ... other parameters
    }
```

## Expected Runtime

With Intel i7-14700 (20 cores):
- TODO1: < 1 minute
- TODO2-3: ~5 minutes (downloading pre-trained model)
- TODO4: ~2 minutes (sampling)
- TODO5: 
  - 10% data: ~1 hour
  - 20% data: ~2 hours
  - 30% data: ~3 hours
- TODO6-7: ~10 minutes

## Output Files

- `questions-words.csv` - Processed analogy dataset
- `wiki_texts_combined.txt` - Combined Wikipedia corpus
- `wiki_sample.txt` - Sampled Wikipedia data
- `models/word2vec_optimized.model` - Trained Word2Vec model
- `models/fasttext_optimized.model` - Trained FastText model
- `word_relationships.png` - t-SNE visualization (pre-trained)
- `word_relationships_custom.png` - t-SNE visualization (custom)
- `evaluation_report_detailed.csv` - Detailed evaluation results

## Common Issues

### Memory Error
If you encounter memory errors, reduce `sample_ratio` in TODO5:
```python
trainer = Todo5WordEmbeddingTrainer({'sample_ratio': 0.1})
```

### File Not Found
Ensure all Wikipedia files (`wiki_texts_part_*.txt`) are in the project directory.

### Model Not Found
Run TODO5 before TODO6-7. Models must be trained first.

## Key Results

Expected accuracy on Google Analogy Dataset:
- Pre-trained FastText: ~87%
- Custom Word2Vec (20% data): ~24%
- Custom FastText (20% data): ~30%

The lower accuracy of custom models is due to:
1. Limited training data (20% vs full Wikipedia)
2. Preprocessing differences
3. Vocabulary coverage issues

