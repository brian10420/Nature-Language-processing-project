# Word Embeddings Training Project

This project implements Word2Vec and FastText models for word analogy tasks using Wikipedia corpus, with significant improvements in preprocessing and hyperparameter tuning.

## Prerequisites

- Python 3.10+ 
- Intel i7-14700 (20 cores) or similar CPU recommended
- NVIDIA GeForce RTX 4090 (optional, for future GPU implementations)
- At least 16GB RAM
- 10GB free disk space

## Data Files Required

Before running, ensure you have these files in your project directory:
- `questions-words.txt` - Google Analogy Dataset
- `wiki_texts_part_0.txt` to `wiki_texts_part_10.txt` - Wikipedia text files

## Installation

### Method 1: Using UV (Recommended) https://docs.astral.sh/uv/getting-started/installation/
```bash
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv sync

# Run the project
uv run main.py
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
python main.py
```

## Project Structure

```
.
├── main.py                  # Main script with TODO1-4, 6-7
├── todo5_trainer.py        # TODO5: Word embedding trainer class
├── model_diagnostic.py     # Model evaluation and diagnostics
├── requirements.txt        # Project dependencies
├── questions-words.txt     # Google Analogy Dataset
├── wiki_texts_part_*.txt  # Wikipedia corpus files (0-10)
└── models/                 # Trained model storage
    ├── word2vec_optimized.model
    └── fasttext_optimized.model
```

## Key Improvements (Updated)

### 1. **Preprocessing Enhancements**
- **Case Normalization**: All text converted to lowercase for consistency
- **Sentence Boundary Preservation**: Maintains sentence structure for better context
- **Dynamic Filtering**: Adjusts minimum word frequency based on data size

### 2. **Hyperparameter Optimization**
```python
# Optimized configuration (varies by data size)
{
    'vector_size': 300,      # Increased from 100
    'window': 5,             # Optimal for semantic tasks
    'min_count': 2-10,       # Dynamic adjustment
    'workers': 20,           # Full CPU utilization
    'epochs': 7,             # Increased from 3
    'sg': 0,                 # CBOW for better frequent word handling
    'sample': 1e-5,          # Optimized downsampling
}
```

### 3. **Critical Bug Fix**
- **Case Mismatch Resolution**: Test data ("Athens") now properly matches training data ("athens")
- This single fix improved semantic accuracy from 4% to 47-72%!

## Performance Results (Latest)

### Batch Experiment Results (5%-50% Wikipedia Sample)

| Model | Sample | Overall | Semantic | Syntactic |
|-------|--------|---------|----------|-----------|
| Word2Vec | 5% | 59.7% | 55.6% | 55.6% |
| Word2Vec | 10% | 63.3% | 69.2% | 58.4% |
| Word2Vec | 20% | 65.3% | 70.8% | 60.7% |
| Word2Vec | 30% | 66.2% | 72.3% | 61.2% |
| Word2Vec | 50% | 66.8% | 72.4% | 61.8% |
| FastText | 5% | 35.9% | 10.3% | 57.1% |
| FastText | 10% | 40.9% | 17.7% | 60.3% |
| FastText | 20% | 48.7% | 31.1% | 63.4% |
| FastText | 30% | 54.4% | 39.5% | 66.8% |
| FastText | 50% | 54.7% | 41.2% | 65.9% |

### Key Findings
1. **Word2Vec outperforms FastText on semantic tasks** (72.4% vs 41.2% at 50% data)
2. **FastText shows consistent syntactic performance** (57-66% across all data sizes)
3. **Semantic accuracy dramatically improved** after case normalization fix (4% → 72%)
4. **Data scaling shows diminishing returns** beyond 30% for both models

## Usage

### Run Complete Pipeline
```bash
# Run all experiments with different sample ratios
uv run python main.py
```

### Run Specific Components
```python
if __name__ == "__main__":
    # TODO1: Process Google Analogy Dataset
    df = todo1_process_data_to_dataframe()
    
    # TODO2-3: Test with pre-trained model
    preds, golds, model = todo2_predict_with_pretrained()
    todo3_plot_tsne_family()
    
    # TODO4-5: Train custom models (single ratio)
    trainer = Todo5WordEmbeddingTrainer({'sample_ratio': 0.2})
    trainer.run_complete_training()
    
    # TODO6-7: Evaluate custom models
    results = todo6_predict_with_custom_embeddings()
    todo7_plot_tsne_custom_embeddings()
    
    # Batch experiments (multiple ratios)
    batch_results = run_batch_experiments()
```

## Configuration Options

### Dynamic Configuration (todo5_trainer.py)
```python
def get_dynamic_config(sample_ratio):
    if sample_ratio <= 0.05:
        return {
            'min_count': 2,
            'epochs': 5,
            'vector_size': 100
        }
    elif sample_ratio <= 0.2:
        return {
            'min_count': 5,
            'epochs': 7,
            'vector_size': 200
        }
    else:  # >= 0.3
        return {
            'min_count': 8,
            'epochs': 10,
            'vector_size': 300,
            'sg': 0  # CBOW for larger datasets
        }
```

## Expected Runtime

With Intel i7-14700 (20 cores):
- TODO1: < 1 minute
- TODO2-3: ~5 minutes (downloading pre-trained model)
- TODO4: ~2 minutes (sampling)
- TODO5 (single ratio): 
  - 5% data: ~30 minutes
  - 20% data: ~2 hours
  - 50% data: ~4 hours
- TODO6-7: ~10 minutes
- Batch experiments (5%-50%): ~8 hours total

## Output Files

- `questions-words.csv` - Processed analogy dataset
- `wiki_texts_combined.txt` - Combined Wikipedia corpus
- `wiki_sample.txt` - Sampled Wikipedia data
- `models/` - Trained models directory
- `experiments/` - Experiment logs and results
- `batch_results_*.json` - Detailed batch experiment results
- `model_comparison_separate.png` - Performance comparison charts
- `word_relationships_custom.png` - t-SNE visualizations
- `evaluation_report_detailed.csv` - Detailed evaluation metrics

## Troubleshooting

### Memory Issues
```python
# Reduce batch size if encountering memory errors
config = {'batch_size': 5000}  # Default: 10000
```

### Poor Semantic Performance
- Ensure case normalization in evaluation (`words.lower()`)
- Check vocabulary coverage with `model_diagnostic.py`
- Verify preprocessing consistency between training and testing

### FastText Underperformance
Consider adjusting:
- `min_n`: 2 (instead of 3)
- `max_n`: 4 (instead of 6)
- `sample`: 1e-4 (less aggressive downsampling)

## Model Diagnostics

Run diagnostics to understand model behavior:
```bash
python model_diagnostic.py
```

This will show:
- Vocabulary coverage statistics
- OOV (out-of-vocabulary) analysis
- Key word frequency analysis
- Model similarity tests

## Future Improvements

1. **GPU Acceleration**: Implement CUDA support for RTX 4090
2. **Advanced Preprocessing**: Lemmatization and named entity recognition
3. **Hyperparameter Tuning**: Grid search or Bayesian optimization
4. **Ensemble Methods**: Combine Word2Vec and FastText predictions
5. **Transfer Learning**: Fine-tune pre-trained models on domain-specific data

## Citation

If using this code for research, please cite:
```
@misc{ncue_nlp_2025,
  author = {NCUE M1452024},
  title = {Word Embeddings Training on Wikipedia Corpus},
  year = {2025},
  publisher = {Taiwan AI Academy},
  course = {Natural Language Processing}
}
```