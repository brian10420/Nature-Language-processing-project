# TAICA NLP Course Projects Repository

This repository contains all assignments for the **Taiwan AI Academy (TAICA)** Natural Language Processing course. All projects are managed using **UV** for consistent dependency management and reproducible environments.

## 📚 Course Information

- **Institution**: Taiwan AI Academy (TAICA)
- **Course**: Natural Language Processing
- **Year**: 2025-2026
- **Student ID**: M1452024
- **University**: National Changhua University of Education (NCUE)

## 🗂️ Project Overview

| Assignment | Title | Description | Status | Key Technologies |
|------------|-------|-------------|---------|------------------|
| HW1 | **Word Embeddings Training** | Implementation of Word2Vec and FastText models for word analogy tasks using Wikipedia corpus | ✅ Complete | Gensim, Word2Vec, FastText |
| HW2 | *[Upcoming]* | TBD | 🔄 Pending | - |
| HW3 | *[Upcoming]* | TBD | 🔄 Pending | - |
| HW4 | *[Upcoming]* | TBD | 🔄 Pending | - |
| HW5 | *[Upcoming]* | TBD | 🔄 Pending | - |

## 📁 Repository Structure

```
TAICA-NLP-Assignments/
│
├── README.md                    # This file
├── .gitignore                   # Global gitignore
│
├── HW1_Word_Embeddings/
│   ├── NLP_HW1_NCUE_M1452024.py
│   ├── todo5_trainer.py
│   ├── requirements.txt
│   ├── pyproject.toml          # UV configuration
│   └── README.md
│
├── HW2_[Topic]/                # Future assignment
│   ├── main.py
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── README.md
│
└── ...
```

## 🛠️ Development Environment

### Prerequisites
- **Python**: 3.10 or higher
- **UV**: Package and project manager ([Installation Guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Hardware**: 
  - CPU: Intel i7-14700 (20 cores) or equivalent
  - RAM: Minimum 16GB
  - Storage: 10-20GB per project

### UV Installation

#### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Linux/macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 🚀 Quick Start

### Running a Specific Assignment

```bash
# Navigate to assignment folder
cd HW1_Word_Embeddings/

# Create virtual environment with UV
uv venv

# Install dependencies
uv sync

# Run the project
uv run NLP_HW1_NCUE_M1452024.py
```

## 📊 Assignment Summaries

### HW1: Word Embeddings Training
- **Objective**: Train and evaluate Word2Vec and FastText models on Google Analogy Dataset
- **Dataset**: Wikipedia corpus (sampled at 5%, 10%, 20%, 30%)
- **Key Findings**:
  - Pre-trained FastText achieves ~87% accuracy
  - Custom models achieve 24-30% due to limited training data
  - FastText handles OOV words but suffers from subword overfitting
  - Geographic categories show poorest performance (0% accuracy)
- **Deliverables**: 7 TODO implementations, evaluation reports, t-SNE visualizations

## 📈 Performance Benchmarks

| Model | Dataset Size | Accuracy (Overall) | Semantic | Syntactic |
|-------|--------------|-------------------|----------|-----------|
| Pre-trained FastText | Full | 87.27% | 85.23% | 88.96% |
| Custom Word2Vec | 20% Wiki | 24.08% | 4.41% | 38.19% |
| Custom FastText | 20% Wiki | 30.43% | 3.39% | 51.98% |
| Custom Word2Vec | 30% Wiki | 25.06% | 4.68% | 41.99% |
| Custom FastText | 30% Wiki | 30.58% | 3.99% | 52.67% |

### HW2-5: [To be updated]
*Details will be added as assignments are released*



## 🔧 Common Commands

```bash
# Create new project with UV
uv init HW2_project_name

# Add dependency
uv add pandas numpy gensim

# Run specific TODO (example)
uv run python -c "from main import todo1_process_data_to_dataframe; todo1_process_data_to_dataframe()"

# Export requirements
uv pip compile pyproject.toml -o requirements.txt
```

## 📝 Notes

- All projects use UV for dependency management to ensure reproducibility
- Each assignment folder contains its own README with specific instructions
- Model files and large datasets are excluded from version control (see .gitignore)
- Training times vary significantly based on hardware and data size

## 🤝 Contributing

This is a personal academic repository for TAICA coursework. While not open for contributions, feedback and discussions about the implementations are welcome.

## 📄 License

Academic use only. All assignments are property of Taiwan AI Academy and National Changhua University of Education.

## 📧 Contact

For questions about the implementations, please refer to the individual assignment READMEs or contact through the TAICA course platform.

---

*Last Updated: September 2025*
