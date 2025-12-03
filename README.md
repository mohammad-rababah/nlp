# IMDB Sentiment Analysis

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Setup

The IMDB dataset should be placed in the following location:
```
aclImdb/
├── train/
│   ├── pos/  (positive reviews)
│   └── neg/  (negative reviews)
└── test/
    ├── pos/  (positive reviews)
    └── neg/  (negative reviews)
```

The dataset path is: `aclImdb` (relative to the script location)

## Run

```bash
python imdb_sentiment_analysis.py
```

