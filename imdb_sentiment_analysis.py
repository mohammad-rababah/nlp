import glob
import os
import re
import ssl
import warnings



import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag

def word_tokenize(text):
    """
    Manual implementation of word tokenization.
    Splits text into tokens by whitespace and handles punctuation.
    
    Parameters:
    -----------
    text : str
        Input text to tokenize
    
    Returns:
    --------
    list
        List of tokens
    """
    if not text or not isinstance(text, str):
        return []
    
    # Split on whitespace (handles multiple spaces, tabs, newlines)
    tokens = text.split()
    
    # Process each token to handle punctuation
    result = []
    punctuation = '.,!?;:()[]{}"\''
    
    for token in tokens:
        if not token:
            continue
            
        # Handle tokens with punctuation
        # Split punctuation from words while preserving contractions
        i = 0
        current_token = []
        
        while i < len(token):
            char = token[i]
            
            # If it's punctuation
            if char in punctuation:
                # Save current token if any
                if current_token:
                    result.append(''.join(current_token))
                    current_token = []
                # Add punctuation as separate token
                result.append(char)
            # Handle apostrophes/contractions (keep them with the word)
            elif char in "'’":
                current_token.append(char)
            # Regular character
            else:
                current_token.append(char)
            
            i += 1
        
        # Add remaining token if any
        if current_token:
            result.append(''.join(current_token))
    
    # Filter out empty strings
    result = [token for token in result if token.strip()]
    
    return result
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

RANDOM_STATE = 42  # Change this value to use different random state
SAMPLE_SIZE_PER_CLASS = 1800  # Number of samples per class (positive/negative) for training
TEST_SAMPLE_PER_CLASS = 200  # Number of samples per class for test

# Toggle optional workflow components here. Add or remove names to customize the run.
WORKFLOW_FEATURES = [
    'download_nltk_data',
    'plot_confusion_matrices',
    'plot_accuracy_comparison',
    'save_results',
    'example_predictions',
    'draw_bag_of_words'
]

# Configure text preprocessing features.
PREPROCESSING_FEATURES = [
    'remove_html_tags',
    'stop_words',
    'stemming',
    'wordnet_lemmatizer',
    'pos_tagging'
]

# Configure vectorization features.
VECTOR_FEATURES = [
    'tfidf_vectorizer',
    'unigram',
    'bigram',
    'trigram',
    'ppmi'
]

# Select which models to train/evaluate.
ENABLED_MODELS = [
    'Linear',
    'Naive Bayes',
    'Custom Naive Bayes',
    'Decision Tree'
]


def is_workflow_feature_enabled(feature_name, enabled_features=None):
    features = set(enabled_features or WORKFLOW_FEATURES)
    return feature_name in features


def is_preprocessing_feature_enabled(feature_name, enabled_features=None):
    features = set(enabled_features or PREPROCESSING_FEATURES)
    return feature_name in features


def is_vector_feature_enabled(feature_name, enabled_features=None):
    features = set(enabled_features or VECTOR_FEATURES)
    return feature_name in features


def is_model_enabled(model_name, enabled_models=None):
    models = set(enabled_models or ENABLED_MODELS)
    return model_name in models

def convert_to_array(matrix, dtype=np.float32):
    if hasattr(matrix, "toarray"):
        return matrix.toarray().astype(dtype)
    return np.asarray(matrix, dtype=dtype)

def combine_feature_blocks(blocks):
    if not blocks:
        raise ValueError("No feature blocks provided for combination.")
    if len(blocks) == 1:
        return blocks[0]
    return np.hstack(blocks)

def compute_idf_vector(train_counts):
    doc_freq = np.count_nonzero(train_counts > 0, axis=0)
    n_docs = train_counts.shape[0]
    idf = np.log((1 + n_docs) / (1 + doc_freq)) + 1
    return idf.astype(np.float32)

def compute_tfidf_from_counts(counts, idf_vector):
    if idf_vector is None:
        return counts.astype(np.float32)
    doc_lengths = counts.sum(axis=1, keepdims=True)
    tf = np.divide(
        counts,
        doc_lengths,
        out=np.zeros_like(counts, dtype=np.float32),
        where=doc_lengths > 0
    )
    tfidf = tf * idf_vector[np.newaxis, :]
    return tfidf.astype(np.float32)

def compute_ppmi_statistics(train_counts):
    total_count = float(train_counts.sum())
    if total_count == 0:
        return np.zeros(train_counts.shape[1], dtype=np.float32), total_count
    word_counts = train_counts.sum(axis=0)
    word_prob = word_counts / total_count
    return word_prob.astype(np.float32), total_count

def compute_ppmi_from_counts(counts, word_prob, total_count):
    counts = counts.astype(np.float32)
    if word_prob is None or total_count <= 0:
        return np.zeros_like(counts, dtype=np.float32)
    doc_totals = counts.sum(axis=1, keepdims=True)
    p_d = np.divide(
        doc_totals,
        total_count,
        out=np.zeros_like(doc_totals, dtype=np.float32),
        where=doc_totals > 0
    )
    p_dw = counts / total_count
    denominator = p_d * word_prob[np.newaxis, :]
    valid = (denominator > 0) & (counts > 0)
    ratio = np.zeros_like(p_dw, dtype=np.float32)
    np.divide(
        p_dw,
        denominator,
        out=ratio,
        where=valid
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.log(ratio, where=ratio > 0, out=np.zeros_like(ratio, dtype=np.float32))
    log_ratio[~np.isfinite(log_ratio)] = 0.0
    log_ratio[log_ratio < 0] = 0.0
    return log_ratio.astype(np.float32)

class FeatureConfig:
    def __init__(self, vectorizer, features_used, idf_vector=None, word_prob=None, total_token_count=None):
        self.vectorizer = vectorizer
        self.features_used = list(features_used)
        self.idf_vector = idf_vector
        self.word_prob = word_prob
        self.total_token_count = total_token_count

    def transform_counts(self, counts):
        counts = counts.astype(np.float32)
        feature_blocks = []
        if 'tfidf_vectorizer' in self.features_used:
            feature_blocks.append(compute_tfidf_from_counts(counts, self.idf_vector))
        if 'ppmi' in self.features_used:
            feature_blocks.append(compute_ppmi_from_counts(counts, self.word_prob, self.total_token_count))
        if not feature_blocks:
            feature_blocks.append(counts)
        return combine_feature_blocks(feature_blocks)

    def transform_texts(self, texts):
        counts_sparse = self.vectorizer.transform(texts)
        counts = convert_to_array(counts_sparse)
        return self.transform_counts(counts)


class NGramCountVectorizer:
    def __init__(self, ngram_range=(1, 1), max_features=None, min_df=1, max_df=1.0):
        if min_df < 1:
            raise ValueError("min_df must be at least 1 for this vectorizer.")
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary_ = {}

    @staticmethod
    def _tokenize(text):
        return text.split()

    def _generate_ngrams(self, tokens):
        min_n, max_n = self.ngram_range
        grams = []
        token_count = len(tokens)
        for n in range(min_n, max_n + 1):
            grams.extend([' '.join(tokens[i:i + n]) for i in range(token_count - n + 1)])
        return grams

    def _build_vocabulary(self, texts):
        term_doc_counts = defaultdict(int)
        term_frequencies = Counter()
        total_docs = len(texts)

        for text in texts:
            tokens = self._tokenize(text)
            ngrams = self._generate_ngrams(tokens)
            doc_counts = Counter(ngrams)
            term_frequencies.update(doc_counts)
            for gram in doc_counts:
                term_doc_counts[gram] += 1

        max_df_threshold = self.max_df * total_docs if self.max_df <= 1.0 else self.max_df
        candidates = []
        for term, doc_freq in term_doc_counts.items():
            if doc_freq < self.min_df:
                continue
            if doc_freq > max_df_threshold:
                continue
            candidates.append((term, term_frequencies[term]))

        candidates.sort(key=lambda item: (-item[1], item[0]))
        if self.max_features is not None:
            candidates = candidates[:self.max_features]

        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(candidates)}

    def fit(self, texts):
        texts = list(texts)
        self._build_vocabulary(texts)
        return self

    def _transform_single(self, text):
        tokens = self._tokenize(text)
        ngrams = self._generate_ngrams(tokens)
        counts = Counter(ngrams)
        vector = np.zeros(len(self.vocabulary_), dtype=np.float32)
        for gram, freq in counts.items():
            index = self.vocabulary_.get(gram)
            if index is not None:
                vector[index] = freq
        return vector

    def transform(self, texts):
        texts = list(texts)
        if not self.vocabulary_:
            raise ValueError("The vectorizer must be fitted before calling transform.")
        vectors = [self._transform_single(text) for text in texts]
        if not vectors:
            return np.zeros((0, len(self.vocabulary_)), dtype=np.float32)
        return np.vstack(vectors)

    def fit_transform(self, texts):
        texts = list(texts)
        self.fit(texts)
        return self.transform(texts)


class CustomMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def get_params(self, deep=False):
        return {'alpha': self.alpha}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @staticmethod
    def _ensure_array(X):
        return convert_to_array(X, dtype=np.float32)

    def fit(self, X, y):
        X = self._ensure_array(X)
        y = np.asarray(y)
        self.classes_, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        class_feature_counts = np.zeros((n_classes, n_features), dtype=np.float64)
        class_token_totals = np.zeros(n_classes, dtype=np.float64)
        class_doc_counts = np.zeros(n_classes, dtype=np.float64)

        for idx, class_index in enumerate(y_indices):
            class_feature_counts[class_index] += X[idx]
            class_token_totals[class_index] += X[idx].sum()
            class_doc_counts[class_index] += 1

        smoothed_counts = class_feature_counts + self.alpha
        smoothed_totals = class_token_totals + self.alpha * n_features
        self.feature_log_prob_ = np.log(smoothed_counts / smoothed_totals[:, None])
        self.class_log_prior_ = np.log(class_doc_counts / class_doc_counts.sum())
        return self

    def _joint_log_likelihood(self, X):
        X = self._ensure_array(X)
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X):
        log_probs = self._joint_log_likelihood(X)
        class_indices = np.argmax(log_probs, axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, X):
        log_probs = self._joint_log_likelihood(X)
        log_probs = log_probs - log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs_sum = probs.sum(axis=1, keepdims=True)
        probs /= np.where(probs_sum == 0, 1, probs_sum)
        return probs

def download_nltk_data():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng')

def load_imdb_data(data_path='aclImdb'):

    print(f"Loading IMDB dataset with {SAMPLE_SIZE_PER_CLASS} samples per class...")
    
    def load_reviews_from_directory(directory, sentiment, max_samples=None):
        reviews = []
        sentiment_labels = []        
        txt_files = glob.glob(os.path.join(directory, "*.txt")) # read all txt files in dir
        np.random.seed(RANDOM_STATE) # to preserve same random each time
        np.random.shuffle(txt_files)
        
        # Limit number of files if max_samples is specified
        if max_samples:
            txt_files = txt_files[:max_samples] # 12.5k keep 1600
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    review_text = file.read().strip()
                    if review_text:  # Only add non-empty reviews
                        reviews.append(review_text)
                        sentiment_labels.append(sentiment)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        return reviews, sentiment_labels
    
    # Load training data with sample limit
    train_pos_dir = os.path.join(data_path, "train", "pos") # aclImdb/train/pos
    train_neg_dir = os.path.join(data_path, "train", "neg")
    
    train_pos_reviews, train_pos_labels = load_reviews_from_directory(train_pos_dir, "positive", SAMPLE_SIZE_PER_CLASS)
    train_neg_reviews, train_neg_labels = load_reviews_from_directory(train_neg_dir, "negative", SAMPLE_SIZE_PER_CLASS)
    
    # Combine training data
    train_reviews = train_pos_reviews + train_neg_reviews
    train_labels = train_pos_labels + train_neg_labels
    
    train_data = pd.DataFrame({
        'text': train_reviews,
        'sentiment': train_labels
    })
    train_data = train_data.sample(frac=1, random_state=RANDOM_STATE)

    # Load test data with sample limit
    test_pos_dir = os.path.join(data_path, "test", "pos")
    test_neg_dir = os.path.join(data_path, "test", "neg")
    
    test_pos_reviews, test_pos_labels = load_reviews_from_directory(test_pos_dir, "positive", TEST_SAMPLE_PER_CLASS)
    test_neg_reviews, test_neg_labels = load_reviews_from_directory(test_neg_dir, "negative", TEST_SAMPLE_PER_CLASS)
    
    # Combine test data
    test_reviews = test_pos_reviews + test_neg_reviews
    test_labels = test_pos_labels + test_neg_labels
    
    test_data = pd.DataFrame({
        'text': test_reviews,
        'sentiment': test_labels
    })
    test_data = test_data.sample(frac=1, random_state=RANDOM_STATE)
    print(f"Training data loaded: {len(train_data)} samples")
    print(f"Test data loaded: {len(test_data)} samples")
    print(f"Training class distribution:\n{train_data['sentiment'].value_counts()}")
    print(f"Test class distribution:\n{test_data['sentiment'].value_counts()}")
    
    return train_data, test_data

def preprocess_text(text, preprocess_config):
    use_pos_tagging = preprocess_config.get('pos_tagging', False)
    
    # Apply HTML tag removal first if enabled
    if preprocess_config.get('remove_html_tags', False):
        text = re.sub(r'<[^>]+>', ' ', text)
    
    # Lowercase the text
    text_lower = text.lower()
    
    # Tokenize for POS tagging (with punctuation for better accuracy)
    if use_pos_tagging:
        try:
            # Tokenize with punctuation preserved for better POS tagging
            tokens_for_pos = word_tokenize(text_lower)
            pos_tagged = pos_tag(tokens_for_pos)
            # Create a mapping: token -> POS tag (normalize tokens for matching)
            pos_dict = {}
            for token, tag in pos_tagged:
                # Remove punctuation from token for matching
                clean_token = re.sub(r'[^a-zA-Z]', '', token)
                if clean_token and len(clean_token) >= 1:
                    # Use the first tag found for each clean token
                    if clean_token not in pos_dict:
                        pos_dict[clean_token] = tag
        except Exception as e:
            print(f"Warning: POS tagging failed: {e}")
            pos_dict = {}
            use_pos_tagging = False
    else:
        pos_dict = {}
    
    # Now apply standard preprocessing (remove punctuation)
    text_clean = re.sub(r'[^a-zA-Z\s]', '', text_lower)
    tokens = word_tokenize(text_clean)

    stop_words = preprocess_config.get('stop_words')
    lemmatizer = preprocess_config.get('lemmatizer')
    stemmer = preprocess_config.get('stemmer')
    min_token_length = preprocess_config.get('min_token_length', 3)

    processed_tokens = []
    for token in tokens:
        if stop_words and token in stop_words:
            continue
        
        # Get POS tag before transformation (for better matching)
        pos_tag_val = None
        if use_pos_tagging:
            pos_tag_val = pos_dict.get(token, None)
        
        transformed_token = token
        if lemmatizer:
            transformed_token = lemmatizer.lemmatize(transformed_token)
        if stemmer:
            transformed_token = stemmer.stem(transformed_token)
        
        if len(transformed_token) >= min_token_length:
            # Append POS tag to token if POS tagging is enabled and tag was found
            if use_pos_tagging and pos_tag_val:
                processed_tokens.append(f"{transformed_token}_{pos_tag_val}")
            else:
                processed_tokens.append(transformed_token)

    return ' '.join(processed_tokens)

def preprocess_dataset(data, preprocess_config):
    print("Preprocessing text data...")
    data = data.copy()
    data['text'] = data['text'].fillna('').astype(str)
    data['processed_text'] = data['text'].apply(
        lambda x: preprocess_text(x, preprocess_config)
    )
    data['processed_text'] = data['processed_text'].astype(str)
    data = data[data['processed_text'].str.len() > 0]
    print(f"After preprocessing: {len(data)} samples")
    return data

def train_test_split(data, test_size=0.15, random_state=None, stratify=None):
    """
    Native implementation of train_test_split with stratification support.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset to split
    test_size : float, default=0.15
        Proportion of dataset to include in the test split
    random_state : int, default=None
        Random seed for reproducibility
    stratify : array-like, default=None
        If not None, data is split in a stratified fashion using this as class labels
    
    Returns:
    --------
    train_data : pandas.DataFrame
        Training set
    test_data : pandas.DataFrame
        Test set
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    data = data.copy()
    data = data.reset_index(drop=True)
    
    if stratify is not None:
        # Stratified split: maintain class distribution
        train_indices = []
        test_indices = []
        
        # Get unique classes
        if hasattr(stratify, 'values'):
            stratify = stratify.values
        
        unique_classes = np.unique(stratify)
        
        for class_label in unique_classes:
            # Get indices for this class
            class_indices = np.where(stratify == class_label)[0]
            n_class = len(class_indices)
            n_test = int(n_class * test_size)
            
            # Shuffle indices for this class
            shuffled_indices = class_indices.copy()
            np.random.shuffle(shuffled_indices)
            
            # Split indices
            class_test_indices = shuffled_indices[:n_test]
            class_train_indices = shuffled_indices[n_test:]
            
            test_indices.extend(class_test_indices)
            train_indices.extend(class_train_indices)
        
        # Convert to arrays and shuffle to mix classes
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        
        train_data = data.iloc[train_indices].reset_index(drop=True)
        test_data = data.iloc[test_indices].reset_index(drop=True)
    else:
        # Simple random split
        n_samples = len(data)
        n_test = int(n_samples * test_size)
        
        # Shuffle indices
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        train_data = data.iloc[train_indices].reset_index(drop=True)
        test_data = data.iloc[test_indices].reset_index(drop=True)
    
    return train_data, test_data

def create_train_val_test_split(train_data, val_size=0.15):
    print(f"Creating train/validation/test splits with validation size: {val_size}")  
    # Split training data into train and validation
    # 1600 pos , 1600 neg -> total
    # 320 pos , 320 neg -> validation
    #
    train_data_split, val_data = train_test_split(
        train_data, 
        test_size=val_size, 
        random_state=RANDOM_STATE, 
        stratify=train_data['sentiment'] # imprtant balance
    )
    
    print(f"Final splits - Train: {len(train_data_split)}, Validation: {len(val_data)}")
    print(f"Train class distribution:\n{train_data_split['sentiment'].value_counts()}")
    print(f"Validation class distribution:\n{val_data['sentiment'].value_counts()}")

    return train_data_split, val_data,

def extract_features(train_data, val_data, test_data, max_features=10000, vector_features=None):

    vector_feature_flags = vector_features or VECTOR_FEATURES
    # hi what is your name
    # [hi,what,is,your,name]
    # [hi what,what is , is your , your name] I have not
    # [hi,what,is,your,name,hi what,what is , is your , your name]
    ngram_options = [] # nrange -> 1,2 -> 1 word , 2 words []
    if is_vector_feature_enabled('unigram', vector_feature_flags):
        ngram_options.append(1)
    if is_vector_feature_enabled('bigram', vector_feature_flags):
        ngram_options.append(2)
    if is_vector_feature_enabled('trigram', vector_feature_flags):
        ngram_options.append(3)
    if not ngram_options:
        ngram_options.append(1)
    ngram_options = sorted(set(ngram_options))
    if len(ngram_options) == 1:
        ngram_range = (ngram_options[0], ngram_options[0]) #example (2,2) if I set only bi gram
    else:
        ngram_range = (ngram_options[0], ngram_options[-1]) # if I enable everything [1,2,3] -> (1,3)
    #     (1,3) -> (1,1)
    print(f"Extracting features using custom count vectorization with n-gram range {ngram_range}...")

    count_vectorizer = NGramCountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95
    )

    train_counts = count_vectorizer.fit_transform(train_data['processed_text'].tolist())
    val_counts = count_vectorizer.transform(val_data['processed_text'].tolist())
    test_counts = count_vectorizer.transform(test_data['processed_text'].tolist())

    transform_features = []
    idf_vector = None
    word_prob = None
    total_token_count = None

    if is_vector_feature_enabled('tfidf_vectorizer', vector_feature_flags):
        idf_vector = compute_idf_vector(train_counts)
        transform_features.append('tfidf_vectorizer')

    if is_vector_feature_enabled('ppmi', vector_feature_flags):
        word_prob, total_token_count = compute_ppmi_statistics(train_counts)
        transform_features.append('ppmi')

    feature_config = FeatureConfig(
        vectorizer=count_vectorizer,
        features_used=transform_features,
        idf_vector=idf_vector,
        word_prob=word_prob,
        total_token_count=total_token_count
    )

    X_train = feature_config.transform_counts(train_counts)
    X_val = feature_config.transform_counts(val_counts)
    X_test = feature_config.transform_counts(test_counts)

    y_train = train_data['sentiment']
    y_val = val_data['sentiment']
    y_test = test_data['sentiment']

    print(f"Feature matrix shape - Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    print(f"Active feature transforms: {transform_features if transform_features else ['count']}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_config

def get_available_models():
    return {
        'Linear': SGDClassifier(random_state=RANDOM_STATE, max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Custom Naive Bayes': CustomMultinomialNB(alpha=1.0),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=None)
    }


def train_models(X_train, y_train, X_val, y_val, X_test, y_test, enabled_models=None):
    print("Training machine learning models...")
    available_models = get_available_models()
    enabled = [
        model_name for model_name in available_models.keys()
        if is_model_enabled(model_name, enabled_models)
    ]
    if not enabled:
        raise ValueError("No models enabled. Please add at least one model name to ENABLED_MODELS.")
    models = {name: available_models[name] for name in enabled}
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_val_pred = model.predict(X_val)
        model_label_encoder = None
        
        accuracy = accuracy_score(y_test, y_pred) # calculate test accuracy
        val_accuracy = accuracy_score(y_val, y_val_pred) # calculate validation accuracy
        results[name] = {
            'model': model,
            'label_encoder': model_label_encoder,
            'accuracy': accuracy,
            'val_accuracy': val_accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        print(f"{name} - Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {accuracy:.4f}")
    
    return results

def evaluate_models(results):
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Validation Accuracy': [results[model].get('val_accuracy', 0.0) for model in results.keys()],
        'Test Accuracy': [results[model]['accuracy'] for model in results.keys()]
    }).sort_values('Test Accuracy', ascending=False)
    print("\nModel Performance Comparison:")
    print(results_df.to_string(index=False))
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORTS")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        print(result['classification_report'])

def plot_confusion_matrices(results, save_path='imdb_confusion_matrices.png'):
    n_models = len(results)
    cols = 2
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, (model_name, result) in enumerate(results.items()):
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        val_acc = result.get('val_accuracy', 0.0)
        test_acc = result["accuracy"]
        axes[i].set_title(f'{model_name}\nVal Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        
        # Set labels for confusion matrix
        classes = sorted(result['y_test'].unique())
        axes[i].set_xticklabels(classes)
        axes[i].set_yticklabels(classes)
    
    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrices saved to {save_path}")

def plot_accuracy_comparison(results, save_path='imdb_accuracy_comparison.png'):
    models = list(results.keys())
    val_accuracies = [results[model].get('val_accuracy', 0.0) for model in models]
    test_accuracies = [results[model]['accuracy'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, val_accuracies, width, label='Validation Accuracy', color='skyblue')
    bars2 = plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy', color='lightcoral')
    
    plt.title('IMDB Sentiment Analysis - Model Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(x, models, rotation=45)
    plt.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Accuracy comparison saved to {save_path}")

def predict_sentiment(text, feature_config, model, preprocess_config, label_encoder=None):
    processed_text = preprocess_text(text, preprocess_config)
    feature_matrix = feature_config.transform_texts([processed_text])
    if label_encoder is not None:
        prediction_encoded = model.predict(feature_matrix)
        prediction = label_encoder.inverse_transform(prediction_encoded)[0]
    else:
        prediction = model.predict(feature_matrix)[0]
    if hasattr(model, 'predict_proba'):
        confidence = np.max(model.predict_proba(feature_matrix))
    else:
        confidence = 1.0
    return prediction, confidence

def save_results(results, data_info, filename='imdb_sentiment_analysis_results.txt'):
    with open(filename, 'w') as f:
        f.write("IMDB SENTIMENT ANALYSIS RESULTS\n")
        f.write("="*50 + "\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"Training samples: {data_info['train_samples']}\n")
        f.write(f"Validation samples: {data_info['val_samples']}\n")
        f.write(f"Test samples: {data_info['test_samples']}\n")
        f.write(f"Features: {data_info['features']}\n\n")
        
        f.write("Model Performance:\n")
        f.write("-" * 30 + "\n")
        for model_name, result in results.items():
            val_acc = result.get('val_accuracy', 0.0)
            test_acc = result['accuracy']
            f.write(f"{model_name}: Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}\n")
        
        f.write("\nDetailed Classification Reports:\n")
        f.write("="*50 + "\n")
        for model_name, result in results.items():
            f.write(f"\n{model_name}:\n")
            f.write("-" * 30 + "\n")
            f.write(result['classification_report'])
            f.write("\n")
    
    print(f"Results saved to {filename}")

DEFAULT_EXAMPLE_TEXTS = [
    "I absolutely loved this movie! The performances were outstanding and the story kept me engaged.",
    "What a waste of time. The plot was dull and the acting was terrible.",
    "The film had its moments, but overall it felt too long and uneven."
]

def draw_bag_of_words_diagram(save_path='bag_of_words_diagram.png', train_data=None, preprocess_config=None):
    """
    Create histograms showing most frequent words in positive and negative reviews.
    """
    # Use actual IMDB data if provided, otherwise load sample
    if train_data is None or preprocess_config is None:
        print("Loading IMDB data for word frequency histograms...")
        # Load data for visualization
        train_data_sample, _ = load_imdb_data()
        # Use more samples for better statistics
        train_data_sample = train_data_sample.head(500)  # Use 500 samples for better word frequency
        
        # Initialize preprocessing config
        remove_html_tags = is_preprocessing_feature_enabled('remove_html_tags')
        lemmatizer = WordNetLemmatizer() if is_preprocessing_feature_enabled('wordnet_lemmatizer') else None
        stemmer = PorterStemmer() if is_preprocessing_feature_enabled('stemming') else None
        stop_words = set(stopwords.words('english')) if is_preprocessing_feature_enabled('stop_words') else set()
        pos_tagging = is_preprocessing_feature_enabled('pos_tagging')
        
        preprocess_config = {
            'remove_html_tags': remove_html_tags,
            'stop_words': stop_words,
            'lemmatizer': lemmatizer,
            'stemmer': stemmer,
            'pos_tagging': pos_tagging,
            'min_token_length': 3
        }
        
        # Preprocess the sample
        train_data_sample = preprocess_dataset(train_data_sample, preprocess_config)
    else:
        # Use provided data
        train_data_sample = train_data.copy()
    
    # Separate positive and negative reviews
    positive_reviews = train_data_sample[train_data_sample['sentiment'] == 'positive']['processed_text'].tolist()
    negative_reviews = train_data_sample[train_data_sample['sentiment'] == 'negative']['processed_text'].tolist()
    
    # Count word frequencies for each sentiment
    positive_word_counts = Counter()
    negative_word_counts = Counter()
    
    for review in positive_reviews:
        words = review.split()
        positive_word_counts.update(words)
    
    for review in negative_reviews:
        words = review.split()
        negative_word_counts.update(words)
    
    # Get top N most frequent words for each sentiment
    top_n = 20
    top_positive_words = [word for word, count in positive_word_counts.most_common(top_n)]
    top_negative_words = [word for word, count in negative_word_counts.most_common(top_n)]
    
    # Get frequencies
    positive_freqs = [positive_word_counts[word] for word in top_positive_words]
    negative_freqs = [negative_word_counts[word] for word in top_negative_words]
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Positive reviews histogram
    bars1 = ax1.barh(range(len(top_positive_words)), positive_freqs, color='green', alpha=0.7)
    ax1.set_yticks(range(len(top_positive_words)))
    ax1.set_yticklabels(top_positive_words)
    ax1.invert_yaxis()  # Top word at the top
    ax1.set_xlabel('Word Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top {top_n} Most Frequent Words - Positive Reviews', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, freq) in enumerate(zip(bars1, positive_freqs)):
        ax1.text(bar.get_width() + max(positive_freqs) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{freq}', ha='left', va='center', fontweight='bold', fontsize=9)
    
    # Negative reviews histogram
    bars2 = ax2.barh(range(len(top_negative_words)), negative_freqs, color='red', alpha=0.7)
    ax2.set_yticks(range(len(top_negative_words)))
    ax2.set_yticklabels(top_negative_words)
    ax2.invert_yaxis()  # Top word at the top
    ax2.set_xlabel('Word Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Top {top_n} Most Frequent Words - Negative Reviews', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, freq) in enumerate(zip(bars2, negative_freqs)):
        ax2.text(bar.get_width() + max(negative_freqs) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{freq}', ha='left', va='center', fontweight='bold', fontsize=9)
    
    plt.suptitle('Most Frequent Words in Positive vs Negative Reviews - IMDB Sentiment Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Word frequency histograms saved to {save_path}")
    print(f"Positive reviews analyzed: {len(positive_reviews)}")
    print(f"Negative reviews analyzed: {len(negative_reviews)}")


def generate_example_predictions(examples, best_model_name, results, feature_config, preprocess_config):
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    best_model_info = results[best_model_name]
    best_model = best_model_info['model']
    label_encoder = best_model_info.get('label_encoder')
    for text in examples:
        prediction, confidence = predict_sentiment(
            text,
            feature_config,
            best_model,
            preprocess_config,
            label_encoder=label_encoder
        )
        print(f"Review: {text}")
        print(f"Predicted Sentiment: {prediction} (confidence {confidence:.2f})")
        print("-" * 40)

def main():
    print("IMDB Sentiment Analysis Assignment 2025-2026")
    print("="*50)
    
    # Draw Bag of Words diagram if enabled
    if is_workflow_feature_enabled('draw_bag_of_words'):
        draw_bag_of_words_diagram()
    else:
        print("Skipping Bag of Words diagram (feature disabled).")
    
    # Download NLTK data
    if is_workflow_feature_enabled('download_nltk_data'):
        download_nltk_data()
    else:
        print("Skipping NLTK data download (feature disabled).")
    
    # Initialize preprocessing tools
    remove_html_tags = is_preprocessing_feature_enabled('remove_html_tags')
    lemmatizer = WordNetLemmatizer() if is_preprocessing_feature_enabled('wordnet_lemmatizer') else None
    stemmer = PorterStemmer() if is_preprocessing_feature_enabled('stemming') else None
    stop_words = set(stopwords.words('english')) if is_preprocessing_feature_enabled('stop_words') else set()
    pos_tagging = is_preprocessing_feature_enabled('pos_tagging')

    preprocess_config = {
        'remove_html_tags': remove_html_tags,
        'stop_words': stop_words,
        'lemmatizer': lemmatizer,
        'stemmer': stemmer,
        'pos_tagging': pos_tagging,
        'min_token_length': 3
    }
    print(
        "Preprocessing configuration - "
        f"remove_html_tags: {'on' if remove_html_tags else 'off'}, "
        f"stop_words: {'on' if stop_words else 'off'}, "
        f"stemming: {'on' if stemmer else 'off'}, "
        f"lemmatizer: {'on' if lemmatizer else 'off'}, "
        f"pos_tagging: {'on' if pos_tagging else 'off'}"
    )
    if stop_words:
        sample_stop_words = sorted(list(stop_words))[:5]
        print(f"Stop words sample: {sample_stop_words}")
    
    # Load IMDB dataset
    train_data, test_data = load_imdb_data()

    # Preprocess datasets
    train_data = preprocess_dataset(train_data, preprocess_config)
    test_data = preprocess_dataset(test_data, preprocess_config)
    print("train_data sample after preprocessing: ", train_data.head())
    print("test_data sample after preprocessing: ", test_data.head())
    # Create train/validation/test splits
    train_data, val_data = create_train_val_test_split(train_data)
    print("train_data sample: \n", train_data.head())
    print("development sample: \n", val_data.head())
    print("test_data sample: \n", test_data.head())
    # Extract features
    X_train, X_val, X_test, y_train, y_val, y_test, feature_configuration = extract_features(
        train_data, val_data, test_data, max_features=10000, vector_features=VECTOR_FEATURES
    )
    
    # Train models
    results = train_models(X_train, y_train, X_val, y_val, X_test, y_test, enabled_models=ENABLED_MODELS)
    
    # Evaluate models
    evaluate_models(results)
    
    # Create visualizations
    if is_workflow_feature_enabled('plot_confusion_matrices'):
        plot_confusion_matrices(results)
    else:
        print("Skipping confusion matrix plotting (feature disabled).")

    if is_workflow_feature_enabled('plot_accuracy_comparison'):
        plot_accuracy_comparison(results)
    else:
        print("Skipping accuracy comparison plot (feature disabled).")
    
    
    data_info = {
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'features': X_train.shape[1]
    }

    if is_workflow_feature_enabled('save_results'):
        save_results(results, data_info, filename='imdb_sentiment_analysis_results.txt')
    else:
        print("Skipping results saving (feature disabled).")

    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    print(f"Best model: {best_model_name}")

    if is_workflow_feature_enabled('example_predictions'):
        generate_example_predictions(
            DEFAULT_EXAMPLE_TEXTS,
            best_model_name,
            results,
            feature_configuration,
            preprocess_config
        )
    else:
        print("Skipping example predictions (feature disabled).")


if __name__ == "__main__":
    main()
