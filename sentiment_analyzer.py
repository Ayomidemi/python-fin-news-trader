import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import re
import string
from collections import Counter

# Download NLTK resources if needed
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If model not found, we'll need to download it
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Financial-specific terms to adjust sentiment scores
FINANCIAL_POSITIVE_TERMS = {
    'beat', 'beats', 'bullish', 'outperform', 'strong', 'strength', 'upgrade', 
    'upside', 'opportunity', 'opportunities', 'positive', 'profit', 'profitable',
    'growth', 'growing', 'dividend', 'dividends', 'recommendation', 'recommended',
    'invest', 'investing', 'buy', 'buying', 'recovery', 'gain', 'gains', 'winner'
}

FINANCIAL_NEGATIVE_TERMS = {
    'miss', 'misses', 'bearish', 'underperform', 'weak', 'weakness', 'downgrade',
    'downside', 'risk', 'risks', 'negative', 'loss', 'losses', 'bankruptcy',
    'debt', 'debts', 'decline', 'declining', 'sell', 'selling', 'recession',
    'drop', 'drops', 'loser', 'volatile', 'volatility', 'investigation', 'lawsuit'
}

# Custom financial news sentiment analyzer
def analyze_sentiment(text, financial_adjustment=True):
    """
    Analyze sentiment of financial news text
    
    Args:
        text (str): Text content to analyze
        financial_adjustment (bool): Whether to apply financial-specific adjustments
        
    Returns:
        float: Sentiment score between -1 (negative) and 1 (positive)
    """
    if not text:
        return 0.0
        
    # Clean text
    text = clean_text(text)
    
    # Get base sentiment from VADER
    sentiment = sia.polarity_scores(text)
    compound_score = sentiment['compound']
    
    # Apply financial domain-specific adjustments
    if financial_adjustment:
        words = set(text.lower().split())
        
        # Count financial terms
        positive_matches = words.intersection(FINANCIAL_POSITIVE_TERMS)
        negative_matches = words.intersection(FINANCIAL_NEGATIVE_TERMS)
        
        # Adjust score based on financial terms
        pos_adjustment = len(positive_matches) * 0.05
        neg_adjustment = len(negative_matches) * 0.05
        
        # Apply adjustments, keeping score between -1 and 1
        adjusted_score = compound_score + pos_adjustment - neg_adjustment
        adjusted_score = max(-1.0, min(1.0, adjusted_score))
        
        return adjusted_score
    
    return compound_score

def clean_text(text):
    """
    Clean text for NLP processing
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_named_entities(text):
    """
    Extract named entities from text using spaCy
    
    Args:
        text (str): Text to extract entities from
        
    Returns:
        dict: Dictionary of entity types and their values
    """
    if not text:
        return {}
        
    # Limit text length for processing efficiency
    text = text[:10000] if len(text) > 10000 else text
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Extract entities
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    
    # Count occurrences of each entity
    for label, values in entities.items():
        counter = Counter(values)
        entities[label] = [{"text": text, "count": count} for text, count in counter.most_common(5)]
    
    return entities

def get_keywords(text, n=10):
    """
    Extract key phrases and words from text
    
    Args:
        text (str): Text to extract keywords from
        n (int): Number of keywords to return
        
    Returns:
        list: List of keywords
    """
    if not text:
        return []
        
    # Clean text
    text = clean_text(text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    try:
        nltk.data.find('stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    
    # Filter tokens
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and len(word) > 2]
    
    # Count frequency
    word_freq = Counter(filtered_tokens)
    
    # Return most common words
    return word_freq.most_common(n)
