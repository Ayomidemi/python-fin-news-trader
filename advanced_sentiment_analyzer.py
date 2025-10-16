"""
Advanced Sentiment Analysis using FinBERT and other ML models
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Structured sentiment analysis result"""
    text: str
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float
    scores: Dict[str, float]  # Detailed scores
    model_used: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis using multiple models"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.vectorizer = None
        self.classifier = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize sentiment analysis models"""
        try:
            # FinBERT - Financial sentiment analysis
            logger.info("Loading FinBERT model...")
            self.models['finbert'] = AutoModelForSequenceClassification.from_pretrained(
                'ProsusAI/finbert'
            )
            self.tokenizers['finbert'] = AutoTokenizer.from_pretrained(
                'ProsusAI/finbert'
            )
            
            # RoBERTa for general sentiment
            logger.info("Loading RoBERTa model...")
            self.models['roberta'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # BERT for general sentiment
            logger.info("Loading BERT model...")
            self.models['bert'] = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                return_all_scores=True
            )
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            # Fallback to basic models
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load fallback models if main models fail"""
        try:
            logger.info("Loading fallback models...")
            self.models['vader'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            logger.info("Fallback models loaded")
        except Exception as e:
            logger.error(f"Error loading fallback models: {str(e)}")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stop words
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def analyze_with_finbert(self, text: str) -> SentimentResult:
        """Analyze sentiment using FinBERT"""
        try:
            if 'finbert' not in self.models:
                raise Exception("FinBERT model not available")
            
            # Tokenize input
            inputs = self.tokenizers['finbert'](
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self.models['finbert'](**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Map labels (FinBERT uses: positive, negative, neutral)
            labels = ['negative', 'neutral', 'positive']
            scores = predictions[0].tolist()
            
            # Get predicted label and confidence
            predicted_idx = np.argmax(scores)
            predicted_label = labels[predicted_idx]
            confidence = float(scores[predicted_idx])
            
            return SentimentResult(
                text=text,
                sentiment=predicted_label,
                confidence=confidence,
                scores=dict(zip(labels, scores)),
                model_used='finbert',
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in FinBERT analysis: {str(e)}")
            raise
    
    def analyze_with_roberta(self, text: str) -> SentimentResult:
        """Analyze sentiment using RoBERTa"""
        try:
            if 'roberta' not in self.models:
                raise Exception("RoBERTa model not available")
            
            # Get predictions
            results = self.models['roberta'](text)
            
            # Extract scores
            scores = {}
            for result in results[0]:
                scores[result['label']] = result['score']
            
            # Get predicted label and confidence
            predicted_label = max(scores, key=scores.get)
            confidence = scores[predicted_label]
            
            # Map to standard labels
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive'
            }
            sentiment = label_mapping.get(predicted_label, predicted_label)
            
            return SentimentResult(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                scores=scores,
                model_used='roberta',
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in RoBERTa analysis: {str(e)}")
            raise
    
    def analyze_with_bert(self, text: str) -> SentimentResult:
        """Analyze sentiment using BERT"""
        try:
            if 'bert' not in self.models:
                raise Exception("BERT model not available")
            
            # Get predictions
            results = self.models['bert'](text)
            
            # Extract scores
            scores = {}
            for result in results[0]:
                scores[result['label']] = result['score']
            
            # Get predicted label and confidence
            predicted_label = max(scores, key=scores.get)
            confidence = scores[predicted_label]
            
            # Map to standard labels
            label_mapping = {
                '1 star': 'negative',
                '2 stars': 'negative',
                '3 stars': 'neutral',
                '4 stars': 'positive',
                '5 stars': 'positive'
            }
            sentiment = label_mapping.get(predicted_label, predicted_label)
            
            return SentimentResult(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                scores=scores,
                model_used='bert',
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in BERT analysis: {str(e)}")
            raise
    
    def ensemble_analysis(self, text: str, models: List[str] = None) -> SentimentResult:
        """Ensemble analysis using multiple models"""
        if models is None:
            models = ['finbert', 'roberta', 'bert']
        
        results = []
        for model_name in models:
            try:
                if model_name == 'finbert':
                    result = self.analyze_with_finbert(text)
                elif model_name == 'roberta':
                    result = self.analyze_with_roberta(text)
                elif model_name == 'bert':
                    result = self.analyze_with_bert(text)
                else:
                    continue
                
                results.append(result)
            except Exception as e:
                logger.warning(f"Error with {model_name}: {str(e)}")
                continue
        
        if not results:
            raise Exception("No models available for analysis")
        
        # Weighted ensemble (FinBERT gets higher weight for financial text)
        weights = {'finbert': 0.5, 'roberta': 0.3, 'bert': 0.2}
        
        # Calculate weighted scores
        sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_weight = 0
        
        for result in results:
            weight = weights.get(result.model_used, 0.1)
            total_weight += weight
            
            for sentiment, score in result.scores.items():
                if sentiment in sentiment_scores:
                    sentiment_scores[sentiment] += score * weight
        
        # Normalize scores
        if total_weight > 0:
            for sentiment in sentiment_scores:
                sentiment_scores[sentiment] /= total_weight
        
        # Get predicted sentiment
        predicted_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        confidence = sentiment_scores[predicted_sentiment]
        
        return SentimentResult(
            text=text,
            sentiment=predicted_sentiment,
            confidence=confidence,
            scores=sentiment_scores,
            model_used='ensemble',
            timestamp=datetime.now(),
            metadata={'individual_results': results}
        )
    
    def analyze_batch(self, texts: List[str], model: str = 'ensemble') -> List[SentimentResult]:
        """Analyze multiple texts in batch"""
        results = []
        for text in texts:
            try:
                if model == 'ensemble':
                    result = self.ensemble_analysis(text)
                elif model == 'finbert':
                    result = self.analyze_with_finbert(text)
                elif model == 'roberta':
                    result = self.analyze_with_roberta(text)
                elif model == 'bert':
                    result = self.analyze_with_bert(text)
                else:
                    raise ValueError(f"Unknown model: {model}")
                
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing text: {str(e)}")
                # Create error result
                results.append(SentimentResult(
                    text=text,
                    sentiment='neutral',
                    confidence=0.0,
                    scores={'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                    model_used=model,
                    timestamp=datetime.now(),
                    metadata={'error': str(e)}
                ))
        
        return results
    
    def train_custom_model(self, training_data: List[Tuple[str, str]], model_type: str = 'logistic'):
        """Train a custom sentiment model on financial data"""
        try:
            # Prepare data
            texts = [self.preprocess_text(text) for text, _ in training_data]
            labels = [label for _, label in training_data]
            
            # Vectorize texts
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X = self.vectorizer.fit_transform(texts)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.2, random_state=42
            )
            
            # Train model
            if model_type == 'logistic':
                self.classifier = LogisticRegression(random_state=42)
            elif model_type == 'random_forest':
                self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.classifier.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Custom model trained with accuracy: {accuracy:.3f}")
            logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
            
            # Save model
            model_path = f"models/custom_sentiment_{model_type}.joblib"
            os.makedirs("models", exist_ok=True)
            joblib.dump({
                'classifier': self.classifier,
                'vectorizer': self.vectorizer
            }, model_path)
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error training custom model: {str(e)}")
            raise
    
    def load_custom_model(self, model_path: str):
        """Load a trained custom model"""
        try:
            model_data = joblib.load(model_path)
            self.classifier = model_data['classifier']
            self.vectorizer = model_data['vectorizer']
            logger.info(f"Custom model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading custom model: {str(e)}")
            raise
    
    def analyze_with_custom_model(self, text: str) -> SentimentResult:
        """Analyze sentiment using custom trained model"""
        if self.classifier is None or self.vectorizer is None:
            raise Exception("Custom model not loaded")
        
        try:
            # Preprocess and vectorize
            processed_text = self.preprocess_text(text)
            X = self.vectorizer.transform([processed_text])
            
            # Get prediction
            prediction = self.classifier.predict(X)[0]
            confidence = self.classifier.predict_proba(X)[0].max()
            
            return SentimentResult(
                text=text,
                sentiment=prediction,
                confidence=confidence,
                scores={prediction: confidence},
                model_used='custom',
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in custom model analysis: {str(e)}")
            raise

# Global analyzer instance
sentiment_analyzer = AdvancedSentimentAnalyzer()

# Convenience functions
def analyze_sentiment(text: str, model: str = 'ensemble') -> SentimentResult:
    """Analyze sentiment of a single text"""
    return sentiment_analyzer.ensemble_analysis(text) if model == 'ensemble' else getattr(sentiment_analyzer, f'analyze_with_{model}')(text)

def analyze_sentiments(texts: List[str], model: str = 'ensemble') -> List[SentimentResult]:
    """Analyze sentiment of multiple texts"""
    return sentiment_analyzer.analyze_batch(texts, model)

def get_sentiment_score(text: str, model: str = 'ensemble') -> float:
    """Get sentiment score (-1 to 1)"""
    result = analyze_sentiment(text, model)
    if result.sentiment == 'positive':
        return result.confidence
    elif result.sentiment == 'negative':
        return -result.confidence
    else:
        return 0.0
