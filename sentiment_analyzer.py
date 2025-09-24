import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import streamlit as st

class SentimentAnalyzer:
    """
    Sentiment analysis class supporting multiple algorithms
    """
    
    def __init__(self):
        self.vader_analyzer = None
        self._initialize_vader()
    
    def _initialize_vader(self):
        """Initialize VADER sentiment analyzer"""
        try:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            st.warning(f"Could not initialize VADER analyzer: {e}")
            self.vader_analyzer = None
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text for sentiment analysis
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return str(text)
        
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle empty strings
        if not text:
            return "No content"
        
        return text
    
    def analyze_with_textblob(self, text, confidence_threshold=0.1):
        """
        Analyze sentiment using TextBlob
        
        Args:
            text (str): Text to analyze
            confidence_threshold (float): Minimum confidence for classification
            
        Returns:
            dict: Sentiment analysis results
        """
        try:
            processed_text = self.preprocess_text(text)
            blob = TextBlob(processed_text)
            
            # Get polarity (-1 to 1) and subjectivity (0 to 1)
            polarity = float(blob.sentiment.polarity)
            subjectivity = float(blob.sentiment.subjectivity)
            
            # Convert polarity to sentiment categories
            if polarity > confidence_threshold:
                sentiment = 'positive'
            elif polarity < -confidence_threshold:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Calculate confidence based on absolute polarity
            confidence = abs(polarity)
            
            # Create normalized scores (0 to 1)
            if polarity >= 0:
                positive_score = (polarity + 1) / 2
                negative_score = 0.0
            else:
                positive_score = 0.0
                negative_score = (-polarity + 1) / 2
            
            neutral_score = 1 - (positive_score + negative_score)
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'scores': {
                    'positive': positive_score,
                    'neutral': neutral_score,
                    'negative': negative_score
                },
                'raw_polarity': polarity,
                'subjectivity': subjectivity,
                'method': 'textblob'
            }
            
        except Exception as e:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
                'raw_polarity': 0.0,
                'subjectivity': 0.0,
                'method': 'textblob',
                'error': str(e)
            }
    
    def analyze_with_vader(self, text, confidence_threshold=0.1):
        """
        Analyze sentiment using VADER
        
        Args:
            text (str): Text to analyze
            confidence_threshold (float): Minimum confidence for classification
            
        Returns:
            dict: Sentiment analysis results
        """
        if self.vader_analyzer is None:
            # Fallback to TextBlob if VADER is not available
            return self.analyze_with_textblob(text, confidence_threshold)
        
        try:
            processed_text = self.preprocess_text(text)
            scores = self.vader_analyzer.polarity_scores(processed_text)
            
            # Get compound score (-1 to 1)
            compound = scores['compound']
            
            # Determine sentiment based on compound score
            if compound >= confidence_threshold:
                sentiment = 'positive'
            elif compound <= -confidence_threshold:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Use compound score as confidence
            confidence = abs(compound)
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'scores': {
                    'positive': scores['pos'],
                    'neutral': scores['neu'],
                    'negative': scores['neg']
                },
                'raw_compound': compound,
                'method': 'vader'
            }
            
        except Exception as e:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
                'raw_compound': 0.0,
                'method': 'vader',
                'error': str(e)
            }
    
    def analyze_single(self, text, method='textblob', confidence_threshold=0.1):
        """
        Analyze sentiment for a single text
        
        Args:
            text (str): Text to analyze
            method (str): Analysis method ('textblob' or 'vader')
            confidence_threshold (float): Minimum confidence for classification
            
        Returns:
            dict: Sentiment analysis results with comment text
        """
        if method.lower() == 'vader':
            result = self.analyze_with_vader(text, confidence_threshold)
        else:
            result = self.analyze_with_textblob(text, confidence_threshold)
        
        result['comment'] = text
        return result
    
    def analyze_batch(self, texts, method='textblob', confidence_threshold=0.1):
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts (list): List of texts to analyze
            method (str): Analysis method ('textblob' or 'vader')
            confidence_threshold (float): Minimum confidence for classification
            
        Returns:
            list: List of sentiment analysis results
        """
        results = []
        
        # Use progress bar for large batches
        progress_bar = None
        status_text = None
        if len(texts) > 10:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for i, text in enumerate(texts):
            result = self.analyze_single(text, method, confidence_threshold)
            results.append(result)
            
            # Update progress for large batches
            if len(texts) > 10 and progress_bar is not None and status_text is not None:
                progress = (i + 1) / len(texts)
                progress_bar.progress(progress)
                status_text.text(f'Processing comment {i + 1} of {len(texts)}')
        
        # Clean up progress indicators
        if len(texts) > 10 and progress_bar is not None and status_text is not None:
            progress_bar.empty()
            status_text.empty()
        
        return results
    
    def get_sentiment_statistics(self, results):
        """
        Generate statistics from sentiment analysis results
        
        Args:
            results (list): List of sentiment analysis results
            
        Returns:
            dict: Statistics summary
        """
        if not results:
            return {}
        
        sentiments = [r['sentiment'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        stats = {
            'total_comments': len(results),
            'positive_count': sentiments.count('positive'),
            'neutral_count': sentiments.count('neutral'),
            'negative_count': sentiments.count('negative'),
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
        
        # Calculate percentages
        total = stats['total_comments']
        stats['positive_percentage'] = (stats['positive_count'] / total) * 100
        stats['neutral_percentage'] = (stats['neutral_count'] / total) * 100
        stats['negative_percentage'] = (stats['negative_count'] / total) * 100
        
        return stats
