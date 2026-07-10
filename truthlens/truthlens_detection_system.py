"""
TruthLens - Fake News Detection System with Trained ML Model
Uses pre-trained Passive Aggressive Classifier and TF-IDF Vectorizer

This module loads and uses your trained model for predictions.
"""

import joblib
import pickle
import os
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import langid


# ── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class Signal:
    """Represents a detection signal with analysis details"""
    label: str
    value: str
    note: str
    positive: bool


@dataclass
class Prediction:
    """Represents a prediction result with confidence and signals"""
    label: str  # "REAL" or "FAKE"
    confidence: float  # 0.0 to 1.0
    real_prob: float  # Probability of real news
    fake_prob: float  # Probability of fake news
    signals: List[Signal]


# ── Model Loading ────────────────────────────────────────────────────────────

class ModelLoader:
    """Handles loading trained model and vectorizer"""
    
    def __init__(self, model_path: str = "fakeNewsDS/Models/trained_model.pkl",
                 vectorizer_path: str = "fakeNewsDS/Models/tfidf_vectorizer.pkl",
                 metadata_path: str = "fakeNewsDS/Models/model_metadata.json"):
        """
        Initialize model loader
        
        Args:
            model_path: Path to trained model file
            vectorizer_path: Path to TF-IDF vectorizer file
            metadata_path: Path to model metadata JSON
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.metadata_path = metadata_path
        
        self.model = None
        self.vectorizer = None
        self.metadata = None
        
        self._load_model()
    
    def _load_model(self):
        """Load model, vectorizer, and metadata"""
        try:
            # Try loading with joblib first (more efficient)
            if os.path.exists(self.model_path):
                try:
                    self.model = joblib.load(self.model_path)
                except:
                    # Fall back to pickle if joblib fails
                    with open(self.model_path, 'rb') as f:
                        self.model = pickle.load(f)
                print(f"✅ Model loaded: {self.model_path}")
            else:
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            # Load vectorizer
            if os.path.exists(self.vectorizer_path):
                try:
                    self.vectorizer = joblib.load(self.vectorizer_path)
                except:
                    with open(self.vectorizer_path, 'rb') as f:
                        self.vectorizer = pickle.load(f)
                print(f"✅ Vectorizer loaded: {self.vectorizer_path}")
            else:
                raise FileNotFoundError(f"Vectorizer not found: {self.vectorizer_path}")
            
            # Load metadata if available
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ Metadata loaded: {self.metadata_path}")
            else:
                print(f"⚠️ Metadata not found: {self.metadata_path}")
                self.metadata = {}
        
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model and vectorizer are loaded"""
        return self.model is not None and self.vectorizer is not None


# ── Global Model Instance ────────────────────────────────────────────────────

# Try to load model with different path variations
_model_loader = None

def _initialize_model():
    """Initialize model loader with fallback paths"""
    global _model_loader
    
    if _model_loader is not None:
        return
    
    # Try different path variations
    possible_paths = [
        ("fakeNewsDS/Models/trained_model.pkl", "fakeNewsDS/Models/tfidf_vectorizer.pkl"),
        ("FAKENEWSDS/Models/trained_model.pkl", "FAKENEWSDS/Models/tfidf_vectorizer.pkl"),
        ("Models/trained_model.pkl", "Models/tfidf_vectorizer.pkl"),
        ("trained_model.pkl", "tfidf_vectorizer.pkl"),
        ("models/trained_model.pkl", "models/tfidf_vectorizer.pkl"),
    ]
    
    for model_path, vec_path in possible_paths:
        try:
            if os.path.exists(model_path) and os.path.exists(vec_path):
                _model_loader = ModelLoader(model_path, vec_path)
                return
        except:
            continue
    
    # If no model found, raise error
    raise RuntimeError(
        "Could not find trained model and vectorizer. "
        "Expected paths:\n"
        "  - fakeNewsDS/Models/trained_model.pkl\n"
        "  - fakeNewsDS/Models/tfidf_vectorizer.pkl"
    )


# ── Core Analysis Engine ─────────────────────────────────────────────────────

def analyze_article(text: str) -> Prediction:
    """
    Analyzes a news article using trained ML model.
    
    Args:
        text (str): The full text of the news article to analyze
        
    Returns:
        Prediction: Object containing label, confidence, probabilities, and signals
    """
    
    global _model_loader
    
    # Initialize model if not already done
    if _model_loader is None:
        _initialize_model()
    
    if not _model_loader.is_loaded():
        raise RuntimeError("Model not loaded successfully")
    
    # ── Vectorize Text ──────────────────────────────────────────────────────
    try:
        # Transform text using trained vectorizer
        text_features = _model_loader.vectorizer.transform([text])
    except Exception as e:
        raise RuntimeError(f"Error vectorizing text: {e}")
    
    # ── Make Prediction ─────────────────────────────────────────────────────
    try:
        # Get prediction
        prediction = _model_loader.model.predict(text_features)[0]
        
        # Get prediction probabilities if available
        if hasattr(_model_loader.model, 'predict_proba'):
            probabilities = _model_loader.model.predict_proba(text_features)[0]
            # probabilities are [fake_prob, real_prob] (0=fake, 1=real)
            fake_prob = float(probabilities[0])
            real_prob = float(probabilities[1])
        else:
            # If no predict_proba, estimate from decision function
            decision = _model_loader.model.decision_function(text_features)[0]
            # Convert decision score to probability (sigmoid)
            real_prob = 1 / (1 + np.exp(-decision))
            fake_prob = 1 - real_prob
        
        # Determine label
        label = "REAL" if prediction == 1 else "FAKE"
        confidence = max(real_prob, fake_prob)
        
    except Exception as e:
        raise RuntimeError(f"Error making prediction: {e}")
    
    # ── Generate Detection Signals ──────────────────────────────────────────
    signals = _generate_signals(text, label, confidence)
    
    return Prediction(
        label=label,
        confidence=confidence,
        real_prob=real_prob,
        fake_prob=fake_prob,
        signals=signals
    )


def _generate_signals(text: str, prediction: str, confidence: float) -> List[Signal]:
    """
    Generate human-readable signals explaining the prediction.
    
    Args:
        text: Article text
        prediction: "REAL" or "FAKE"
        confidence: Confidence score (0-1)
        
    Returns:
        List of Signal objects
    """
    
    # Basic text metrics
    lower_text = text.lower()
    words = text.strip().split()
    word_count = len(words)
    
    # Sentence analysis
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences) if sentences else 1
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # ── Signal 1: Article Length ────────────────────────────────────────────
    signal_1 = Signal(
        label="Article length",
        value=f"{word_count} words",
        note="Short articles often lack verifiable detail." if word_count < 80 
             else "Length appears substantive.",
        positive=word_count >= 80
    )
    
    # ── Signal 2: Source References ─────────────────────────────────────────
    credible_patterns = [
        r'\b(according to|reported by|said|stated|confirmed|announced)\b',
        r'\b(study|research|survey|data|statistics|percent|per cent)\b',
        r'\b(published|journal|university|institute|official|government)\b',
    ]
    
    credible_hits = 0
    for pattern in credible_patterns:
        matches = re.findall(pattern, lower_text, re.IGNORECASE)
        credible_hits += len(matches)
    
    signal_2 = Signal(
        label="Source references",
        value=f"{credible_hits} reference(s) found",
        note="References to officials, studies, or institutions detected." if credible_hits > 0 
             else "No verifiable source references found.",
        positive=credible_hits > 0
    )
    
    # ── Signal 3: Sensational Language ──────────────────────────────────────
    caps_words = len(re.findall(r'\b[A-Z]{4,}\b', text))
    exclamations = len(re.findall(r'!', text))
    
    signal_3 = Signal(
        label="Sensational language",
        value=f"{exclamations} exclamation(s) · {caps_words} ALL-CAPS word(s)",
        note="Sensational formatting is common in misleading content." if (exclamations > 1 or caps_words > 2)
             else "Tone appears measured.",
        positive=exclamations <= 1 and caps_words <= 2
    )
    
    # ── Signal 4: Sentence Structure ────────────────────────────────────────
    signal_4 = Signal(
        label="Avg. sentence length",
        value=f"{avg_sentence_length:.1f} words/sentence",
        note="Very short sentences may indicate emotionally charged writing." if avg_sentence_length < 10
             else "Sentence structure looks typical.",
        positive=avg_sentence_length >= 10
    )
    
    return [signal_1, signal_2, signal_3, signal_4]


# ── Utility Functions ───────────────────────────────────────────────────────

def validate_article(text: str) -> Tuple[bool, Optional[str]]:
    """
    Validates article input before analysis.
    
    Args:
        text (str): The article text to validate
        
    Returns:
        tuple: (is_valid: bool, error_message: Optional[str])
    """
    if not text or not text.strip():
        return False, "Please paste a news article before analysing."

    cleaned_text = re.sub(r"\s+", " ", text).strip()
    word_count = len(cleaned_text.split())
    if word_count < 20:
        return False, "Article is too short. Please paste at least a few complete sentences or paragraphs for accurate analysis."

    try:
        lang, _ = langid.classify(cleaned_text)
        if lang != "en":
            return False, "This system supports English news articles only. Please paste an English article for analysis."
    except Exception:
        return False, "This system supports English news articles only. Please paste an English article for analysis."
    
    return True, None


def format_confidence(confidence: float) -> str:
    """
    Formats confidence score as a percentage string.
    
    Args:
        confidence (float): Confidence value between 0 and 1
        
    Returns:
        str: Formatted percentage string
    """
    return f"{(confidence * 100):.1f}%"


def get_signal_emoji(signal: Signal) -> str:
    """Get emoji indicator for signal"""
    return "✓" if signal.positive else "✗"


# ── Model Information ───────────────────────────────────────────────────────

def get_model_info() -> dict:
    """Get model information from metadata or defaults"""
    global _model_loader
    
    if _model_loader is None:
        _initialize_model()
    
    if _model_loader.metadata:
        return _model_loader.metadata
    
    # Default model info
    return {
        "name": "TruthLens",
        "version": "1.0.0",
        "architecture": "Passive Aggressive Classifier with TF-IDF",
        "training_data": "ISOT Fake News Dataset",
        "input": "Raw news article text (plain text, no HTML)",
        "output": "Binary label (REAL / FAKE) with class probability scores",
        "features": "TF-IDF word n-grams",
        "language": "English only",
        "accuracy": "99.38%",
        "precision": "99.32%",
        "recall": "99.37%",
        "f1_score": "0.9935",
        "limitations": "Satire, opinion, and highly technical articles may be misclassified. Does not fact-check specific claims.",
    }

# Model information dictionary
MODEL_INFO = {
    "name": "TruthLens",
    "version": "1.0.0",
    "architecture": "Passive Aggressive Classifier with TF-IDF",
    "training_data": "ISOT Fake News Dataset",
    "input": "Raw news article text (plain text, no HTML)",
    "output": "Binary label (REAL / FAKE) with class probability scores",
    "features": "TF-IDF word n-grams",
    "language": "English only",
    "accuracy": "99.38%",
    "precision": "99.32%",
    "recall": "99.37%",
    "f1_score": "0.9935",
    "limitations": "Satire, opinion, and highly technical articles may be misclassified. Does not fact-check specific claims.",
}

HOW_IT_WORKS = [
    {
        "icon": "📝",
        "title": "Text Vectorization",
        "description": "Converts article text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency)."
    },
    {
        "icon": "🤖",
        "title": "ML Classification",
        "description": "Trained Passive Aggressive Classifier analyzes patterns learned from thousands of labeled articles."
    },
    {
        "icon": "📊",
        "title": "Probability Scoring",
        "description": "Returns confidence scores and probability distribution for real vs. fake classification."
    },
]

DISCLAIMER = (
    "This tool uses machine learning to estimate credibility and is not infallible. "
    "Always cross-check news with primary sources, established fact-checkers, and official reporting."
)


# ── Example Usage ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        # Example article for testing
        sample_article = """
        Scientists at the University of Cambridge have published a groundbreaking study 
        in the journal Nature, confirming that regular exercise reduces the risk of heart disease. 
        According to the research team, participants who engaged in 150 minutes of moderate activity 
        per week showed a 30% reduction in cardiovascular incidents. The study, conducted over 
        five years, analyzed data from 50,000 participants across multiple countries. 
        Dr. James Smith, director of the research institute, stated that these findings align 
        with previous research and provide additional evidence for public health recommendations.
        """
        
        # Validate input
        is_valid, error = validate_article(sample_article)
        if not is_valid:
            print(f"Error: {error}")
        else:
            # Analyze article
            result = analyze_article(sample_article)
            
            # Display results
            print(f"\n{'='*60}")
            print(f"ANALYSIS RESULT: {result.label}")
            print(f"{'='*60}")
            print(f"Confidence: {format_confidence(result.confidence)}")
            print(f"Real News Probability: {format_confidence(result.real_prob)}")
            print(f"Fake News Probability: {format_confidence(result.fake_prob)}")
            
            print(f"\n{'─'*60}")
            print("DETECTION SIGNALS:")
            print(f"{'─'*60}")
            
            for signal in result.signals:
                status = "✓" if signal.positive else "✗"
                print(f"\n{status} {signal.label}")
                print(f"   Value: {signal.value}")
                print(f"   Note: {signal.note}")
            
            print(f"\n{'='*60}\n")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure your model files are in the correct location:")
        print("  - fakeNewsDS/Models/trained_model.pkl")
        print("  - fakeNewsDS/Models/tfidf_vectorizer.pkl")
