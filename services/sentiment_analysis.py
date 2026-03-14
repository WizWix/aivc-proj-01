import os
import logging
import re
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Models:
# 1. Korean: nlp04/korean_sentiment_analysis_kcelectra (KcELECTRA fine-tuned on 11 emotions)
# 2. English: cardiffnlp/twitter-roberta-base-sentiment-latest (RoBERTa 3-class)

KO_MODEL = "nlp04/korean_sentiment_analysis_kcelectra"
EN_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

_ko_classifier = None
_en_classifier = None

def get_ko_classifier():
    global _ko_classifier
    if _ko_classifier is None:
        logging.info(f"Initializing Korean sentiment model: {KO_MODEL}...")
        try:
            _ko_classifier = pipeline("sentiment-analysis", model=KO_MODEL)
        except Exception as e:
            logging.error(f"Failed to load Korean model: {e}")
            raise e
    return _ko_classifier

def get_en_classifier():
    global _en_classifier
    if _en_classifier is None:
        logging.info(f"Initializing English sentiment model: {EN_MODEL}...")
        try:
            _en_classifier = pipeline("sentiment-analysis", model=EN_MODEL)
        except Exception as e:
            logging.error(f"Failed to load English model: {e}")
            raise e
    return _en_classifier

def is_korean(text):
    return bool(re.search("[가-힣]", text))

def analyze_sentiment(text: str):
    if not text or len(text.strip()) == 0:
        return {"label": "error", "score": 0.0, "message": "텍스트가 비어 있습니다."}

    try:
        if is_korean(text):
            classifier = get_ko_classifier()
            results = classifier(text)
            res = results[0]
            
            # nlp04/korean_sentiment_analysis_kcelectra mappings:
            # Positive: 기쁨(행복한), 고마운, 설레는(기대하는), 사랑하는, 즐거운(신나는)
            # Neutral: 일상적인, 생각이 많은
            # Negative: 슬픔(우울한), 힘듦(지침), 짜증남, 걱정스러운(불안한)
            
            label_name = res['label']
            
            # If the model returns LABEL_X instead of the name, we map it
            if label_name.startswith('LABEL_'):
                idx = int(label_name.split('_')[-1])
                # Mappings based on model config id2label
                if idx in [0, 1, 2, 3, 4]: label = "positive"
                elif idx in [5, 6]: label = "neutral"
                else: label = "negative"
            else:
                # Direct string comparison
                pos_list = ['기쁨(행복한)', '고마운', '설레는(기대하는)', '사랑하는', '즐거운(신나는)']
                neu_list = ['일상적인', '생각이 많은']
                if label_name in pos_list:
                    label = "positive"
                elif label_name in neu_list:
                    label = "neutral"
                else:
                    label = "negative"
                
            return {
                "label": label,
                "score": float(res['score']),
                "status": "success",
                "model": "KcELECTRA (11 Emotions)",
                "raw_label": label_name
            }
        else:
            classifier = get_en_classifier()
            results = classifier(text)
            res = results[0]
            
            label_map = {
                "negative": "negative", "neutral": "neutral", "positive": "positive",
                "LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"
            }
            label = label_map.get(res['label'].lower(), res['label'].lower())
            
            return {
                "label": label,
                "score": float(res['score']),
                "status": "success",
                "model": "Twitter-RoBERTa"
            }
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        return {"label": "error", "score": 0.0, "message": str(e)}

if __name__ == "__main__":
    test_texts = [
        "이 프로젝트 정말 마음에 들어요!",
        "만우절",
        "4월 1일",
        "주말",
        "정말 별로네요. 실망했습니다."
    ]
    
    for text in test_texts:
        result = analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"Result: {result}\n")
