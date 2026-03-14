from services import sentiment_analysis
test_texts = ["이 프로젝트 정말 마음에 들어요!", "만우절", "4월 1일", "주말", "정말 별로네요."]
for text in test_texts:
    print(f"Text: {text} -> {sentiment_analysis.analyze_sentiment(text)}")
