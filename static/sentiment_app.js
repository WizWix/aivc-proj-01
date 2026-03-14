document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const textInput = document.getElementById('text-input');
    const resultPanel = document.getElementById('result-panel');
    const sentimentBadge = document.getElementById('sentiment-badge');
    const scoreText = document.getElementById('score-text');
    const scoreBar = document.getElementById('score-bar');
    const analysisSummary = document.getElementById('analysis-summary');
    const errorMsg = document.getElementById('error-msg');
    const spinner = document.getElementById('loading-spinner');

    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) {
            showError('분석할 텍스트를 입력해 주세요.');
            return;
        }

        hideError();
        showLoading();
        resultPanel.style.display = 'none';

        try {
            const response = await fetch('/sentiment-analysis/api/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            });

            const data = await response.json();

            if (data.status === 'success') {
                displayResult(data);
            } else {
                showError(data.detail || data.message || '분석 중 오류가 발생했습니다.');
            }
        } catch (error) {
            console.error('Error:', error);
            showError('서버와 통신하는 중 오류가 발생했습니다.');
        } finally {
            hideLoading();
        }
    });

    function displayResult(data) {
        resultPanel.style.display = 'block';
        
        const label = data.label.toLowerCase();
        let labelText = '';
        let badgeClass = '';
        let message = '';

        if (label === 'positive') {
            labelText = '긍정 (Positive)';
            badgeClass = 'label-positive';
            message = '입력하신 텍스트에서 긍정적인 기운이 느껴집니다! 😊';
        } else if (label === 'neutral') {
            labelText = '중립 (Neutral)';
            badgeClass = 'label-neutral';
            message = '준립적이거나 객관적인 내용으로 분석되었습니다. 😐';
        } else {
            labelText = '부정 (Negative)';
            badgeClass = 'label-negative';
            message = '부정적이거나 다소 우울한 감정이 감지되었습니다. 😟';
        }

        sentimentBadge.textContent = labelText;
        sentimentBadge.className = 'sentiment-badge ' + badgeClass;
        
        const percentage = (data.score * 100).toFixed(1);
        scoreText.textContent = percentage + '%';
        
        scoreBar.style.width = percentage + '%';
        scoreBar.style.backgroundColor = getComputedStyle(sentimentBadge).color;
        
        analysisSummary.textContent = message;
        
        // Scroll to result
        resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function showLoading() {
        spinner.style.display = 'block';
        analyzeBtn.disabled = true;
    }

    function hideLoading() {
        spinner.style.display = 'none';
        analyzeBtn.disabled = false;
    }

    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.style.display = 'block';
    }

    function hideError() {
        errorMsg.style.display = 'none';
    }
});
