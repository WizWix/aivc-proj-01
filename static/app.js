async function handleUpload(input, url, resultId) {
    const file = input.files[0];
    if (!file) return;

    const resultDiv = document.getElementById(resultId);
    const spinner = input.closest('.service-card').querySelector('.spinner');
    
    resultDiv.style.display = 'none';
    spinner.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        displayDecoratedResult(resultId, data);
        resultDiv.style.display = 'block';
    } catch (error) {
        resultDiv.innerHTML = `<p class="error">오류 발생: ${error.message}</p>`;
        resultDiv.style.display = 'block';
    } finally {
        spinner.style.display = 'none';
    }
}

function displayDecoratedResult(resultId, data) {
    const resultDiv = document.getElementById(resultId);
    resultDiv.innerHTML = '';

    if (data.error) {
        resultDiv.innerHTML = `<p class="error">${data.error}</p>`;
        return;
    }

    if (resultId === 'ocr-results') {
        if (data.results && data.results.length > 0) {
            const list = document.createElement('ul');
            list.className = 'result-list';
            data.results.forEach(item => {
                const li = document.createElement('li');
                li.innerHTML = `<strong>${item.text}</strong> (신뢰도: ${(item.confidence * 100).toFixed(1)}%)`;
                list.appendChild(li);
            });
            resultDiv.appendChild(list);
        } else {
            resultDiv.innerHTML = '<p>감지된 텍스트가 없습니다.</p>';
        }
    } else if (resultId === 'class-results') {
        resultDiv.innerHTML = `<div class="status-badge success">성공</div>
                               <p>분류 결과: <strong>${data.label}</strong> (index: ${data.predicted_class_index})</p>`;
    } else if (resultId === 'pose-results') {
        resultDiv.innerHTML = `<div class="status-badge success">성공</div>
                               <p>감지된 포즈 수: <strong>${data.detected_poses}</strong></p>
                               ${data.result_image ? `<img src="${data.result_image}" alt="Pose Result" style="width: 100%; border-radius: 0.5rem; margin-top: 0.5rem;">` : ''}`;
    } else if (resultId === 'object-results') {
        resultDiv.innerHTML = `<div class="status-badge success">성공</div>
                               <p>감지된 객체 수: <strong>${data.detected_objects}</strong></p>
                               ${data.result_image ? `<img src="${data.result_image}" alt="Object Result" style="width: 100%; border-radius: 0.5rem; margin-top: 0.5rem;">` : ''}`;
    }
}

async function handleFaceRecognition() {
    const file1 = document.getElementById('face-input-1').files[0];
    const file2 = document.getElementById('face-input-2').files[0];
    
    if (!file1 || !file2) {
        alert('두 이미지를 모두 선택해 주세요.');
        return;
    }

    const resultDiv = document.getElementById('face-results');
    const spinner = document.querySelector('#face-service .spinner');

    resultDiv.style.display = 'none';
    spinner.style.display = 'block';

    const formData = new FormData();
    formData.append('img1', file1);
    formData.append('img2', file2);

    try {
        const response = await fetch('/api/face-recognition', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.error) {
            resultDiv.innerHTML = `<p class="error">${data.error}</p>`;
        } else {
            const statusClass = data.is_same ? 'success' : 'failure';
            const statusText = data.is_same ? '예 (동일인물)' : '아니오 (다른 인물)';
            resultDiv.innerHTML = `
                <div class="result-box ${statusClass}">
                    <p>동일인 여부: <strong>${statusText}</strong></p>
                    <p>유사도 수준: <strong>${(data.similarity * 100).toFixed(2)}%</strong></p>
                </div>
            `;
        }
        resultDiv.style.display = 'block';
    } catch (error) {
        resultDiv.innerHTML = `<p class="error">오류 발생: ${error.message}</p>`;
        resultDiv.style.display = 'block';
    } finally {
        spinner.style.display = 'none';
    }
}
