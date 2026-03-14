document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const blurSlider = document.getElementById('blur-slider');
    const blurValue = document.getElementById('blur-value');
    const processBtn = document.getElementById('process-btn');
    const resultImg = document.getElementById('result-img');
    const placeholder = document.getElementById('placeholder-text');
    const spinner = document.getElementById('loading-spinner');
    const errorMsg = document.getElementById('error-msg');
    const downloadBtn = document.getElementById('download-btn');
    const dropZone = document.getElementById('drop-zone');

    blurSlider.addEventListener('input', function() {
        blurValue.innerText = this.value;
    });

    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const fileName = this.files[0].name;
            document.getElementById('upload-prompt').innerHTML = `
                <p style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem; color: #4ade80;">파일 선택됨</p>
                <p style="font-size: 0.9rem; color: var(--text-muted);">${fileName}</p>
            `;
        }
    });

    // Drag and drop handling
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        // Trigger change event
        const event = new Event('change');
        fileInput.dispatchEvent(event);
    });

    processBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) {
            alert('이미지를 먼저 선택하세요.');
            return;
        }

        errorMsg.style.display = 'none';
        spinner.style.display = 'block';
        resultImg.style.display = 'none';
        placeholder.style.display = 'none';
        downloadBtn.style.display = 'none';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('blur_intensity', blurSlider.value);

        try {
            const response = await fetch('/selfie-segmentation/api/process', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || '처리 중 오류가 발생했습니다.');
            }

            const data = await response.json();
            resultImg.src = data.result_image;
            resultImg.style.display = 'block';
            downloadBtn.style.display = 'block';
        } catch (error) {
            errorMsg.innerText = error.message;
            errorMsg.style.display = 'block';
            placeholder.style.display = 'block';
        } finally {
            spinner.style.display = 'none';
        }
    });

    downloadBtn.addEventListener('click', () => {
        const link = document.createElement('a');
        link.href = resultImg.src;
        link.download = 'selfie_blurred.png';
        link.click();
    });
});
