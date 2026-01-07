function collectModelSelections() {
  const selectedModels = [];
  const modelThresholds = {};

  document.querySelectorAll('input[name="analysis"]:checked').forEach((checkbox) => {
    const model = checkbox.value;
    selectedModels.push(model);
    const thresholdInput = document.getElementById(`${model}-threshold`);
    const thresholdValue = parseFloat(thresholdInput?.value) || 0.5;
    modelThresholds[model] = thresholdValue;
  });

  if (!selectedModels.length) {
    alert('กรุณาเลือกโมเดลอย่างน้อย 1 โมเดลก่อนอัปโหลดค่ะ');
    return null;
  }

  return {
    selectedModels,
    modelThresholds
  };
}

function collectUniqueLabels(source) {
  const labels = [];
  const seen = new Set();

  const addLabel = (value) => {
    if (typeof value !== 'string') {
      return;
    }
    const trimmed = value.trim();
    if (!trimmed || seen.has(trimmed)) {
      return;
    }
    seen.add(trimmed);
    labels.push(trimmed);
  };

  const traverse = (item) => {
    if (!item) return;
    if (Array.isArray(item)) {
      item.forEach(traverse);
      return;
    }
    if (typeof item === 'string') {
      addLabel(item);
      return;
    }
    if (typeof item === 'object') {
      if (item.label) {
        addLabel(item.label);
      }
      if (Array.isArray(item.detections)) {
        traverse(item.detections);
      }
    }
  };

  traverse(source);
  return labels;
}

function resetMediaDisplay() {
  const imagePreview = document.getElementById('imagePreview');
  const processedImage = document.getElementById('processedImage');
  const videoPreview = document.getElementById('videoPreview');
  const processedVideo = document.getElementById('processedVideo');
  const blurredVideo = document.getElementById('blurredVideo');
  const containers = [
    'imageOriginalContainer',
    'imageProcessedContainer',
    'videoOriginalContainer',
    'videoProcessedContainer',
    'videoBlurredContainer',
  ];

  [imagePreview, processedImage].forEach((img) => {
    if (!img) return;
    img.style.display = 'none';
    img.src = '';
  });

  [videoPreview, processedVideo, blurredVideo].forEach((video) => {
    if (!video) return;
    if (video.dataset && video.dataset.objectUrl) {
      URL.revokeObjectURL(video.dataset.objectUrl);
      delete video.dataset.objectUrl;
    }
    video.pause();
    video.removeAttribute('src');
    video.load();
    video.style.display = 'none';
  });

  containers.forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.style.display = 'none';
    }
  });
}

function setLoadingState(isLoading) {
  const loadingSpinner = document.getElementById('loadingSpinner');
  if (!loadingSpinner) return;
  loadingSpinner.style.display = isLoading ? 'block' : 'none';
  if (isLoading) {
    setResultMessage('', 'info');
  }
}

function setResultMessage(message, type = 'info') {
  const resultText = document.getElementById('resultText');
  if (!resultText) {
    return;
  }
  resultText.textContent = message || '';
  const colors = {
    success: '#2ecc71',
    error: '#e74c3c',
    info: '#f1c40f',
  };
  resultText.style.color = colors[type] || '#ecf0f1';
}

async function uploadImage() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';

  input.onchange = async () => {
    const file = input.files?. [0];
    if (!file) return;

    const selection = collectModelSelections();
    if (!selection) {
      return;
    }

    const {
      selectedModels,
      modelThresholds
    } = selection;

    const formData = new FormData();
    formData.append('image', file);
    formData.append('analysis_types', JSON.stringify(selectedModels));
    formData.append('thresholds', JSON.stringify(modelThresholds));

    resetMediaDisplay();
    setLoadingState(true);

    const imagePreview = document.getElementById('imagePreview');
    const processedImage = document.getElementById('processedImage');

    const reader = new FileReader();
    reader.onload = () => {
      if (imagePreview) {
        imagePreview.src = reader.result;
        imagePreview.style.display = 'block';
        const container = document.getElementById('imageOriginalContainer');
        if (container) {
          container.style.display = 'block';
        }
      }
    };
    reader.readAsDataURL(file);

    try {
      const response = await fetch(`${window.API_BASE_URL}/analyze-image`, {
        method: 'POST',
        headers: {
          'x-api-key': window.API_KEY,
        },
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        const detections = Array.isArray(data.detections) ? data.detections : [];
        const labels = collectUniqueLabels(detections);
        const status =
          (typeof data.status === 'string' && data.status.toLowerCase()) ||
          (labels.length ? 'failed' : 'passed');
        const labelsSuffix = labels.length ? ` | รายการตรวจพบ: ${labels.join(', ')}` : '';
        const noLabelsSuffix = ' | ไม่พบวัตถุที่ตรงตามเงื่อนไข';

        if (status === 'failed') {
          setResultMessage(
            `ผลลัพธ์: ไม่ผ่านการทดสอบ${labelsSuffix || noLabelsSuffix}`,
            'error',
          );
        } else {
          setResultMessage(`ผลลัพธ์: ผ่านการทดสอบ${noLabelsSuffix}`, 'success');
        }

        if (data.processed_image_url && processedImage) {
          processedImage.src = data.processed_image_url;
          processedImage.style.display = 'block';
          const container = document.getElementById('imageProcessedContainer');
          if (container) {
            container.style.display = 'block';
          }
        }
      } else {
        setResultMessage(`ข้อผิดพลาด: ${data.error || 'เกิดข้อผิดพลาด'}`, 'error');
      }
    } catch (error) {
      setResultMessage('ข้อผิดพลาด: ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์', 'error');
    } finally {
      setLoadingState(false);
    }
  };

  input.click();
}

async function uploadVideo() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'video/*';

  input.onchange = async () => {
    const file = input.files?. [0];
    if (!file) return;

    const selection = collectModelSelections();
    if (!selection) {
      return;
    }

    const {
      selectedModels,
      modelThresholds
    } = selection;

    const formData = new FormData();
    formData.append('media', file);
    formData.append('analysis_types', JSON.stringify(selectedModels));
    formData.append('thresholds', JSON.stringify(modelThresholds));

    resetMediaDisplay();
    setLoadingState(true);

    const videoPreview = document.getElementById('videoPreview');
    const processedVideo = document.getElementById('processedVideo');
    const blurredVideo = document.getElementById('blurredVideo');

    if (videoPreview) {
      const objectUrl = URL.createObjectURL(file);
      videoPreview.src = objectUrl;
      videoPreview.dataset.objectUrl = objectUrl;
      videoPreview.style.display = 'block';
      const container = document.getElementById('videoOriginalContainer');
      if (container) {
        container.style.display = 'block';
      }
    }

    try {
      const response = await fetch(`${window.API_BASE_URL}/analyze-video`, {
        method: 'POST',
        headers: {
          'x-api-key': window.API_KEY,
        },
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        const status = (data.status || '').toLowerCase();
        const summaryLabels = Array.isArray(data.summary_labels) ?
          data.summary_labels :
          Object.keys(data.summary || {});
        const labels = collectUniqueLabels(
          summaryLabels.length ? summaryLabels : data.detections,
        );
        const resolvedStatus =
          status || (labels.length ? 'failed' : 'passed');
        const labelsSuffix = labels.length ? ` | รายการตรวจพบ: ${labels.join(', ')}` : '';
        const noLabelsSuffix = ' | ไม่พบวัตถุที่ตรงตามเงื่อนไข';

        if (resolvedStatus === 'failed') {
          setResultMessage(
            `ผลลัพธ์: ไม่ผ่านการทดสอบ${labelsSuffix || noLabelsSuffix}`,
            'error',
          );
        } else {
          const successSuffix = labelsSuffix || noLabelsSuffix;
          setResultMessage(`ผลลัพธ์: ผ่านการทดสอบ${successSuffix}`, 'success');
        }

        if (data.processed_video_url && processedVideo) {
          processedVideo.src = `${data.processed_video_url}?t=${Date.now()}`;
          processedVideo.style.display = 'block';
          processedVideo.load();
          const container = document.getElementById('videoProcessedContainer');
          if (container) {
            container.style.display = 'block';
          }
        }

        if (data.processed_blurred_video_url && blurredVideo) {
          blurredVideo.src = `${data.processed_blurred_video_url}?t=${Date.now()}`;
          blurredVideo.style.display = 'block';
          blurredVideo.load();
          const container = document.getElementById('videoBlurredContainer');
          if (container) {
            container.style.display = 'block';
          }
        }
      } else {
        setResultMessage(`ข้อผิดพลาด: ${data.error || 'เกิดข้อผิดพลาด'}`, 'error');
      }
    } catch (error) {
      setResultMessage('ข้อผิดพลาด: ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์', 'error');
    } finally {
      setLoadingState(false);
    }
  };

  input.click();
}

function downloadManual() {
  const url = `${window.API_BASE_URL}/manual`;
  window.location.href = url;
}

function toggleAdvanced() {
  const advancedSection = document.getElementById('advanced-settings');
  if (!advancedSection) return;
  advancedSection.style.display = advancedSection.style.display === 'none' ? 'block' : 'none';
}

window.uploadImage = uploadImage;
window.uploadVideo = uploadVideo;
window.downloadManual = downloadManual;
window.toggleAdvanced = toggleAdvanced;