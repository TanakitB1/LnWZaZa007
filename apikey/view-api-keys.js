(function syncTokenFromUrl() {

    const params = new URLSearchParams(window.location.search);

    const token = params.get('token');

    if (token) {

        localStorage.setItem('token', token);

        const newUrl = window.location.origin + window.location.pathname;

        window.history.replaceState({}, document.title, newUrl);

    }

})();



function escapeHtml(value) {

    if (value === null || value === undefined) {

        return '';

    }

    return String(value)

        .replace(/&/g, '&amp;')

        .replace(/</g, '&lt;')

        .replace(/>/g, '&gt;')

        .replace(/"/g, '&quot;')

        .replace(/'/g, '&#39;');

}



function parseDate(value) {

    if (!value) {

        return null;

    }

    if (value instanceof Date && !Number.isNaN(value.getTime())) {

        return value;

    }

    if (typeof value === 'string') {

        let candidate = value.trim();
        if (!candidate) {
            return null;
        }

        if (!candidate.includes('T') && candidate.includes(' ')) {

            candidate = candidate.replace(' ', 'T');

        }

        const hasTimezone =
            /([zZ]|[+-]\d{2}:?\d{2})$/.test(candidate);

        let parsed;
        if (hasTimezone) {
            parsed = new Date(candidate);
        } else {
            const isoLike = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?$/.test(candidate);
            parsed = new Date(isoLike ? `${candidate}Z` : candidate);
            if (Number.isNaN(parsed.getTime()) && !isoLike) {
                parsed = new Date(`${candidate}Z`);
            }
        }

        if (parsed && !Number.isNaN(parsed.getTime())) {

            return parsed;

        }

    }

    return null;

}



function formatDateTime(value) {
    const parsed = parseDate(value);
    if (!parsed) {
        return value || '—';
    }

    try {
        // แปลงเวลา UTC ให้เป็นเวลาไทย
        return parsed.toLocaleString('th-TH', {
            dateStyle: 'medium',
            timeStyle: 'short',
            timeZone: 'Asia/Bangkok'
        });
    } catch (err) {
        return parsed.toISOString();
    }
}



function formatQuota(quota) {

    if (quota === -1) {

        return 'ไม่จำกัดการใช้งาน';

    }

    if (quota === null || quota === undefined) {

        return '—';

    }

    return quota;

}



function formatAnalysisTypes(types) {

    if (!Array.isArray(types) || types.length === 0) {

        return '—';

    }

    return types.join(', ');

}



function formatThresholds(thresholds) {

    if (!thresholds || typeof thresholds !== 'object' || Array.isArray(thresholds)) {

        return '—';

    }

    const entries = Object.entries(thresholds);

    if (!entries.length) {

        return '—';

    }

    return entries

        .map(([key, value]) => {

            const numeric = Number.parseFloat(value);

            if (Number.isFinite(numeric)) {

                return `${key}: ${numeric.toFixed(2)}`;

            }

            return `${key}: ${value}`;

        })

        .join(', ');

}



function formatMediaAccess(access) {

    if (!Array.isArray(access) || access.length === 0) {

        return '—';

    }

    const labels = {
        image: 'Image',
        video: 'Video',
    };

    return access

        .map((item) => labels[item] || item)

        .join(', ');

}



function formatOutputModes(modes) {

    if (!Array.isArray(modes) || modes.length === 0) {

        return '—';

    }

    const labels = {
        blur: 'Blur',
        bbox: 'Bounding Box',
    };

    return modes

        .map((mode) => labels[mode] || mode)

        .join(', ');

}



function formatStatusBadge(status) {
    const normalized = (status || '').toLowerCase();

    const labels = {
        passed: 'ผ่าน',
        failed: 'ไม่ผ่าน',
        error: 'ข้อผิดพลาด',
    };

    const safeClass = normalized.replace(/[^a-z0-9-]/g, '') || 'unknown';
    const label = labels[normalized] || status || 'ไม่ทราบสถานะ';
    return `<span class="status-badge status-${safeClass}">${escapeHtml(label)}</span>`;

}





async function fetchUsername() {

    const token = localStorage.getItem('token');

    const usernameDisplay = document.getElementById('usernameDisplay');

    if (!token) {

        usernameDisplay.textContent = '⚠️ กรุณาเข้าสู่ระบบ';

        return;

    }
    try {
        const res = await fetch(`${window.API_BASE_URL}/get-username`, {
            headers: {
                Authorization: `Bearer ${token}`,
            },
        });
        const data = await res.json();
        if (res.ok && data.username) {
            usernameDisplay.textContent = `👤 สวัสดีคุณ: ${data.username}`;
        } else if (data.error) {
            usernameDisplay.textContent = `👤 ${data.error}`;
        } else {
            usernameDisplay.textContent = '👤 ไม่พบชื่อผู้ใช้';
        }
    } catch (error) {
        console.error('Error fetching username:', error);
        usernameDisplay.textContent = '👤 ดึงชื่อผู้ใช้ไม่สำเร็จ';
    }

}



async function fetchApiKeys() {

    const token = localStorage.getItem('token');

    const listElement = document.getElementById('apiKeysList');



    if (!token) {

        listElement.innerHTML = '<p>⚠️ กรุณาเข้าสู่ระบบก่อน</p>';

        return;

    }



    listElement.innerHTML = '<p>กำลังโหลดข้อมูล...</p>';



    try {

        const response = await fetch(`${window.API_BASE_URL}/get-api-keys`, {

            headers: {

                Authorization: `Bearer ${token}`,

            },

        });

        const data = await response.json();



        if (!response.ok || data.error) {

            listElement.innerHTML = `<p>${escapeHtml(data.error || 'เกิดข้อผิดพลาดในการดึงข้อมูล API Keys')}</p>`;

            return;

        }



        if (!Array.isArray(data.api_keys) || data.api_keys.length === 0) {

            listElement.innerHTML = '<p>ยังไม่มี API Key สำหรับบัญชีนี้</p>';

            return;

        }



        const cards = data.api_keys.map((key) => {

            const analysisText = escapeHtml(formatAnalysisTypes(key.analysis_types));

            const thresholdsText = escapeHtml(formatThresholds(key.thresholds));

            const createdText = escapeHtml(formatDateTime(key.created_at));

            const lastUsedText = escapeHtml(formatDateTime(key.last_used_at));

            const expiresText = escapeHtml(formatDateTime(key.expires_at));

            const planText = escapeHtml(key.plan || '—');

            const packageText = escapeHtml(key.package || '—');

            const usageCount = escapeHtml(typeof key.usage_count === 'number' ? key.usage_count : 0);

            const mediaAccessText = escapeHtml(formatMediaAccess(key.media_access));

            const outputModesText = escapeHtml(formatOutputModes(key.output_modes));



            return `

                <div class="api-key">

                    <p><strong>API Key:</strong> ${escapeHtml(key.api_key || '—')}</p>

                    <p><strong>Plan:</strong> ${planText}</p>

                    <p><strong>Package:</strong> ${packageText}</p>

                    <p><strong>Media Access:</strong> ${mediaAccessText}</p>

                    <p><strong>Output Modes:</strong> ${outputModesText}</p>

                    <p><strong>Usage Count:</strong> ${usageCount}</p>

                    <p><strong>Analysis Types:</strong> ${analysisText}</p>

                    <p><strong>Thresholds:</strong> ${thresholdsText}</p>

                    <p><strong>Created At:</strong> ${createdText}</p>

                    <p><strong>Last Used:</strong> ${lastUsedText}</p>

                    ${key.expires_at ? `<p><strong>Expires At:</strong> ${expiresText}</p>` : ''}

                </div>

            `;

        });



        listElement.innerHTML = cards.join('');

    } catch (error) {

        console.error('Error fetching API keys:', error);

        listElement.innerHTML = '<p>เกิดข้อผิดพลาดในการดึงข้อมูล API Keys</p>';

    }

}



async function fetchApiKeyHistory() {

    const token = localStorage.getItem('token');

    const historyElement = document.getElementById('historyList');



    if (!token) {

        historyElement.innerHTML = '<p>⚠️ กรุณาเข้าสู่ระบบก่อน</p>';

        return;

    }



    historyElement.innerHTML = '<p>กำลังโหลดประวัติ...</p>';



    try {

        const response = await fetch(`${window.API_BASE_URL}/get-api-key-history?limit=50`, {

            headers: {

                Authorization: `Bearer ${token}`,

            },

        });

        const data = await response.json();



        if (!response.ok || data.error) {

            historyElement.innerHTML = `<p>${escapeHtml(data.error || 'เกิดข้อผิดพลาดในการดึงประวัติการใช้งาน')}</p>`;

            return;

        }



        if (!Array.isArray(data.history) || data.history.length === 0) {

            historyElement.innerHTML = '<p>ยังไม่มีประวัติการใช้งานสำหรับ API Key นี้</p>';

            return;

        }



        const entries = data.history.map((entry) => {

            const statusBadge = formatStatusBadge(entry.status);

            const fileName = escapeHtml(entry.original_filename || '�');
            const createdText = escapeHtml(formatDateTime(entry.created_at));

            const models = escapeHtml(formatAnalysisTypes(entry.analysis_types));

            const thresholds = escapeHtml(formatThresholds(entry.thresholds));

            const mediaAccess = escapeHtml(formatMediaAccess(entry.media_access));

            const outputModes = escapeHtml(formatOutputModes(entry.output_modes));


            const mediaType = (entry.media_type || '').toLowerCase();

            const mediaTypeLabel = mediaType === 'video' ? 'วิดีโอ' : mediaType === 'image' ? 'รูปภาพ' : 'ไม่ทราบ';

            const isVideo = mediaType === 'video';

            const detectionSummary = Array.isArray(entry.detection_summary) && entry.detection_summary.length ?
                escapeHtml(entry.detection_summary.join(', ')) :
                'ไม่มีการตรวจจับ';



            const links = [];

            if (isVideo) {

                if (entry.processed_video_url) {
                    links.push(`<a href='${escapeHtml(entry.processed_video_url)}' target='_blank' rel='noopener'>ดูวิดีโอ</a>`);
                }
                if (entry.processed_blurred_video_url) {
                    links.push(`<a href='${escapeHtml(entry.processed_blurred_video_url)}' target='_blank' rel='noopener'>ดูวิดีโอ (เบลอ)</a>`);
                }

            } else {

                if (entry.processed_image_url) {
                    links.push(`<a href='${escapeHtml(entry.processed_image_url)}' target='_blank' rel='noopener'>ดูภาพ</a>`);
                }
                if (entry.processed_blurred_image_url) {
                    links.push(`<a href='${escapeHtml(entry.processed_blurred_image_url)}' target='_blank' rel='noopener'>ดูภาพ (เบลอ)</a>`);
                }

            }



            const actions = links.length ? `<div class='history-actions'>${links.join('')}</div>` : '';

            const preview = '';




            return `

                <div class='history-entry'>


                    <p><strong>API Key:</strong> ${escapeHtml(entry.api_key || '?')}</p>
                    <p><strong>ชื่อไฟล์:</strong> ${fileName}</p>
                    <p><strong>สถานะ:</strong> ${statusBadge}</p>
                    <p><strong>ประเภทสื่อ:</strong> ${escapeHtml(mediaTypeLabel)}</p>
                    <p><strong>สรุปการตรวจจับ:</strong> ${detectionSummary}</p>
                    <p><strong>โมเดล:</strong> ${models}</p>
                    <p><strong>Thresholds:</strong> ${thresholds}</p>
                    <p><strong>สิทธิ์สื่อ:</strong> ${mediaAccess}</p>
                    <p><strong>โหมดเอาต์พุต:</strong> ${outputModes}</p>
                    <p><strong>วันที่สร้าง:</strong> ${createdText}</p>

                    ${preview}

                    ${actions}

                </div>

            `;

        });



        historyElement.innerHTML = entries.join('');

    } catch (error) {

        console.error('Error fetching API key history:', error);

        historyElement.innerHTML = '<p>เกิดข้อผิดพลาดในการดึงประวัติการใช้งาน</p>';

    }

}

document.addEventListener('click', function (event) {
    const button = event.target.closest('.show-video-btn');
    if (!button) {
        return;
    }

    const container = button.closest('.history-preview');
    if (!container) {
        return;
    }

    const videoUrl = button.getAttribute('data-video-url');
    if (!videoUrl) {
        return;
    }

    const videoElement = document.createElement('video');
    videoElement.controls = true;
    videoElement.preload = 'metadata';
    videoElement.src = videoUrl;
    videoElement.setAttribute('playsinline', '');
    videoElement.className = 'history-preview-video';

    container.innerHTML = '';
    container.appendChild(videoElement);
});



window.onload = async function () {

    const token = localStorage.getItem('token');

    if (!token) {

        document.getElementById('usernameDisplay').textContent = '⚠️ กรุณาเข้าสู่ระบบ';

        document.getElementById('apiKeysList').innerHTML = '<p>⚠️ กรุณาเข้าสู่ระบบก่อน</p>';

        document.getElementById('historyList').innerHTML = '<p>⚠️ กรุณาเข้าสู่ระบบก่อน</p>';

        return;

    }

    if (typeof window.refreshMenubarAuthState === 'function') {
        window.refreshMenubarAuthState();
    }



    await fetchUsername();

    await Promise.all([fetchApiKeys(), fetchApiKeyHistory()]);

};



function logout() {

    localStorage.removeItem('token');
    if (typeof window.refreshMenubarAuthState === 'function') {
        window.refreshMenubarAuthState();
    }

    window.location.href = '../homepage/index.html';

}