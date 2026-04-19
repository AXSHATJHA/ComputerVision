const COLORS = {
    'EfficientNetB3': { border: '#38bdf8', bg: 'rgba(56, 189, 248, 0.2)' },
    'MobileNetV2': { border: '#a855f7', bg: 'rgba(168, 85, 247, 0.2)' },
    'ResNet50': { border: '#10b981', bg: 'rgba(16, 185, 129, 0.2)' }
};

Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.scale.grid.color = 'rgba(255, 255, 255, 0.05)';

let lineAccChart, lineLossChart, barAccChart, barF1Chart;

async function loadData() {
    try {
        const response = await fetch('metrics.json');
        if (!response.ok) throw new Error("Metrics not ready yet");
        
        const data = await response.json();
        
        document.getElementById('training-status').textContent = 'Training Completed / Data Synced';
        document.getElementById('training-status').previousElementSibling.style.backgroundColor = '#10b981';
        
        processMetrics(data);
    } catch (e) {
        document.getElementById('training-status').textContent = 'Training in Progress...';
        document.getElementById('training-status').previousElementSibling.style.backgroundColor = '#f59e0b';
        setTimeout(loadData, 5000); // Poll every 5s if file not there yet
    }
}

function processMetrics(data) {
    const models = Object.keys(data);
    if (models.length === 0) return;

    // 1. Calculate Hero Stats
    let topAcc = { val: 0, mod: '' };
    let topF1 = { val: 0, mod: '' };
    let lowestLoss = { val: Infinity, mod: '' };

    models.forEach(m => {
        if(data[m].test_accuracy > topAcc.val) {
            topAcc.val = data[m].test_accuracy;
            topAcc.mod = m;
        }
        if(data[m].f1_score > topF1.val) {
            topF1.val = data[m].f1_score;
            topF1.mod = m;
        }
        if(data[m].test_loss < lowestLoss.val) {
            lowestLoss.val = data[m].test_loss;
            lowestLoss.mod = m;
        }
    });

    document.getElementById('top-acc-value').textContent = (topAcc.val * 100).toFixed(2) + '%';
    document.getElementById('top-acc-model').textContent = topAcc.mod;

    document.getElementById('top-f1-value').textContent = (topF1.val * 100).toFixed(2) + '%';
    document.getElementById('top-f1-model').textContent = topF1.mod;

    document.getElementById('bottom-loss-value').textContent = lowestLoss.val.toFixed(4);
    document.getElementById('bottom-loss-model').textContent = lowestLoss.mod;

    // 2. Prepare Chart Datasets
    const epochs = data[models[0]]?.history?.val_accuracy?.map((_, i) => `Epoch ${i+1}`) || [];
    
    const accDatasets = models.map(m => ({
        label: m,
        data: data[m].history?.val_accuracy || [],
        borderColor: COLORS[m]?.border || '#ffffff',
        backgroundColor: COLORS[m]?.bg || 'rgba(255,255,255,0.1)',
        tension: 0.4,
        fill: true
    }));

    const lossDatasets = models.map(m => ({
        label: m,
        data: data[m].history?.val_loss || [],
        borderColor: COLORS[m]?.border || '#ffffff',
        backgroundColor: COLORS[m]?.bg || 'rgba(255,255,255,0.1)',
        tension: 0.4
    }));

    const testAccData = models.map(m => data[m].test_accuracy * 100);
    const f1Data = models.map(m => data[m].f1_score * 100);
    const barColors = models.map(m => COLORS[m]?.border || '#ffffff');
    const barBgs = models.map(m => COLORS[m]?.bg || 'rgba(255,255,255,0.1)');

    // 3. Render Line Charts
    renderLineChart('accuracyHistoryChart', epochs, accDatasets, 'Validation Accuracy');
    renderLineChart('lossHistoryChart', epochs, lossDatasets, 'Validation Loss');

    // 4. Render Bar Charts
    renderBarChart('barAccuracyChart', models, testAccData, barColors, barBgs, 'Test Accuracy (%)');
    renderBarChart('barF1Chart', models, f1Data, barColors, barBgs, 'Macro F1 Score (%)');
}

function renderLineChart(ctxId, labels, datasets, yTitle) {
    const ctx = document.getElementById(ctxId).getContext('2d');
    if(window[ctxId + 'Instance']) window[ctxId + 'Instance'].destroy();

    window[ctxId + 'Instance'] = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            },
            scales: {
                y: {
                    title: { display: true, text: yTitle },
                    beginAtZero: false
                }
            }
        }
    });
}

function renderBarChart(ctxId, labels, dataArr, borders, bgs, title) {
    const ctx = document.getElementById(ctxId).getContext('2d');
    if(window[ctxId + 'Instance']) window[ctxId + 'Instance'].destroy();

    window[ctxId + 'Instance'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: title,
                data: dataArr,
                backgroundColor: bgs,
                borderColor: borders,
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}

// Start sequence
document.addEventListener('DOMContentLoaded', () => {
    loadData();

    // Tab switching logic
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = e.currentTarget.getAttribute('data-target');
            if (!targetId) return;

            // Update active link
            document.querySelectorAll('.nav-links li').forEach(li => li.classList.remove('active'));
            e.currentTarget.parentElement.classList.add('active');

            // Update active tab pane
            document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add('active');
        });
    });
});
