// Initialize Chart.js
Chart.defaults.font.family = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(0, 0, 0, 0.8)';
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 4;

// Chart configurations for real-time metrics
const chartConfigs = {
    packetLoss: {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Packet Loss (%)',
                data: [],
                borderColor: '#dc3545',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    },
    latency: {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Latency (ms)',
                data: [],
                borderColor: '#0d6efd',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    },
    jitter: {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Jitter (ms)',
                data: [],
                borderColor: '#198754',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    },
    bandwidth: {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Bandwidth (Mbps)',
                data: [],
                borderColor: '#6f42c1',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    }
};

// Initialize real-time charts
const charts = {};
Object.keys(chartConfigs).forEach(key => {
    const ctx = document.getElementById(`${key}Chart`);
    if (ctx) { // Check if canvas element exists
        charts[key] = new Chart(ctx.getContext('2d'), chartConfigs[key]);
    }
});

// Analysis Modal (Historical Metrics)
let analysisModal = null;
let historicalCharts = {};

document.getElementById('analysisBtn').addEventListener('click', async () => {
    try {
        // Initialize modal if not already done
        if (!analysisModal) {
            analysisModal = new bootstrap.Modal(document.getElementById('analysisModal'));
        }

        // Show loading state
        const modalBody = document.querySelector('#analysisModal .modal-body');
        modalBody.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;

        // Show modal
        analysisModal.show();

        // Fetch data (historical metrics)
        const historicalDataResponse = await fetch('/api/get-data');
        if (!historicalDataResponse.ok) {
            throw new Error(`HTTP error! status: ${historicalDataResponse.status}`);
        }
        const historicalData = await historicalDataResponse.json();

        if (!historicalData || historicalData.length === 0) {
            modalBody.innerHTML = `
                <div class="alert alert-warning">
                    No data available for analysis. Please collect more data.
                </div>
            `;
            return;
        }

        // Clear existing historical charts
        Object.values(historicalCharts).forEach(chart => { if (chart) chart.destroy(); });
        historicalCharts = {};

        // Construct modal content for historical metrics only
        modalBody.innerHTML = `
            <div class="row">
                <div class="col-12 mb-4">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="card-title mb-0">Packet Loss Analysis</h6>
                        </div>
                        <div class="card-body">
                            <canvas id="historicalPacketLossChart"></canvas>
                        </div>
                    </div>
                </div>
                <div class="col-12 mb-4">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="card-title mb-0">Jitter Analysis</h6>
                        </div>
                        <div class="card-body">
                            <canvas id="historicalJitterChart"></canvas>
                        </div>
                    </div>
                </div>
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="card-title mb-0">Bandwidth Analysis</h6>
                        </div>
                        <div class="card-body">
                            <canvas id="historicalBandwidthChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Create charts after DOM is updated
        setTimeout(() => {
            historicalCharts.packetLoss = createHistoricalChart('historicalPacketLossChart', 'Packet Loss (%)', historicalData, '#dc3545', 'packet_loss');
            historicalCharts.jitter = createHistoricalChart('historicalJitterChart', 'Jitter (ms)', historicalData, '#198754', 'jitter');
            historicalCharts.bandwidth = createHistoricalChart('historicalBandwidthChart', 'Bandwidth (Mbps)', historicalData, '#6f42c1', 'bandwidth');
        }, 100);

    } catch (error) {
        console.error('Error showing analysis:', error);
        const modalBody = document.querySelector('#analysisModal .modal-body');
        modalBody.innerHTML = `
            <div class="alert alert-danger">
                Error loading analysis data. Please try again.
                <br>
                <small class="text-muted">${error.message}</small>
            </div>
        `;
    }
});

// Quality Analysis Modal
let qualityAnalysisModal = null;
let qualityCharts = {}; // Dedicated object for quality analysis charts

// Variables to hold data fetched when the modal opens
let currentQualityHistoricalData = null;
let currentQualityPredictionsData = null;
let currentQualitySuggestionsData = null;

document.getElementById('qualityAnalysisBtn').addEventListener('click', async () => {
    try {
        // Initialize modal if not already done
        if (!qualityAnalysisModal) {
            qualityAnalysisModal = new bootstrap.Modal(document.getElementById('qualityAnalysisModal'));
        }

        // Show loading state
        const modalBody = document.querySelector('#qualityAnalysisModal .modal-body');
        modalBody.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;

        // Show modal immediately
        qualityAnalysisModal.show();

        // Fetch data
        const historicalDataResponse = await fetch('/api/get-data');
        if (!historicalDataResponse.ok) {
            throw new Error(`HTTP error! status: ${historicalDataResponse.status}`);
        }
        currentQualityHistoricalData = await historicalDataResponse.json();

        const predictionsResponse = await fetch('/api/get-predictions');
        if (!predictionsResponse.ok) {
            throw new Error(`HTTP error! status: ${predictionsResponse.status}`);
        }
        const predictionsData = await predictionsResponse.json();
        currentQualityPredictionsData = predictionsData.data;

        const suggestionsResponse = await fetch('/api/get-suggestions');
        if (!suggestionsResponse.ok) {
            throw new Error(`HTTP error! status: ${suggestionsResponse.status}`);
        }
        const suggestionsData = await suggestionsResponse.json();
        currentQualitySuggestionsData = suggestionsData.data;

        if (!currentQualityHistoricalData || currentQualityHistoricalData.length === 0) {
            modalBody.innerHTML = `
                <div class="alert alert-warning">
                    No data available for quality analysis. Please collect more data and train the model.
                </div>
            `;
            return;
        }

        // Construct modal content for quality analysis
        modalBody.innerHTML = `
            <div class="row">
                <div class="col-12 mb-4">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="card-title mb-0">Quality Score Analysis (Actual vs. AI Predicted)</h6>
                        </div>
                        <div class="card-body chart-container" style="min-height: 250px;">
                            <canvas id="qualityScoreChart"></canvas>
                        </div>
                    </div>
                </div>
                <div class="col-12 mb-4">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="card-title mb-0">Feature Importance</h6>
                        </div>
                        <div class="card-body chart-container" style="min-height: 250px;">
                            <canvas id="featureImportanceChart"></canvas>
                        </div>
                    </div>
                </div>
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="card-title mb-0">AI Improvement Suggestions</h6>
                        </div>
                        <div class="card-body" id="qualitySuggestionsContainer">
                            <!-- Suggestions will be loaded here -->
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Clear existing charts
        Object.values(qualityCharts).forEach(chart => { if (chart) chart.destroy(); });
        qualityCharts = {};

        // Create charts
        const qualityScoreCtx = document.getElementById('qualityScoreChart');
        if (qualityScoreCtx) {
            qualityCharts.qualityScore = new Chart(qualityScoreCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: currentQualityHistoricalData.map(d => new Date(d.timestamp * 1000).toLocaleTimeString()),
                    datasets: [{
                        label: 'Actual Quality Score',
                        data: currentQualityHistoricalData.map(d => d.quality_score),
                        borderColor: '#0d6efd',
                        backgroundColor: '#0d6efd20',
                        fill: true,
                        tension: 0.4
                    }, {
                        label: 'AI Predicted Quality',
                        data: currentQualityPredictionsData.predictions,
                        borderColor: '#198754',
                        backgroundColor: '#19875420',
                        fill: true,
                        tension: 0.4,
                        borderDash: [5, 5]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Quality Score (0-100)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top'
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        }
                    }
                }
            });
        }

        // Feature importance chart
        const featureImportanceCtx = document.getElementById('featureImportanceChart');
        if (featureImportanceCtx) {
            const importanceData = currentQualityPredictionsData.feature_importance;
            qualityCharts.featureImportance = new Chart(featureImportanceCtx.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: importanceData.map(f => f.feature),
                    datasets: [{
                        label: 'Feature Importance',
                        data: importanceData.map(f => f.importance),
                        backgroundColor: '#0dcaf0'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            title: {
                                display: true,
                                text: 'Importance Score'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }

        // Load suggestions
        const suggestionsContainer = document.getElementById('qualitySuggestionsContainer');
        if (suggestionsContainer) {
            suggestionsContainer.innerHTML = `
                <div class="list-group">
                    ${currentQualitySuggestionsData.suggestions.map(suggestion => `
                        <div class="list-group-item">
                            <h6 class="mb-1">${suggestion.metric}</h6>
                            <p class="mb-1">Current: ${suggestion.current}</p>
                            <small class="text-muted">${suggestion.suggestion}</small>
                        </div>
                    `).join('')}
                </div>
            `;
        }

    } catch (error) {
        console.error('Error showing quality analysis:', error);
        const modalBody = document.querySelector('#qualityAnalysisModal .modal-body');
        modalBody.innerHTML = `
            <div class="alert alert-danger">
                Error loading quality analysis data. Please try again.
                <br>
                <small class="text-muted">${error.message}</small>
            </div>
        `;
    }
});

// Event listener for when the quality analysis modal is hidden
document.getElementById('qualityAnalysisModal').addEventListener('hidden.bs.modal', () => {
    // Destroy charts when the modal is hidden to free up resources and prevent rendering issues on re-open
    Object.values(qualityCharts).forEach(chart => {
        if (chart) {
            chart.destroy();
        }
    });
    qualityCharts = {};
    // Clear stored data
    currentQualityHistoricalData = null;
    currentQualityPredictionsData = null;
    currentQualitySuggestionsData = null;
});

// Advanced Recommendations
document.getElementById('advancedRecommendationsBtn').addEventListener('click', async () => {
    const modal = new bootstrap.Modal(document.getElementById('advancedRecommendationsModal'));
    const content = document.getElementById('advancedRecommendationsContent');

    try {
        // Show loading spinner
        content.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;

        modal.show();

        // Get advanced recommendations
        const response = await fetch('/api/get-advanced-recommendations');
        const data = await response.json();

        if (data.status === 'success' && data.data) {
            const analysis = data.data;
            
            // Create HTML content
            let html = '';

            // Add analysis content
            if (analysis.analysis) {
                // Split the analysis into sections
                const sections = analysis.analysis.split('\n\n');
                
                sections.forEach(section => {
                    if (section.trim()) {
                        html += `
                            <div class="card mb-4">
                                <div class="card-body">
                                    <div class="analysis-section">
                                        ${section.split('\n').map(line => {
                                            if (line.startsWith('•')) {
                                                return `<p class="mb-2"><i class="fas fa-circle-dot"></i> ${line.substring(1).trim()}</p>`;
                                            } else if (line.startsWith('-')) {
                                                return `<p class="mb-2 ms-4"><i class="fas fa-minus"></i> ${line.substring(1).trim()}</p>`;
                                            } else if (line.includes(':')) {
                                                const [title, content] = line.split(':');
                                                return `<h6 class="mb-2">${title.trim()}:</h6><p class="mb-2">${content.trim()}</p>`;
                                            } else {
                                                return `<p class="mb-2">${line}</p>`;
                                            }
                                        }).join('')}
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                });
            }

            // Add metrics if available
            if (analysis.average_metrics) {
                html += `
                    <div class="card mb-4">
                        <div class="card-header bg-info text-white">
                            <h6 class="mb-0"><i class="fas fa-chart-line"></i> Average Metrics</h6>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-3">
                                    <p class="mb-1">Packet Loss: ${analysis.average_metrics.packet_loss.toFixed(2)}%</p>
                                </div>
                                <div class="col-md-3">
                                    <p class="mb-1">Latency: ${analysis.average_metrics.latency.toFixed(2)}ms</p>
                                </div>
                                <div class="col-md-3">
                                    <p class="mb-1">Jitter: ${analysis.average_metrics.jitter.toFixed(2)}ms</p>
                                </div>
                                <div class="col-md-3">
                                    <p class="mb-1">Bandwidth: ${analysis.average_metrics.bandwidth.toFixed(2)}Mbps</p>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }

            // Add fallback message if using fallback mode
            if (analysis.is_fallback) {
                html += `
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle"></i>
                        ${analysis.analysis}
                    </div>
                `;
            }

            content.innerHTML = html;
        } else {
            content.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-circle"></i>
                    Error loading advanced recommendations. Please try again.
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading advanced recommendations:', error);
        content.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle"></i>
                Error loading advanced recommendations. Please try again.
                <br>
                <small class="text-muted">${error.message}</small>
            </div>
        `;
    }
});

// Function to create historical charts (re-added)
function createHistoricalChart(canvasId, label, data, color, metricKey) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) {
        console.error(`Canvas element ${canvasId} not found`);
        return null;
    }

    return new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: data.map(d => new Date(d.timestamp * 1000).toLocaleTimeString()),
            datasets: [{
                label: label,
                data: data.map(d => d[metricKey]),
                borderColor: color,
                backgroundColor: color + '20',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: label
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// Load network interfaces (re-added)
async function loadNetworkInterfaces() {
    try {
        const response = await fetch('/api/network-interfaces');
        const interfaces = await response.json();
        const select = document.getElementById('interface');
        select.innerHTML = interfaces.map(iface =>
            `<option value="${iface}">${iface}</option>`
        ).join('');
    } catch (error) {
        console.error('Error loading network interfaces:', error);
    }
}

// Download data (re-added)
document.getElementById('downloadBtn').addEventListener('click', async () => {
    const downloadBtn = document.getElementById('downloadBtn');
    const originalText = downloadBtn.innerHTML;

    try {
        downloadBtn.disabled = true;
        downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Preparing Download...';

        const response = await fetch('/api/save-data');
        const data = await response.json();

        if (data.status === 'success') {
            // Create a temporary link to download the file
            const link = document.createElement('a');
            link.href = `/api/download/${data.filename}`;
            link.download = data.filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            // Show success message
            const alertDiv = document.createElement('div');
            alertDiv.className = 'alert alert-success alert-dismissible fade show';
            alertDiv.innerHTML = `
                <strong>Success!</strong> Your data has been downloaded.
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            `;
            document.querySelector('.container').insertBefore(alertDiv, document.querySelector('.row'));

            // Remove alert after 5 seconds
            setTimeout(() => {
                alertDiv.remove();
            }, 5000);
        } else {
            // Show error message
            const alertDiv = document.createElement('div');
            alertDiv.className = 'alert alert-warning alert-dismissible fade show';
            alertDiv.innerHTML = `
                <strong>Warning!</strong> ${data.message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            `;
            document.querySelector('.container').insertBefore(alertDiv, document.querySelector('.row'));
        }
    } catch (error) {
        console.error('Error downloading data:', error);
        // Show error message
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show';
        alertDiv.innerHTML = `
            <strong>Error!</strong> Failed to download data. Please try again.
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        document.querySelector('.container').insertBefore(alertDiv, document.querySelector('.row'));
    } finally {
        downloadBtn.disabled = false;
        downloadBtn.innerHTML = originalText;
    }
});

// Add train model button handler (re-added with correct functionality)
document.getElementById('trainModelBtn').addEventListener('click', async () => {
    try {
        const response = await fetch('/api/train-model', {
            method: 'POST'
        });
        const data = await response.json();

        if (data.status === 'success') {
            alert('Model trained successfully!');
        } else {
            alert(`Error training model: ${data.message}`);
        }
    } catch (error) {
        console.error('Error training model:', error);
        alert('Error training model');
    }
});

// Update button states (re-added)
function updateButtonStates(isMonitoring) {
    document.getElementById('startBtn').disabled = isMonitoring;
    document.getElementById('stopBtn').disabled = !isMonitoring;
    document.getElementById('downloadBtn').disabled = !isMonitoring;
    document.getElementById('analysisBtn').disabled = !isMonitoring;
    document.getElementById('trainModelBtn').disabled = !isMonitoring;
    document.getElementById('qualityAnalysisBtn').disabled = !isMonitoring; // Added new button here
    document.getElementById('advancedRecommendationsBtn').disabled = !isMonitoring;
}

// Start monitoring (re-added)
document.getElementById('monitoringForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = {
        interface: document.getElementById('interface').value,
        target_host: document.getElementById('targetHost').value,
        duration: parseInt(document.getElementById('duration').value)
    };

    try {
        const response = await fetch('/api/start-monitoring', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();
        if (data.status === 'success') {
            updateButtonStates(true);
            startDataCollection();
        } else {
            alert('Monitoring is already running');
        }
    } catch (error) {
        console.error('Error starting monitoring:', error);
        alert('Error starting monitoring');
    }
});

// Stop monitoring (re-added)
document.getElementById('stopBtn').addEventListener('click', async () => {
    try {
        await fetch('/api/stop-monitoring');
        updateButtonStates(false);
    } catch (error) {
        console.error('Error stopping monitoring:', error);
        alert('Error stopping monitoring');
    }
});

// Update charts with new data (re-added)
function updateCharts(data) {
    const timestamp = new Date().toLocaleTimeString();

    Object.keys(charts).forEach(key => {
        const chart = charts[key];
        const value = data[key];

        chart.data.labels.push(timestamp);
        chart.data.datasets[0].data.push(value);

        // Keep only last 20 data points
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }

        chart.update();
    });
}

// Collect data periodically (re-added)
let dataCollectionInterval;
function startDataCollection() {
    dataCollectionInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/get-data');
            const data = await response.json();
            if (data.length > 0) {
                updateCharts(data[data.length - 1]);
            }
        } catch (error) {
            console.error('Error collecting data:', error);
        }
    }, 1000);
}


// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadNetworkInterfaces();
    // Initially, stop monitoring state
    updateButtonStates(false);
}); 