// Initialize Chart.js
Chart.defaults.font.family = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(0, 0, 0, 0.8)';
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 4;

// Chart configurations
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

// Initialize charts
const charts = {};
Object.keys(chartConfigs).forEach(key => {
    const ctx = document.getElementById(`${key}Chart`).getContext('2d');
    charts[key] = new Chart(ctx, chartConfigs[key]);
});

// Historical charts
let historicalCharts = {};

// Function to create historical charts
function createHistoricalChart(canvasId, label, data, color, metricKey) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    new Chart(ctx, {
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
            }
        }
    });
}

// Load network interfaces
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

// Download data
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

// Show analysis
document.getElementById('analysisBtn').addEventListener('click', async () => {
    try {
        const response = await fetch('/api/get-data');
        const data = await response.json();
        
        if (data.length > 0) {
            // Create historical charts with specific metric keys
            createHistoricalChart('historicalPacketLossChart', 'Packet Loss (%)', data, '#dc3545', 'packet_loss');
            createHistoricalChart('historicalJitterChart', 'Jitter (ms)', data, '#0dcaf0', 'jitter');
            createHistoricalChart('historicalBandwidthChart', 'Bandwidth (Mbps)', data, '#198754', 'bandwidth');
            
            // Get AI predictions
            const predictionsResponse = await fetch('/api/get-predictions');
            const predictionsData = await predictionsResponse.json();
            
            if (predictionsData.status === 'success') {
                // Create quality prediction chart
                const qualityScoreCtx = document.getElementById('qualityScoreChart').getContext('2d');
                new Chart(qualityScoreCtx, {
                    type: 'line',
                    data: {
                        labels: data.map(d => new Date(d.timestamp * 1000).toLocaleTimeString()),
                        datasets: [{
                            label: 'Actual Quality Score',
                            data: data.map(d => d.quality_score),
                            borderColor: '#0d6efd',
                            backgroundColor: '#0d6efd20',
                            fill: true,
                            tension: 0.4
                        }, {
                            label: 'AI Predicted Quality',
                            data: predictionsData.data.predictions,
                            borderColor: '#198754',
                            backgroundColor: '#19875420',
                            fill: true,
                            tension: 0.4,
                            borderDash: [5, 5]
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100,
                                title: {
                                    display: true,
                                    text: 'Quality Score (0-100)'
                                }
                            }
                        }
                    }
                });
                
                // Create feature importance chart
                const featureImportanceCtx = document.getElementById('featureImportanceChart').getContext('2d');
                const importanceData = predictionsData.data.feature_importance;
                
                new Chart(featureImportanceCtx, {
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
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1,
                                title: {
                                    display: true,
                                    text: 'Importance Score'
                                }
                            }
                        }
                    }
                });
            }
            
            // Get AI suggestions
            const suggestionsResponse = await fetch('/api/get-suggestions');
            const suggestionsData = await suggestionsResponse.json();
            
            if (suggestionsData.status === 'success') {
                // Create suggestions container
                const suggestionsContainer = document.createElement('div');
                suggestionsContainer.className = 'card mt-4';
                suggestionsContainer.innerHTML = `
                    <div class="card-header">
                        <h5 class="card-title mb-0">AI Improvement Suggestions</h5>
                    </div>
                    <div class="card-body">
                        <div class="list-group">
                            ${suggestionsData.data.suggestions.map(suggestion => `
                                <div class="list-group-item">
                                    <h6 class="mb-1">${suggestion.metric}</h6>
                                    <p class="mb-1">Current: ${suggestion.current}</p>
                                    <small class="text-muted">${suggestion.suggestion}</small>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
                
                // Add suggestions to the modal
                const modalBody = document.querySelector('#analysisModal .modal-body');
                modalBody.appendChild(suggestionsContainer);
            }
            
            // Show modal
            const modal = new bootstrap.Modal(document.getElementById('analysisModal'));
            modal.show();
        } else {
            alert('No data available for analysis. Please collect some data first.');
        }
    } catch (error) {
        console.error('Error showing analysis:', error);
        alert('Error showing analysis');
    }
});

// Add train model button handler
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

// Update button states
function updateButtonStates(isMonitoring) {
    document.getElementById('startBtn').disabled = isMonitoring;
    document.getElementById('stopBtn').disabled = !isMonitoring;
    document.getElementById('downloadBtn').disabled = !isMonitoring;
    document.getElementById('analysisBtn').disabled = !isMonitoring;
    document.getElementById('trainModelBtn').disabled = !isMonitoring;
}

// Start monitoring
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

// Stop monitoring
document.getElementById('stopBtn').addEventListener('click', async () => {
    try {
        await fetch('/api/stop-monitoring');
        updateButtonStates(false);
    } catch (error) {
        console.error('Error stopping monitoring:', error);
        alert('Error stopping monitoring');
    }
});

// Update charts with new data
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

// Collect data periodically
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

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadNetworkInterfaces();
}); 