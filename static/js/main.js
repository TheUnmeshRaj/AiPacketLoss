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

function createHistoricalChart(canvasId, label, data, color, metricKey) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy existing chart if it exists
    if (historicalCharts[canvasId]) {
        historicalCharts[canvasId].destroy();
    }
    
    historicalCharts[canvasId] = new Chart(ctx, {
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
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(2);
                                if (label.includes('Packet Loss')) {
                                    label += '%';
                                } else if (label.includes('Jitter')) {
                                    label += ' ms';
                                } else if (label.includes('Bandwidth')) {
                                    label += ' Mbps';
                                }
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: label
                    },
                    beginAtZero: true
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
            
            // Create quality score chart
            const qualityScoreCtx = document.getElementById('qualityScoreChart').getContext('2d');
            new Chart(qualityScoreCtx, {
                type: 'line',
                data: {
                    labels: data.map(d => new Date(d.timestamp * 1000).toLocaleTimeString()),
                    datasets: [{
                        label: 'Call Quality Score',
                        data: data.map(d => d.quality_score),
                        borderColor: '#0d6efd',
                        backgroundColor: '#0d6efd20',
                        fill: true,
                        tension: 0.4
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
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const score = context.parsed.y;
                                    let quality = '';
                                    if (score >= 80) quality = 'Excellent';
                                    else if (score >= 60) quality = 'Good';
                                    else if (score >= 40) quality = 'Fair';
                                    else if (score >= 20) quality = 'Poor';
                                    else quality = 'Very Poor';
                                    return `Quality: ${score.toFixed(1)} (${quality})`;
                                }
                            }
                        }
                    }
                }
            });
            
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

// Update button states
function updateButtonStates(isMonitoring) {
    document.getElementById('startBtn').disabled = isMonitoring;
    document.getElementById('stopBtn').disabled = !isMonitoring;
    document.getElementById('downloadBtn').disabled = !isMonitoring;
    document.getElementById('analysisBtn').disabled = !isMonitoring;
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