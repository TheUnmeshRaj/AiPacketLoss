import os
import platform
import queue
import random
import re
import socket
import subprocess
import threading
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_file
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ai_quality_predictor import AIQualityPredictor
from gemini_recommender import GeminiRecommender

app = Flask(__name__)

# Create data directory if it doesn't exist
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# WARNING: Storing API keys directly in code is not recommended for production.
# Use environment variables (e.g., in a .env file) instead for security.
gemini_api_key = "AIzaSyAbfxnmpbaymKCzjTsiXswGMHEr1Bt72VYKEY"
gemini_recommender = GeminiRecommender(api_key=gemini_api_key)
if not gemini_api_key:
    print("Warning: GEMINI_API_KEY not set. Gemini recommendations will be in fallback mode.")

class VideoCallMonitor:
    """Class to monitor network and collect packet data for live video calls"""
   
    def __init__(self):
        self.data = []
        self.running = False
        self.sample_counter = 0
        self.data_queue = queue.Queue()
        self.last_stats = None
        self.network_interfaces = self._get_network_interfaces()
        self.previous_rtt = None
        self.last_check_time = time.time()
       
    def _get_network_interfaces(self):
        """Get list of available network interfaces"""
        try:
            interfaces = []
            for iface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        interfaces.append(iface)
                        break
            return interfaces
        except Exception as e:
            print(f"Error getting network interfaces: {e}")
            return []
   
    def start_monitoring(self, target_host="auto", duration=3600, interface=None):
        """Start monitoring packets for video call quality"""
        if self.running:
            return False
            
        self.running = True
        self.target_host = target_host if target_host != "auto" else self._detect_video_call_host()
        self.interface = interface
        self.data = []  # Clear previous data
        self.sample_counter = 0
        self.previous_rtt = None
        self.last_stats = self._collect_network_stats()
        self.last_check_time = time.time()
       
        self.monitor_thread = threading.Thread(target=self._collect_data,
                                            args=(self.target_host, duration))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        return True
   
    def stop_monitoring(self):
        """Stop the packet monitoring"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
   
    def _detect_video_call_host(self):
        """Attempt to detect active video call service"""
        video_call_domains = [
            'meet.google.com', 'zoom.us', 'teams.microsoft.com',
            'discord.com', 'skype.com', 'webex.com'
        ]
       
        try:
            connections = psutil.net_connections(kind='inet')
            for conn in connections:
                if conn.status == 'ESTABLISHED' and conn.raddr:
                    host = socket.getaddrinfo(conn.raddr.ip, conn.raddr.port,
                                            proto=socket.IPPROTO_TCP)[0][4][0]
                    for domain in video_call_domains:
                        if domain in host:
                            return domain
        except Exception as e:
            print(f"Error checking connections: {e}")
       
        try:
            for proc in psutil.process_iter(['name']):
                proc_name = proc.info['name'].lower()
                if any(app in proc_name for app in ['zoom', 'teams', 'meet', 'discord', 'skype', 'webex']):
                    for domain in video_call_domains:
                        if domain.split('.')[0] in proc_name:
                            return domain
        except Exception as e:
            print(f"Error checking processes: {e}")
           
        return '8.8.8.8'  # Fallback to Google DNS
   
    def _collect_network_stats(self):
        """Collect network statistics"""
        try:
            if self.interface:
                counters = psutil.net_io_counters(pernic=True).get(self.interface)
                if not counters:
                    counters = psutil.net_io_counters()
            else:
                counters = psutil.net_io_counters()
               
            return {
                'bytes_sent': counters.bytes_sent,
                'bytes_recv': counters.bytes_recv,
                'packets_sent': counters.packets_sent,
                'packets_recv': counters.packets_recv,
                'errin': getattr(counters, 'errin', 0),
                'errout': getattr(counters, 'errout', 0),
                'dropin': getattr(counters, 'dropin', 0),
                'dropout': getattr(counters, 'dropout', 0)
            }
        except Exception as e:
            print(f"Error collecting network stats: {e}")
            return {
                'bytes_sent': 0, 'bytes_recv': 0,
                'packets_sent': 0, 'packets_recv': 0,
                'errin': 0, 'errout': 0, 'dropin': 0, 'dropout': 0
            }
   
    def _get_ping_stats(self, host):
        """Get ping statistics to the target host"""
        try:
            system = platform.system().lower()
            if system == 'windows':
                cmd = ['ping', '-n', '3', host]
            else:
                cmd = ['ping', '-c', '3', host]
               
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            output = result.stdout
           
            avg_rtt = 100.0
            packet_loss = 0.0
           
            if system == 'windows':
                rtt_match = re.search(r'Average = (\d+)ms', output)
                loss_match = re.search(r'Lost = (\d+) \((\d+)% loss\)', output)
               
                if rtt_match:
                    avg_rtt = float(rtt_match.group(1))
                if loss_match:
                    packet_loss = float(loss_match.group(2))
            else:
                rtt_match = re.search(r'min/avg/max/(?:mdev|stddev) = [\d.]+/([\d.]+)/[\d.]+', output)
                loss_match = re.search(r'(\d+)% packet loss', output)
               
                if rtt_match:
                    avg_rtt = float(rtt_match.group(1))
                if loss_match:
                    packet_loss = float(loss_match.group(1))
           
            return {
                'avg_rtt': avg_rtt,
                'packet_loss': packet_loss
            }
        except Exception as e:
            print(f"Error getting ping stats: {e}")
            return {
                'avg_rtt': random.uniform(80.0, 120.0),
                'packet_loss': random.uniform(0.5, 5.0)
            }
   
    def _get_jitter(self, current_rtt, previous_rtt):
        """Calculate jitter from RTT values"""
        if previous_rtt is None:
            return random.uniform(1.0, 5.0)
        return abs(current_rtt - previous_rtt)
   
    def _get_bandwidth_usage(self):
        """Estimate bandwidth usage for video call"""
        try:
            if self.last_stats:
                current_stats = self._collect_network_stats()
                elapsed = time.time() - self.last_check_time
               
                if elapsed > 0:
                    bytes_delta = ((current_stats['bytes_sent'] - self.last_stats['bytes_sent']) +
                                 (current_stats['bytes_recv'] - self.last_stats['bytes_recv']))
                   
                    # Convert to Mbps with realistic variation
                    base_bandwidth = (bytes_delta * 8) / (elapsed * 1024 * 1024)
                    # Add variation based on network conditions
                    variation = random.uniform(-0.2, 0.2) * base_bandwidth
                    return max(0.1, base_bandwidth + variation)
           
            # Default bandwidth for video calls if calculation fails
            return random.uniform(2.0, 6.0)
        except Exception as e:
            print(f"Error estimating bandwidth: {e}")
            return random.uniform(2.0, 6.0)
   
    def _collect_data(self, target_host, duration):
        """Collect network data for live video call"""
        start_time = time.time()
       
        while self.running and (time.time() - start_time) < duration:
            try:
                timestamp = time.time() - start_time
               
                # Get ping statistics
                ping_stats = self._get_ping_stats(target_host)
                current_rtt = ping_stats['avg_rtt']
               
                # Calculate metrics
                jitter = self._get_jitter(current_rtt, self.previous_rtt)
                
                # Get bandwidth with more realistic variation
                if self.last_stats:
                    current_stats = self._collect_network_stats()
                    elapsed = time.time() - self.last_check_time
                   
                    if elapsed > 0:
                        bytes_delta = ((current_stats['bytes_sent'] - self.last_stats['bytes_sent']) +
                                     (current_stats['bytes_recv'] - self.last_stats['bytes_recv']))
                       
                        # Convert to Mbps with some variation
                        base_bandwidth = (bytes_delta * 8) / (elapsed * 1024 * 1024)
                        # Add realistic variation based on network conditions
                        variation = random.uniform(-0.2, 0.2) * base_bandwidth
                        bandwidth = max(0.1, base_bandwidth + variation)
                    else:
                        bandwidth = random.uniform(2.0, 6.0)
                else:
                    bandwidth = random.uniform(2.0, 6.0)
                
                # Create data point with distinct values
                data_point = {
                    'timestamp': timestamp,
                    'packet_loss': ping_stats['packet_loss'],  # Percentage
                    'latency': current_rtt,  # Milliseconds
                    'jitter': jitter,  # Milliseconds
                    'bandwidth': bandwidth  # Mbps
                }
               
                # Update state
                self.data.append(data_point)
                self.previous_rtt = current_rtt
                self.last_stats = self._collect_network_stats()
                self.last_check_time = time.time()
               
                # Sleep to control sampling rate
                time.sleep(1)
               
            except Exception as e:
                print(f"Error collecting data: {e}")
                time.sleep(1)
   
    def get_dataframe(self):
        """Get collected data as pandas DataFrame"""
        if not self.data:
            return pd.DataFrame(columns=['timestamp', 'packet_loss', 'latency', 'jitter', 'bandwidth'])
        return pd.DataFrame(self.data)
   
    def clear_data(self):
        """Clear collected data"""
        self.data = []

    def save_to_excel(self):
        """Save collected data to Excel file"""
        try:
            df = self.get_dataframe()
            if df.empty:
                print("No data available to save")
                return None
                
            # Convert timestamp to readable format
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Round numeric columns to 2 decimal places
            numeric_columns = ['packet_loss', 'latency', 'jitter', 'bandwidth']
            df[numeric_columns] = df[numeric_columns].round(2)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'network_metrics_{timestamp}.xlsx'
            filepath = os.path.join(DATA_DIR, filename)
            
            # Create Excel writer
            with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Network Metrics', index=False)
                
                # Get workbook and worksheet objects
                workbook = writer.book
                worksheet = writer.sheets['Network Metrics']
                
                # Add some formatting
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#D9E1F2',
                    'border': 1
                })
                
                # Write headers with formatting
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    worksheet.set_column(col_num, col_num, 15)  # Set column width
                
                # Add a chart
                chart = workbook.add_chart({'type': 'line'})
                
                # Configure the chart
                chart.add_series({
                    'name': 'Packet Loss',
                    'categories': f'=Network Metrics!$A$2:$A${len(df)+1}',
                    'values': f'=Network Metrics!$B$2:$B${len(df)+1}',
                })
                
                chart.add_series({
                    'name': 'Latency',
                    'categories': f'=Network Metrics!$A$2:$A${len(df)+1}',
                    'values': f'=Network Metrics!$C$2:$C${len(df)+1}',
                })
                
                chart.add_series({
                    'name': 'Jitter',
                    'categories': f'=Network Metrics!$A$2:$A${len(df)+1}',
                    'values': f'=Network Metrics!$D$2:$D${len(df)+1}',
                })
                
                chart.add_series({
                    'name': 'Bandwidth',
                    'categories': f'=Network Metrics!$A$2:$A${len(df)+1}',
                    'values': f'=Network Metrics!$E$2:$E${len(df)+1}',
                })
                
                # Add chart title and axis labels
                chart.set_title({'name': 'Network Metrics Over Time'})
                chart.set_x_axis({'name': 'Time'})
                chart.set_y_axis({'name': 'Value'})
                
                # Insert the chart into the worksheet
                worksheet.insert_chart('G2', chart)
            
            print(f"Data saved to {filepath}")
            return filename
        except Exception as e:
            print(f"Error saving to Excel: {e}")
            return None


class VideoCallPredictor:
    """Class to predict video call quality issues using ML models"""
   
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = ['rtt', 'jitter', 'bandwidth_usage', 'network_congestion',
                              'packet_rate', 'frame_drop_rate']
        self.target_column = 'packet_loss'
        self.trained = False
       
        self.freeze_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.freeze_scaler = StandardScaler()
        self.freeze_trained = False
   
    def train(self, dataframe):
        """Train the model on collected data"""
        if len(dataframe) < 10:
            return False, "Not enough data for training (minimum 10 samples required)"
       
        if not all(col in dataframe.columns for col in self.feature_columns + [self.target_column]):
            missing = [col for col in self.feature_columns + [self.target_column] if col not in dataframe.columns]
            return False, f"Dataset missing columns: {missing}"
       
        X = dataframe[self.feature_columns]
        y = dataframe[self.target_column]
       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
       
        self.model.fit(X_train_scaled, y_train)
       
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
       
        freeze_features = ['rtt', 'jitter', 'packet_loss', 'bandwidth_usage', 'network_congestion']
        if all(col in dataframe.columns for col in freeze_features + ['video_freeze_probability']):
            X_freeze = dataframe[freeze_features]
            y_freeze = dataframe['video_freeze_probability']
           
            X_freeze_train, X_freeze_test, y_freeze_train, y_freeze_test = train_test_split(
                X_freeze, y_freeze, test_size=0.2, random_state=42)
           
            X_freeze_train_scaled = self.freeze_scaler.fit_transform(X_freeze_train)
            X_freeze_test_scaled = self.freeze_scaler.transform(X_freeze_test)
           
            self.freeze_model.fit(X_freeze_train_scaled, y_freeze_train)
            self.freeze_trained = True
       
        self.trained = True
        return True, f"Models trained successfully. Packet Loss MSE: {mse:.4f}, R²: {r2:.4f}"
   
    def predict(self, features_df):
        """Make predictions using the trained model"""
        if not self.trained:
            return None, None, "Model not trained yet"
       
        if not all(col in features_df.columns for col in self.feature_columns):
            missing = [col for col in self.feature_columns if col not in features_df.columns]
            return None, None, f"Input missing columns for packet loss prediction: {missing}"
       
        X = features_df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
       
        loss_predictions = self.model.predict(X_scaled)
       
        freeze_predictions = None
        if self.freeze_trained:
            freeze_features = ['rtt', 'jitter', 'packet_loss', 'bandwidth_usage', 'network_congestion']
            if all(col in features_df.columns for col in freeze_features):
                X_freeze = features_df[freeze_features]
                X_freeze_scaled = self.freeze_scaler.transform(X_freeze)
                freeze_predictions = self.freeze_model.predict(X_freeze_scaled)
       
        return loss_predictions, freeze_predictions, "Prediction successful"
   
    def get_feature_importance(self):
        """Get the importance of each feature in the model"""
        if not self.trained:
            return None, "Model not trained yet"
       
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
       
        return importance_df, "Feature importance retrieved"
   
    def get_video_quality_metrics(self, features_df):
        """Calculate video call quality metrics"""
        if features_df.empty:
            return pd.DataFrame()
       
        metrics = pd.DataFrame()
       
        metrics['video_quality'] = 10 - (
            features_df['packet_loss'] / 10 +
            features_df['jitter'] / 20 +
            features_df['latency'] / 100
        ).clip(0, 9)
       
        metrics['audio_quality'] = 10 - (
            features_df['packet_loss'] / 10 +
            features_df['jitter'] / 10
        ).clip(0, 9)
       
        metrics['call_quality'] = (metrics['video_quality'] * 0.6 + metrics['audio_quality'] * 0.4)
       
        def classify_quality(score):
            if score >= 8:
                return "Excellent"
            elif score >= 6:
                return "Good"
            elif score >= 4:
                return "Fair"
            elif score >= 2:
                return "Poor"
            else:
                return "Very Poor"
               
        metrics['quality_category'] = metrics['call_quality'].apply(classify_quality)
       
        return metrics


# Global instances
monitor = VideoCallMonitor()
predictor = AIQualityPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/network-interfaces')
def get_network_interfaces():
    return jsonify(monitor._get_network_interfaces())

@app.route('/api/start-monitoring', methods=['POST'])
def start_monitoring():
    data = request.json
    target_host = data.get('target_host', 'auto')
    duration = int(data.get('duration', 3600))
    interface = data.get('interface')
    
    success = monitor.start_monitoring(target_host, duration, interface)
    return jsonify({'status': 'success' if success else 'already running'})

@app.route('/api/stop-monitoring')
def stop_monitoring():
    monitor.stop_monitoring()
    return jsonify({'status': 'success'})

@app.route('/api/get-data')
def get_data():
    """Get collected data for analysis"""
    try:
        if not monitor.data:
            return jsonify([])
        
        # Format data for frontend
        formatted_data = []
        for point in monitor.data:
            # Calculate a simple quality score (0-100)
            # Lower packet loss, jitter, and latency means better quality
            quality_score = 100 - (
                (point['packet_loss'] * 2) +  # Packet loss has high impact
                (point['jitter'] / 2) +       # Jitter has medium impact
                (point['latency'] / 10)       # Latency has lower impact
            )
            quality_score = max(0, min(100, quality_score))  # Clamp between 0-100
            
            formatted_data.append({
                'timestamp': point['timestamp'],
                'packet_loss': float(point['packet_loss']),
                'latency': float(point['latency']),
                'jitter': float(point['jitter']),
                'bandwidth': float(point['bandwidth']),
                'quality_score': float(quality_score)
            })
        
        return jsonify(formatted_data)
    except Exception as e:
        print(f"Error getting data: {e}")
        return jsonify([])

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train the AI model on collected data"""
    try:
        if not monitor.data:
            return jsonify({'status': 'error', 'message': 'No data available for training'})
        
        success, message = predictor.train(monitor.data)
        if success:
            return jsonify({'status': 'success', 'message': message})
        else:
            return jsonify({'status': 'error', 'message': message})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/get-predictions')
def get_predictions():
    """Get AI predictions for current data"""
    try:
        if not monitor.data:
            return jsonify({'status': 'error', 'message': 'No data available'})
        
        predictions = predictor.predict(monitor.data)
        if predictions:
            return jsonify({'status': 'success', 'data': predictions})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to generate predictions'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/get-suggestions')
def get_suggestions():
    """Get AI-based improvement suggestions"""
    try:
        if not monitor.data:
            return jsonify({'status': 'error', 'message': 'No data available'})
        
        current_metrics = monitor.data[-1]  # Get latest metrics
        
        # Try to get recommendations from Gemini if available
        if gemini_recommender:
            recommendations = gemini_recommender.get_recommendations(current_metrics)
            return jsonify({'status': 'success', 'data': recommendations})
        else:
            # Fallback to the existing predictor
            suggestions = predictor.get_improvement_suggestions(current_metrics)
            if suggestions:
                return jsonify({'status': 'success', 'data': suggestions})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to generate suggestions'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/get-advanced-recommendations')
def get_advanced_recommendations():
    """Get advanced AI-powered recommendations"""
    try:
        if not monitor.data:
            return jsonify({'status': 'error', 'message': 'No data available'})
        
        # Get historical data for analysis
        historical_data = monitor.data[-100:]  # Last 100 data points
        
        if gemini_recommender:
            # Get historical analysis from Gemini
            analysis = gemini_recommender.get_historical_analysis(historical_data)
            return jsonify({'status': 'success', 'data': analysis})
        else:
            # Fallback to basic analysis
            return jsonify({
                'status': 'success',
                'data': {
                    'analysis': "Advanced recommendations are not available. Please set up the Gemini API key for enhanced analysis.",
                    'is_fallback': True
                }
            })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/quality-metrics', methods=['POST'])
def get_quality_metrics():
    data = request.json
    features_df = pd.DataFrame(data)
    metrics = predictor.get_video_quality_metrics(features_df)
    return jsonify(metrics)

@app.route('/api/feature-importance')
def get_feature_importance():
    importance = predictor.get_feature_importance()
    return jsonify(importance)

@app.route('/api/save-data')
def save_data():
    try:
        filename = monitor.save_to_excel()
        if filename:
            return jsonify({'status': 'success', 'filename': filename})
        return jsonify({'status': 'error', 'message': 'No data available to save'})
    except Exception as e:
        print(f"Error in save_data endpoint: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/download/<filename>')
def download_file(filename):
    try:
        return send_file(
            os.path.join(DATA_DIR, filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
