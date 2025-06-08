import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class AIQualityPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = 'models/quality_predictor.joblib'
        self.scaler_path = 'models/quality_scaler.joblib'
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self):
        """Load existing model if available"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
                print("Loaded existing model")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def _save_model(self):
        """Save the trained model"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def prepare_features(self, data):
        """Prepare features for the model"""
        features = pd.DataFrame(data)
        
        # Add time-based features
        if 'timestamp' in features.columns:
            features['hour'] = pd.to_datetime(features['timestamp'], unit='s').dt.hour
            features['minute'] = pd.to_datetime(features['timestamp'], unit='s').dt.minute
        
        # Add rolling statistics
        window_size = 5
        for metric in ['packet_loss', 'jitter', 'latency', 'bandwidth']:
            if metric in features.columns:
                features[f'{metric}_rolling_mean'] = features[metric].rolling(window=window_size, min_periods=1).mean()
                features[f'{metric}_rolling_std'] = features[metric].rolling(window=window_size, min_periods=1).std()
        
        # Drop timestamp and handle NaN values
        features = features.drop('timestamp', axis=1, errors='ignore')
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        return features
    
    def train(self, data):
        """Train the model on historical data"""
        try:
            # Prepare features
            features = self.prepare_features(data)
            
            # Calculate target (quality score)
            target = 100 - (
                (features['packet_loss'] * 2) +
                (features['jitter'] / 2) +
                (features['latency'] / 10)
            ).clip(0, 100)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            print(f"Model trained successfully")
            print(f"Training R² score: {train_score:.3f}")
            print(f"Testing R² score: {test_score:.3f}")
            
            self.is_trained = True
            self._save_model()
            
            return True, "Model trained successfully"
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False, str(e)
    
    def predict(self, data):
        """Make quality predictions"""
        if not self.is_trained:
            return None, "Model not trained"
        
        try:
            # Prepare features
            features = self.prepare_features(data)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make predictions
            predictions = self.model.predict(features_scaled)
            
            # Get feature importance
            importance = pd.DataFrame({
                'feature': features.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'predictions': predictions.tolist(),
                'feature_importance': importance.to_dict('records')
            }
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None, str(e)
    
    def _get_advanced_recommendations(self, current_metrics):
        """Generate advanced rule-based recommendations using multiple metrics correlation"""
        advanced_suggestions = []
        
        # Check for correlated issues
        if all(metric in current_metrics for metric in ['packet_loss', 'jitter', 'latency']):
            # High packet loss + high jitter correlation
            if current_metrics['packet_loss'] > 5 and current_metrics['jitter'] > 20:
                advanced_suggestions.append({
                    'type': 'correlated_issue',
                    'metrics': ['Packet Loss', 'Jitter'],
                    'suggestion': "Network instability detected. This combination suggests network congestion or interference. Consider: 1) Check for network congestion during peak hours, 2) Investigate potential interference sources (microwaves, other WiFi networks), 3) Consider implementing traffic shaping, 4) Monitor network usage patterns"
                })
            
            # High latency + high packet loss correlation
            if current_metrics['latency'] > 100 and current_metrics['packet_loss'] > 3:
                advanced_suggestions.append({
                    'type': 'correlated_issue',
                    'metrics': ['Latency', 'Packet Loss'],
                    'suggestion': "Network path issues detected. This combination suggests problems with the network route. Consider: 1) Check for routing issues, 2) Verify DNS settings, 3) Consider using a different DNS server, 4) Check for VPN or proxy interference"
                })
            
            # High jitter + high latency correlation
            if current_metrics['jitter'] > 30 and current_metrics['latency'] > 150:
                advanced_suggestions.append({
                    'type': 'correlated_issue',
                    'metrics': ['Jitter', 'Latency'],
                    'suggestion': "Network buffer issues detected. This combination suggests problems with network buffering. Consider: 1) Adjust network buffer settings, 2) Check for network equipment issues, 3) Consider upgrading network hardware, 4) Implement QoS policies"
                })
        
        # Check for bandwidth-related issues
        if 'bandwidth' in current_metrics:
            bandwidth = current_metrics['bandwidth']
            if bandwidth < 2:
                # Check if other metrics are also poor
                if current_metrics.get('packet_loss', 0) > 3 or current_metrics.get('jitter', 0) > 20:
                    advanced_suggestions.append({
                        'type': 'bandwidth_critical',
                        'metrics': ['Bandwidth', 'Packet Loss/Jitter'],
                        'suggestion': "Critical bandwidth with quality issues. Consider: 1) Immediate bandwidth upgrade required, 2) Implement bandwidth reservation for video calls, 3) Check for bandwidth throttling, 4) Consider using a dedicated connection for video calls"
                    })
        
        # Check for time-based patterns
        if 'timestamp' in current_metrics:
            hour = pd.to_datetime(current_metrics['timestamp'], unit='s').hour
            if 17 <= hour <= 20:  # Peak hours
                if current_metrics.get('packet_loss', 0) > 2 or current_metrics.get('latency', 0) > 100:
                    advanced_suggestions.append({
                        'type': 'time_based',
                        'metrics': ['Time of Day', 'Network Metrics'],
                        'suggestion': "Peak hour performance issues detected. Consider: 1) Schedule important calls outside peak hours, 2) Implement bandwidth reservation during peak hours, 3) Monitor network usage patterns, 4) Consider upgrading network capacity"
                    })
        
        return advanced_suggestions

    def get_improvement_suggestions(self, current_metrics):
        """Generate quality improvement suggestions"""
        if not self.is_trained:
            return None, "Model not trained"
        
        try:
            # Prepare single data point
            features = self.prepare_features(pd.DataFrame([current_metrics]))
            
            # Get feature importance
            importance = pd.DataFrame({
                'feature': features.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Generate basic suggestions based on individual metrics
            suggestions = []
            
            # Packet Loss Recommendations
            if 'packet_loss' in current_metrics:
                packet_loss = current_metrics['packet_loss']
                if packet_loss > 10:
                    suggestions.append({
                        'metric': 'Packet Loss',
                        'current': f"{packet_loss:.1f}%",
                        'suggestion': "Critical packet loss detected. Consider: 1) Check for network congestion, 2) Reduce video quality, 3) Use a wired connection instead of WiFi, 4) Close bandwidth-intensive applications"
                    })
                elif packet_loss > 5:
                    suggestions.append({
                        'metric': 'Packet Loss',
                        'current': f"{packet_loss:.1f}%",
                        'suggestion': "High packet loss. Consider: 1) Check for network interference, 2) Optimize network settings, 3) Consider using QoS settings on your router"
                    })
                elif packet_loss > 2:
                    suggestions.append({
                        'metric': 'Packet Loss',
                        'current': f"{packet_loss:.1f}%",
                        'suggestion': "Moderate packet loss. Consider: 1) Monitor network usage, 2) Check for background applications using bandwidth, 3) Ensure stable network connection"
                    })
            
            # Jitter Recommendations
            if 'jitter' in current_metrics:
                jitter = current_metrics['jitter']
                if jitter > 50:
                    suggestions.append({
                        'metric': 'Jitter',
                        'current': f"{jitter:.1f}ms",
                        'suggestion': "Severe jitter detected. Consider: 1) Check for network congestion, 2) Use a wired connection, 3) Enable QoS on your router, 4) Consider upgrading your network equipment"
                    })
                elif jitter > 20:
                    suggestions.append({
                        'metric': 'Jitter',
                        'current': f"{jitter:.1f}ms",
                        'suggestion': "High jitter. Consider: 1) Check for network interference, 2) Optimize network settings, 3) Consider using a network buffer"
                    })
                elif jitter > 10:
                    suggestions.append({
                        'metric': 'Jitter',
                        'current': f"{jitter:.1f}ms",
                        'suggestion': "Moderate jitter. Consider: 1) Monitor network stability, 2) Check for background processes, 3) Ensure consistent network connection"
                    })
            
            # Latency Recommendations
            if 'latency' in current_metrics:
                latency = current_metrics['latency']
                if latency > 200:
                    suggestions.append({
                        'metric': 'Latency',
                        'current': f"{latency:.1f}ms",
                        'suggestion': "Very high latency. Consider: 1) Check server distance, 2) Use a closer server if available, 3) Check for network congestion, 4) Consider upgrading your internet plan"
                    })
                elif latency > 100:
                    suggestions.append({
                        'metric': 'Latency',
                        'current': f"{latency:.1f}ms",
                        'suggestion': "High latency. Consider: 1) Check network path, 2) Optimize network settings, 3) Consider using a wired connection"
                    })
                elif latency > 50:
                    suggestions.append({
                        'metric': 'Latency',
                        'current': f"{latency:.1f}ms",
                        'suggestion': "Moderate latency. Consider: 1) Monitor network performance, 2) Check for background applications, 3) Ensure optimal network configuration"
                    })
            
            # Bandwidth Recommendations
            if 'bandwidth' in current_metrics:
                bandwidth = current_metrics['bandwidth']
                if bandwidth < 1:
                    suggestions.append({
                        'metric': 'Bandwidth',
                        'current': f"{bandwidth:.1f}Mbps",
                        'suggestion': "Critical bandwidth. Consider: 1) Upgrade your internet plan, 2) Reduce video quality, 3) Close bandwidth-intensive applications, 4) Check for network throttling"
                    })
                elif bandwidth < 2:
                    suggestions.append({
                        'metric': 'Bandwidth',
                        'current': f"{bandwidth:.1f}Mbps",
                        'suggestion': "Low bandwidth. Consider: 1) Check network usage, 2) Optimize video settings, 3) Consider upgrading your connection"
                    })
                elif bandwidth < 4:
                    suggestions.append({
                        'metric': 'Bandwidth',
                        'current': f"{bandwidth:.1f}Mbps",
                        'suggestion': "Moderate bandwidth. Consider: 1) Monitor bandwidth usage, 2) Check for background downloads, 3) Ensure optimal network settings"
                    })
            
            # Add general recommendations based on overall quality
            quality_score = 100 - (
                (current_metrics.get('packet_loss', 0) * 2) +
                (current_metrics.get('jitter', 0) / 2) +
                (current_metrics.get('latency', 0) / 10)
            )
            quality_score = max(0, min(100, quality_score))
            
            if quality_score < 50:
                suggestions.append({
                    'metric': 'Overall Quality',
                    'current': f"{quality_score:.1f}/100",
                    'suggestion': "Overall quality is poor. Consider: 1) Check all network metrics, 2) Contact your network administrator, 3) Consider using a different network connection, 4) Schedule a network maintenance check"
                })
            elif quality_score < 70:
                suggestions.append({
                    'metric': 'Overall Quality',
                    'current': f"{quality_score:.1f}/100",
                    'suggestion': "Overall quality needs improvement. Consider: 1) Review all network metrics, 2) Optimize network settings, 3) Check for network interference, 4) Monitor network performance"
                })
            
            # Add advanced rule-based recommendations
            advanced_suggestions = self._get_advanced_recommendations(current_metrics)
            suggestions.extend(advanced_suggestions)
            
            # Add AI-based predictions and confidence
            if len(features) > 0:
                features_scaled = self.scaler.transform(features)
                prediction = self.model.predict(features_scaled)[0]
                confidence = self.model.score(features_scaled, [prediction])
                
                suggestions.append({
                    'type': 'ai_prediction',
                    'metrics': ['AI Quality Score'],
                    'current': f"{prediction:.1f}/100",
                    'confidence': f"{confidence:.1%}",
                    'suggestion': f"AI predicts quality score of {prediction:.1f} with {confidence:.1%} confidence. " + 
                                ("Consider implementing all suggested improvements." if prediction < 70 else 
                                 "Current quality is acceptable, but can be improved with suggested optimizations.")
                })
            
            return {
                'suggestions': suggestions,
                'feature_importance': importance.to_dict('records')
            }
            
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            return None, str(e) 