import google.generativeai as genai
import os
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

class GeminiRecommender:
    def __init__(self, api_key: str):
        """Initialize Gemini API with the provided API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        self.cache = {}
        self.cache_duration = 600  # Cache results for 10 minutes
        self.last_request_time = 0
        self.min_request_interval = 5  # Minimum 5 seconds between requests
        self.daily_request_count = 0
        self.last_reset_time = datetime.now()
        self.MAX_DAILY_REQUESTS = 60  # Assuming a default MAX_DAILY_REQUESTS
        self.rules_file = 'recommendation_rules.json'
        self._load_rules()
        
    def _load_rules(self):
        """Load recommendation rules from JSON file"""
        if os.path.exists(self.rules_file):
            with open(self.rules_file, 'r') as f:
                self.rules = json.load(f)
        else:
            # Default rules if file doesn't exist
            self.rules = {
                'packet_loss': {
                    'critical': {'threshold': 10, 'weight': 2.0},
                    'high': {'threshold': 5, 'weight': 1.5},
                    'moderate': {'threshold': 2, 'weight': 1.0}
                },
                'latency': {
                    'critical': {'threshold': 200, 'weight': 2.0},
                    'high': {'threshold': 100, 'weight': 1.5},
                    'moderate': {'threshold': 50, 'weight': 1.0}
                },
                'jitter': {
                    'critical': {'threshold': 50, 'weight': 2.0},
                    'high': {'threshold': 20, 'weight': 1.5},
                    'moderate': {'threshold': 10, 'weight': 1.0}
                },
                'bandwidth': {
                    'critical': {'threshold': 1, 'weight': 2.0},
                    'high': {'threshold': 2, 'weight': 1.5},
                    'moderate': {'threshold': 4, 'weight': 1.0}
                }
            }
            self._save_rules()

    def _save_rules(self):
        """Save recommendation rules to JSON file"""
        with open(self.rules_file, 'w') as f:
            json.dump(self.rules, f, indent=4)

    def update_rules(self, new_rules: Dict):
        """Update recommendation rules"""
        self.rules.update(new_rules)
        self._save_rules()

    def _calculate_health_score(self, metrics: Dict) -> float:
        """Calculate overall network health score based on rules"""
        score = 100.0
        
        for metric, thresholds in self.rules.items():
            value = metrics.get(metric, 0)
            if metric == 'bandwidth':
                # For bandwidth, lower is worse
                for level, config in thresholds.items():
                    if value < config['threshold']:
                        score -= config['weight'] * (config['threshold'] - value)
            else:
                # For other metrics, higher is worse
                for level, config in thresholds.items():
                    if value > config['threshold']:
                        score -= config['weight'] * (value - config['threshold'])
        
        return max(0, min(100, score))

    def _get_metric_analysis(self, metric: str, value: float, thresholds: Dict) -> Dict:
        """Get analysis for a specific metric based on rules"""
        analysis = {
            'value': value,
            'level': 'normal',
            'weight': 0,
            'threshold': 0
        }
        
        for level, config in thresholds.items():
            if (metric == 'bandwidth' and value < config['threshold']) or \
               (metric != 'bandwidth' and value > config['threshold']):
                analysis.update({
                    'level': level,
                    'weight': config['weight'],
                    'threshold': config['threshold']
                })
        
        return analysis

    def _get_recommendations_from_rules(self, metrics: Dict) -> Dict:
        """Generate recommendations based on rules and metrics"""
        analyses = {}
        for metric, thresholds in self.rules.items():
            analyses[metric] = self._get_metric_analysis(metric, metrics.get(metric, 0), thresholds)
        
        health_score = self._calculate_health_score(metrics)
        
        # Generate recommendations based on analyses
        recommendations = []
        
        # Add metric-specific recommendations
        for metric, analysis in analyses.items():
            if analysis['level'] != 'normal':
                recommendations.append(f"{metric.upper()} Analysis ({analysis['value']}):")
                recommendations.append(f"• Level: {analysis['level']}")
                recommendations.append(f"• Threshold: {analysis['threshold']}")
                recommendations.append("• Recommendations:")
                recommendations.append("  - Monitor and optimize network settings")
                recommendations.append("  - Check for interference or congestion")
                recommendations.append("  - Consider upgrading if persistent")
                recommendations.append("  - Implement appropriate QoS settings")
                recommendations.append("")
        
        # Add overall health assessment
        recommendations.append(f"Overall Network Health Score: {health_score:.1f}/100")
        if health_score < 50:
            recommendations.append("• Status: Critical - Immediate attention required")
        elif health_score < 70:
            recommendations.append("• Status: Needs Improvement - Monitor closely")
        else:
            recommendations.append("• Status: Good - Continue monitoring")
        
        return {
            'analysis': "\n".join(recommendations),
            'metrics': metrics,
            'timestamp': metrics.get('timestamp', ''),
            'is_fallback': True,
            'health_score': health_score
        }

    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        
        self.last_request_time = time.time()
    
    def _get_cached_result(self, key: str) -> Optional[Dict]:
        """Get cached result if available and not expired"""
        if key in self.cache:
            timestamp, result = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_duration):
                return result
        return None
    
    def _cache_result(self, key: str, result: Dict):
        """Cache a result with current timestamp"""
        self.cache[key] = (datetime.now(), result)
    
    def _get_fallback_recommendations(self, metrics: Dict) -> Dict:
        """Provide fallback recommendations when API is unavailable"""
        packet_loss = metrics.get('packet_loss', 0)
        latency = metrics.get('latency', 0)
        jitter = metrics.get('jitter', 0)
        bandwidth = metrics.get('bandwidth', 0)
        
        recommendations = []
        
        # Packet Loss Analysis
        if packet_loss > 10:
            recommendations.append("""Critical Packet Loss ({}%):
• Root Cause: Likely network congestion or connection instability
• Immediate Actions:
  - Check for network congestion by monitoring bandwidth usage
  - Reduce video quality settings
  - Switch to a wired connection if using WiFi
  - Close bandwidth-intensive applications
• Long-term Solutions:
  - Consider upgrading your internet plan
  - Implement QoS (Quality of Service) settings on your router
  - Use a network monitoring tool to identify patterns
  - Consider using a different network path or VPN""".format(packet_loss))
        elif packet_loss > 5:
            recommendations.append("""High Packet Loss ({}%):
• Root Cause: Possible network interference or suboptimal settings
• Immediate Actions:
  - Check for network interference sources
  - Optimize network settings
  - Monitor application bandwidth usage
  - Consider using a network buffer
• Long-term Solutions:
  - Implement QoS settings
  - Update network drivers
  - Consider network path optimization
  - Monitor and maintain network equipment""".format(packet_loss))
        elif packet_loss > 2:
            recommendations.append("""Moderate Packet Loss ({}%):
• Root Cause: Normal network variation or minor interference
• Immediate Actions:
  - Monitor network usage patterns
  - Check for background applications
  - Ensure stable network connection
  - Verify network settings
• Long-term Solutions:
  - Regular network maintenance
  - Keep network drivers updated
  - Monitor network performance trends
  - Implement basic QoS settings""".format(packet_loss))
            
        # Latency Analysis
        if latency > 200:
            recommendations.append("""Very High Latency ({}ms):
• Root Cause: Distance to server or network congestion
• Immediate Actions:
  - Check server distance and consider using a closer server
  - Verify network path efficiency
  - Check for network congestion
  - Consider using a wired connection
• Long-term Solutions:
  - Upgrade internet plan if bandwidth is limited
  - Implement network optimization
  - Consider using a CDN
  - Regular network path analysis""".format(latency))
        elif latency > 100:
            recommendations.append("""High Latency ({}ms):
• Root Cause: Suboptimal network path or settings
• Immediate Actions:
  - Check network path efficiency
  - Optimize network settings
  - Consider using a wired connection
  - Monitor background processes
• Long-term Solutions:
  - Network path optimization
  - Regular network maintenance
  - Update network equipment
  - Consider network upgrades""".format(latency))
        elif latency > 50:
            recommendations.append("""Moderate Latency ({}ms):
• Root Cause: Normal network conditions
• Immediate Actions:
  - Monitor network performance
  - Check for background applications
  - Verify network settings
  - Ensure optimal configuration
• Long-term Solutions:
  - Regular performance monitoring
  - Network optimization
  - Keep equipment updated
  - Maintain network health""".format(latency))
            
        # Jitter Analysis
        if jitter > 50:
            recommendations.append("""Severe Jitter ({}ms):
• Root Cause: Network instability or congestion
• Immediate Actions:
  - Enable QoS on your router
  - Check for network congestion
  - Use a wired connection
  - Monitor network stability
• Long-term Solutions:
  - Upgrade network equipment
  - Implement traffic shaping
  - Regular network maintenance
  - Consider network path optimization""".format(jitter))
        elif jitter > 20:
            recommendations.append("""High Jitter ({}ms):
• Root Cause: Network interference or suboptimal settings
• Immediate Actions:
  - Check for network interference
  - Optimize network settings
  - Use a network buffer
  - Monitor network stability
• Long-term Solutions:
  - Network optimization
  - Regular maintenance
  - Update network drivers
  - Consider equipment upgrades""".format(jitter))
        elif jitter > 10:
            recommendations.append("""Moderate Jitter ({}ms):
• Root Cause: Normal network variation
• Immediate Actions:
  - Monitor network stability
  - Check background processes
  - Verify network settings
  - Ensure consistent connection
• Long-term Solutions:
  - Regular monitoring
  - Network maintenance
  - Keep equipment updated
  - Optimize settings""".format(jitter))
            
        # Bandwidth Analysis
        if bandwidth < 1:
            recommendations.append("""Critical Bandwidth ({}Mbps):
• Root Cause: Limited bandwidth or network congestion
• Immediate Actions:
  - Reduce video quality
  - Close bandwidth-intensive applications
  - Check for network throttling
  - Monitor bandwidth usage
• Long-term Solutions:
  - Upgrade internet plan
  - Optimize network settings
  - Consider bandwidth management
  - Regular network maintenance""".format(bandwidth))
        elif bandwidth < 2:
            recommendations.append("""Low Bandwidth ({}Mbps):
• Root Cause: Limited bandwidth allocation
• Immediate Actions:
  - Check network usage
  - Optimize video settings
  - Monitor bandwidth consumption
  - Close unnecessary applications
• Long-term Solutions:
  - Consider plan upgrade
  - Network optimization
  - Regular monitoring
  - Bandwidth management""".format(bandwidth))
        elif bandwidth < 4:
            recommendations.append("""Moderate Bandwidth ({}Mbps):
• Root Cause: Normal bandwidth allocation
• Immediate Actions:
  - Monitor bandwidth usage
  - Check for background downloads
  - Optimize network settings
  - Verify connection stability
• Long-term Solutions:
  - Regular monitoring
  - Network optimization
  - Keep equipment updated
  - Maintain network health""".format(bandwidth))
        
        # Add overall network health assessment
        overall_score = 100 - (
            (packet_loss * 2) +
            (latency / 10) +
            (jitter / 2) +
            (max(0, 4 - bandwidth) * 10)
        )
        overall_score = max(0, min(100, overall_score))
        
        if overall_score < 50:
            recommendations.append("""Overall Network Health: Critical
• Current Status: Network performance is significantly degraded
• Immediate Priority:
  - Address critical issues first (packet loss, latency)
  - Implement all recommended immediate actions
  - Consider temporary quality reduction
• Long-term Strategy:
  - Comprehensive network upgrade plan
  - Regular monitoring and maintenance
  - Consider professional network assessment
  - Implement all recommended long-term solutions""")
        elif overall_score < 70:
            recommendations.append("""Overall Network Health: Needs Improvement
• Current Status: Network performance is below optimal
• Immediate Priority:
  - Address high-priority issues
  - Implement key recommendations
  - Monitor improvements
• Long-term Strategy:
  - Regular performance monitoring
  - Gradual implementation of improvements
  - Network optimization plan
  - Regular maintenance schedule""")
        else:
            recommendations.append("""Overall Network Health: Good
• Current Status: Network performance is acceptable
• Immediate Priority:
  - Monitor for any degradation
  - Maintain current settings
  - Watch for trends
• Long-term Strategy:
  - Regular monitoring
  - Preventive maintenance
  - Keep systems updated
  - Document performance patterns""")
        
        return {
            'analysis': "\n\n".join(recommendations),
            'metrics': metrics,
            'timestamp': metrics.get('timestamp', ''),
            'is_fallback': True
        }
    
    def _create_prompt(self, metrics: Dict) -> str:
        """Create a detailed prompt for the Gemini model"""
        return f"""
        Analyze these network metrics and provide concise recommendations:
        
        Metrics:
        - Packet Loss: {metrics.get('packet_loss', 0)}%
        - Latency: {metrics.get('latency', 0)}ms
        - Jitter: {metrics.get('jitter', 0)}ms
        - Bandwidth: {metrics.get('bandwidth', 0)}Mbps
        
        Provide:
        1. Main issues identified
        2. Key recommendations
        3. Expected improvements
        
        Keep the response brief and focused.
        """
    
    def get_recommendations(self, metrics: Dict) -> Dict:
        """Get AI-powered recommendations for network optimization"""
        try:
            # Check cache first
            cache_key = f"recommendations_{metrics.get('timestamp', '')}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            # Check daily request limit
            if self.daily_request_count >= self.MAX_DAILY_REQUESTS:
                return self._get_recommendations_from_rules(metrics)
            
            # Try different prompts in sequence
            prompts = [
                # First attempt: Detailed technical analysis with context
                f"""As a network optimization expert, analyze these network metrics and provide detailed recommendations:

Current Network Metrics:
- Packet Loss: {metrics.get('packet_loss', 0)}%
- Latency: {metrics.get('latency', 0)}ms
- Jitter: {metrics.get('jitter', 0)}ms
- Bandwidth: {metrics.get('bandwidth', 0)}Mbps

Context:
- This is a real-time network monitoring system
- Recommendations should be actionable and specific
- Consider both immediate fixes and long-term solutions
- Focus on practical, implementable solutions

Please provide a comprehensive analysis with the following structure:

1. For each problematic metric:
   - Current value and units
   - Root cause analysis
   - Immediate actions (4 specific steps)
   - Long-term solutions (4 specific steps)
   - Potential impact if not addressed

2. Overall network health assessment:
   - Current status
   - Immediate priorities
   - Long-term strategy
   - Risk assessment

3. Additional considerations:
   - Cost implications
   - Implementation difficulty
   - Expected improvement
   - Monitoring recommendations

Format the response with clear sections and bullet points for better readability.
Include specific values from the metrics in your analysis.
Be concise but detailed in your recommendations.""",

                # Second attempt: User-focused recommendations
                f"""Network Optimization Analysis Request:

Current Network Performance:
- Packet Loss: {metrics.get('packet_loss', 0)}%
- Latency: {metrics.get('latency', 0)}ms
- Jitter: {metrics.get('jitter', 0)}ms
- Bandwidth: {metrics.get('bandwidth', 0)}Mbps

Please provide user-friendly recommendations focusing on:

1. For each issue identified:
   - Problem description in simple terms
   - Why it's happening
   - What users can do right now
   - What users should do long-term
   - What to expect after fixes

2. Overall network status:
   - Current health level
   - Most urgent issues
   - Quick wins
   - Long-term improvements

3. User guidance:
   - Step-by-step instructions
   - Common mistakes to avoid
   - When to seek professional help
   - How to monitor improvements

Use clear, non-technical language where possible.
Format with bullet points and clear sections.""",

                # Third attempt: Critical issues with troubleshooting
                f"""Network Troubleshooting Analysis:

Current Metrics:
- Packet Loss: {metrics.get('packet_loss', 0)}%
- Latency: {metrics.get('latency', 0)}ms
- Jitter: {metrics.get('jitter', 0)}ms
- Bandwidth: {metrics.get('bandwidth', 0)}Mbps

Please provide a troubleshooting guide focusing on:

1. Critical Issues:
   - Problem identification
   - Severity assessment
   - Impact analysis
   - Priority ranking

2. Troubleshooting Steps:
   - Immediate actions
   - Verification steps
   - Common solutions
   - Advanced fixes

3. Prevention:
   - Early warning signs
   - Regular maintenance
   - Best practices
   - Monitoring tips

Format with clear sections and bullet points.
Include specific metrics in the analysis.""",

                # Fourth attempt: Performance optimization focus
                f"""Network Performance Optimization Request:

Current Performance Metrics:
- Packet Loss: {metrics.get('packet_loss', 0)}%
- Latency: {metrics.get('latency', 0)}ms
- Jitter: {metrics.get('jitter', 0)}ms
- Bandwidth: {metrics.get('bandwidth', 0)}Mbps

Please provide optimization recommendations focusing on:

1. Performance Analysis:
   - Current bottlenecks
   - Resource utilization
   - Optimization opportunities
   - Performance targets

2. Optimization Steps:
   - Quick improvements
   - Configuration changes
   - Hardware considerations
   - Software solutions

3. Monitoring and Maintenance:
   - Key metrics to watch
   - Regular checks
   - Performance baselines
   - Improvement tracking

Format with clear sections and bullet points.
Include specific metrics and targets.""",

                # Fifth attempt: Emergency response
                f"""Network Emergency Analysis:

Critical Metrics:
- Packet Loss: {metrics.get('packet_loss', 0)}%
- Latency: {metrics.get('latency', 0)}ms
- Jitter: {metrics.get('jitter', 0)}ms
- Bandwidth: {metrics.get('bandwidth', 0)}Mbps

Please provide emergency response recommendations:

1. Immediate Actions:
   - Critical issues to address
   - Emergency fixes
   - Temporary solutions
   - Impact mitigation

2. Recovery Steps:
   - System restoration
   - Performance recovery
   - Stability measures
   - Prevention steps

3. Emergency Plan:
   - Priority actions
   - Resource allocation
   - Communication steps
   - Recovery timeline

Format with clear sections and bullet points.
Focus on immediate actionable steps."""
            ]

            # Try each prompt until we get a good response
            for prompt in prompts:
                try:
                    # Wait to respect rate limits
                    self._wait_for_rate_limit()
                    
                    # Get response from Gemini
                    response = self.model.generate_content(prompt)
                    recommendations = response.text
                    
                    # Enhanced quality check
                    if len(recommendations.strip()) > 100 and self._is_valid_recommendation(recommendations):
                        # Increment request counter
                        self.daily_request_count += 1
                        
                        # Cache the results
                        result = {
                            'analysis': recommendations,
                            'metrics': metrics,
                            'timestamp': metrics.get('timestamp', ''),
                            'is_fallback': False
                        }
                        self._cache_result(cache_key, result)
                        
                        return result
                    
                except Exception as e:
                    print(f"Error with prompt attempt: {str(e)}")
                    continue
            
            # If all prompts fail, fall back to rules
            print("All Gemini attempts failed, falling back to rules")
            return self._get_recommendations_from_rules(metrics)
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return self._get_recommendations_from_rules(metrics)
    
    def _is_valid_recommendation(self, text: str) -> bool:
        """Check if the recommendation text is valid and well-structured"""
        # Check for minimum length
        if len(text.strip()) < 100:
            return False
            
        # Check for basic structure
        required_sections = ['•', '-', ':']
        if not any(section in text for section in required_sections):
            return False
            
        # Check for metric references
        metric_terms = ['packet', 'loss', 'latency', 'jitter', 'bandwidth']
        if not any(term in text.lower() for term in metric_terms):
            return False
            
        return True

    def get_historical_analysis(self, historical_data: List[Dict]) -> Dict:
        """Analyze historical data patterns with caching"""
        try:
            # Calculate averages
            avg_metrics = {
                'packet_loss': sum(d.get('packet_loss', 0) for d in historical_data) / len(historical_data),
                'latency': sum(d.get('latency', 0) for d in historical_data) / len(historical_data),
                'jitter': sum(d.get('jitter', 0) for d in historical_data) / len(historical_data),
                'bandwidth': sum(d.get('bandwidth', 0) for d in historical_data) / len(historical_data)
            }
            
            # Create cache key
            cache_key = f"historical_{avg_metrics['packet_loss']}_{avg_metrics['latency']}_{avg_metrics['jitter']}_{avg_metrics['bandwidth']}"
            
            # Check cache first
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Check if we've hit daily limit
            if self.daily_request_count >= 60:  # Conservative daily limit
                return {
                    'analysis': self._get_fallback_recommendations(avg_metrics)['analysis'],
                    'average_metrics': avg_metrics,
                    'data_points': len(historical_data),
                    'is_fallback': True
                }
            
            # Wait for rate limit
            self._wait_for_rate_limit()
            
            prompt = f"""
            Analyze these network metrics and provide concise insights:
            
            Average Metrics:
            - Packet Loss: {avg_metrics['packet_loss']:.2f}%
            - Latency: {avg_metrics['latency']:.2f}ms
            - Jitter: {avg_metrics['jitter']:.2f}ms
            - Bandwidth: {avg_metrics['bandwidth']:.2f}Mbps
            
            Data points: {len(historical_data)}
            
            Provide:
            1. Overall assessment
            2. Key patterns
            3. Main recommendations
            
            Keep the response brief and focused.
            """
            
            response = self.model.generate_content(prompt)
            
            result = {
                'analysis': response.text,
                'average_metrics': avg_metrics,
                'data_points': len(historical_data),
                'is_fallback': False
            }
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            # Update request count
            self.daily_request_count += 1
            
            return result
            
        except Exception as e:
            print(f"Error getting historical analysis: {e}")
            return {
                'analysis': self._get_fallback_recommendations(avg_metrics)['analysis'],
                'average_metrics': avg_metrics if 'avg_metrics' in locals() else {},
                'data_points': len(historical_data),
                'is_fallback': True
            } 