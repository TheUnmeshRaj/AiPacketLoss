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
            recommendations.append(f"""Critical Packet Loss ({packet_loss}%):
• Root Cause: Severe network congestion, damaged cabling, faulty hardware (router, modem), or significant wireless interference.
• Immediate Actions:
  - Restart your router and modem.
  - Disconnect unnecessary devices from your network.
  - Use a wired (Ethernet) connection if on Wi-Fi.
  - Temporarily reduce streaming quality or stop large downloads.
• Long-term Solutions:
  - Inspect and replace damaged Ethernet cables.
  - Upgrade aging network hardware (router, modem).
  - Implement QoS (Quality of Service) settings on your router to prioritize critical traffic (e.g., video calls).
  - Contact your Internet Service Provider (ISP) to report network issues or inquire about line quality.
  - Consider a mesh Wi-Fi system if you have coverage issues.
  - Use network analysis tools to identify specific sources of congestion.
""".format(packet_loss))
        elif packet_loss > 5:
            recommendations.append(f"""High Packet Loss ({packet_loss}%):
• Root Cause: Moderate network congestion, Wi-Fi interference, outdated network drivers, or minor hardware issues.
• Immediate Actions:
  - Move closer to your Wi-Fi router or eliminate physical obstructions.
  - Update your network adapter drivers.
  - Check for background applications consuming bandwidth.
  - Temporarily disable VPNs or proxies.
• Long-term Solutions:
  - Change your Wi-Fi channel to avoid interference.
  - Optimize router settings (e.g., enable WMM, disable old protocols).
  - Consider a Wi-Fi extender or upgrading to a dual-band router.
  - Regularly monitor network performance to identify patterns.
  - Implement a basic QoS policy.
""".format(packet_loss))
        elif packet_loss > 2:
            recommendations.append(f"""Moderate Packet Loss ({packet_loss}%):
• Root Cause: Minor network fluctuations, occasional Wi-Fi instability, or background network activity.
• Immediate Actions:
  - Pause large downloads or updates.
  - Ensure no other devices are heavily using the network.
  - Restart your device.
  - Briefly disable and re-enable your network adapter.
• Long-term Solutions:
  - Keep operating system and application software updated.
  - Consider using powerline adapters for stable connections where Wi-Fi is weak.
  - Perform regular network maintenance and checks.
  - Use a network monitoring dashboard to track trends.
""".format(packet_loss))
            
        # Latency Analysis
        if latency > 200:
            recommendations.append(f"""Very High Latency ({latency}ms):
• Root Cause: Significant geographical distance to server, severe network congestion, or multiple hops/poor routing.
• Immediate Actions:
  - Try connecting to a server closer to your location if applicable.
  - Reduce the number of active network connections.
  - Temporarily disable firewalls or security software.
  - Avoid peak network usage hours if possible.
• Long-term Solutions:
  - Upgrade your internet plan for higher bandwidth if bandwidth is a bottleneck.
  - Implement network optimization techniques (e.g., fast path, traffic shaping).
  - Use a VPN with servers closer to your target destination.
  - Consider a CDN (Content Delivery Network) for content delivery.
  - Consult with your ISP about routing optimizations.
""".format(latency))
        elif latency > 100:
            recommendations.append(f"""High Latency ({latency}ms):
• Root Cause: Suboptimal network routing, moderate congestion, or outdated network equipment.
• Immediate Actions:
  - Close unnecessary browser tabs and applications.
  - Check for any ongoing system updates.
  - Ensure your device is not overheating.
  - Try restarting your router.
• Long-term Solutions:
  - Optimize router settings (e.g., enable QoS, update firmware).
  - Consider upgrading to a newer router or modem.
  - Regularly clean up temporary internet files and browser cache.
  - Investigate alternative DNS servers (e.g., Google DNS, Cloudflare DNS).
""".format(latency))
        elif latency > 50:
            recommendations.append(f"""Moderate Latency ({latency}ms):
• Root Cause: Normal internet routing, minor local network activity, or distance to server.
• Immediate Actions:
  - Limit background data usage.
  - Verify no large files are downloading.
  - Ensure your device is connected to the optimal Wi-Fi band (2.4GHz vs 5GHz).
  - Check for any open applications that might be connecting to distant servers.
• Long-term Solutions:
  - Maintain network health through regular router reboots.
  - Consider reducing the number of devices on your network.
  - Ensure sufficient bandwidth for your household needs.
  - Optimize in-home network cabling for stability.
""".format(latency))
            
        # Jitter Analysis
        if jitter > 50:
            recommendations.append(f"""Severe Jitter ({jitter}ms):
• Root Cause: Highly unstable network connection, severe congestion, or significant packet reordering.
• Immediate Actions:
  - Enable QoS on your router to prioritize real-time traffic.
  - Close all other network-intensive applications.
  - Use a wired Ethernet connection.
  - If using Wi-Fi, try moving closer to the router or reduce interference sources.
• Long-term Solutions:
  - Upgrade network equipment with better buffering capabilities.
  - Implement traffic shaping policies on your router.
  - Consider a dedicated internet line for critical applications.
  - Regularly update router firmware and device drivers.
  - Conduct detailed network diagnostics to identify the source of instability.
""".format(jitter))
        elif jitter > 20:
            recommendations.append(f"""High Jitter ({jitter}ms):
• Root Cause: Unstable network, moderate congestion, or varying network load.
• Immediate Actions:
  - Minimize concurrent network activity.
  - Check for bandwidth-hogging applications.
  - Restart your router and device.
  - If possible, ensure your device is the primary user on the network.
• Long-term Solutions:
  - Ensure your router supports 5GHz Wi-Fi for less interference.
  - Implement bufferbloat mitigation techniques.
  - Consider a more robust internet plan if bandwidth is consistently saturated.
  - Regularly clear network caches on devices.
""".format(jitter))
        elif jitter > 10:
            recommendations.append(f"""Moderate Jitter ({jitter}ms):
• Root Cause: Normal network variations, minor background network activity, or slight inconsistencies in packet arrival.
• Immediate Actions:
  - Ensure background applications are not active.
  - Check for any ongoing software updates.
  - Briefly disconnect and reconnect to the network.
  - Verify that your router is not placed in an enclosed space.
• Long-term Solutions:
  - Implement consistent network maintenance schedules.
  - Consider upgrading to a router with better processor performance.
  - Review network device placement for optimal signal.
  - Keep device operating systems and applications updated.
""".format(jitter))
            
        # Bandwidth Analysis
        if bandwidth < 1:
            recommendations.append(f"""Critical Bandwidth ({bandwidth}Mbps):
• Root Cause: Severely limited internet plan, heavy network saturation, or significant issues with ISP connectivity.
• Immediate Actions:
  - Reduce video stream quality to the lowest setting.
  - Close all other applications and devices using the internet.
  - Restart your modem and router.
  - Contact your ISP immediately to check for outages or throttling.
• Long-term Solutions:
  - **Urgent:** Upgrade your internet plan to meet your usage needs.
  - Implement strict bandwidth management and QoS policies.
  - Consider a dedicated business internet line if critical for operations.
  - Analyze network traffic to identify bandwidth-hogging applications or users.
""".format(bandwidth))
        elif bandwidth < 2:
            recommendations.append(f"""Low Bandwidth ({bandwidth}Mbps):
• Root Cause: Insufficient internet plan for current usage, multiple concurrent users, or background downloads/updates.
• Immediate Actions:
  - Close all unused applications and browser tabs.
  - Ask other users on your network to reduce their internet activity.
  - Check for cloud syncs or large updates running in the background.
  - Temporarily reduce resolution or quality of streaming services.
• Long-term Solutions:
  - Upgrade your internet plan to a higher speed tier.
  - Implement a robust QoS strategy on your router to prioritize essential traffic.
  - Schedule large downloads/updates for off-peak hours.
  - Educate household members on bandwidth-intensive activities.
""".format(bandwidth))
        elif bandwidth < 4:
            recommendations.append(f"""Moderate Bandwidth ({bandwidth}Mbps):
• Root Cause: Adequate but potentially strained bandwidth, common during peak usage or with multiple active streams.
• Immediate Actions:
  - Optimize video quality settings to a comfortable level.
  - Ensure no large files are uploading/downloading.
  - Check for other devices consuming bandwidth.
  - Consider using a browser extension to block ads and trackers that consume bandwidth.
• Long-term Solutions:
  - Regularly monitor bandwidth usage to understand patterns.
  - Consider optimizing router settings for better throughput.
  - Review your internet plan periodically against your actual usage.
  - Ensure all network drivers are up-to-date for optimal performance.
""".format(bandwidth))
        
        # Add overall network health assessment
        overall_score = 100 - (
            (packet_loss * 2) +
            (latency / 10) +
            (jitter / 2) +
            (max(0, 4 - bandwidth) * 10)
        )
        overall_score = max(0, min(100, overall_score))
        
        if overall_score < 50:
            recommendations.append("""Overall Network Health: Critical - Immediate Attention Required
• Current Status: Your network performance is severely compromised, likely impacting all online activities, especially real-time applications like video calls. This requires urgent intervention.
• Immediate Priority:
  - Focus on addressing the most critical metrics first (e.g., packet loss > 10%, latency > 200ms).
  - Implement all recommended immediate actions for the problematic metrics without delay.
  - Temporarily reduce the quality settings of all online services to conserve bandwidth.
• Long-term Strategy:
  - Develop a comprehensive network upgrade and optimization plan.
  - Engage professional network assessment if issues persist.
  - Prioritize long-term solutions such as ISP upgrades, new hardware, and advanced QoS configurations.
  - Establish continuous monitoring to track recovery and prevent future degradation.
""")
        elif overall_score < 70:
            recommendations.append("""Overall Network Health: Needs Improvement - Monitor Closely
• Current Status: Your network performance is below optimal, leading to noticeable disruptions, particularly in video calls. Improvements are needed to enhance stability and quality.
• Immediate Priority:
  - Address high-priority issues (e.g., packet loss > 5%, latency > 100ms, jitter > 20ms).
  - Implement key recommendations for specific metrics to achieve quick wins.
  - Monitor closely to observe the impact of applied changes.
• Long-term Strategy:
  - Plan for gradual implementation of recommended improvements.
  - Optimize network configurations and regularly maintain your equipment.
  - Consider incremental upgrades to your internet plan or hardware if persistent issues are observed.
  - Focus on consistent performance monitoring to identify recurring problems.
""")
        else:
            recommendations.append("""Overall Network Health: Good - Maintain and Monitor
• Current Status: Your network performance is generally stable and meets most requirements. Occasional minor fluctuations are normal.
• Immediate Priority:
  - Continue monitoring for any signs of degradation.
  - Maintain current network settings and configurations.
  - No immediate actions are typically required unless specific symptoms arise.
• Long-term Strategy:
  - Implement preventive maintenance practices (e.g., regular router reboots).
  - Keep network devices and software updated.
  - Document performance baselines to easily detect future anomalies.
  - Stay informed about new network technologies that could further enhance performance.
""")
        
        return {
            'analysis': "\n\n".join(recommendations),
            'metrics': metrics,
            'timestamp': metrics.get('timestamp', ''),
            'is_fallback': True
        }
    
    def _create_prompt(self, metrics: Dict) -> str:
        """Create a detailed prompt for the Gemini model for historical analysis"""
        return f"""
        As a network performance expert, analyze the following *average* historical network metrics and provide comprehensive insights.

        Average Metrics:
        - Packet Loss: {metrics.get('packet_loss', 0):.2f}%
        - Latency: {metrics.get('latency', 0):.2f}ms
        - Jitter: {metrics.get('jitter', 0):.2f}ms
        - Bandwidth: {metrics.get('bandwidth', 0):.2f}Mbps

        Based on these averages, identify:
        1. Overall network health assessment (e.g., Excellent, Good, Moderate, Poor, Critical).
        2. Key historical patterns or persistent issues for each metric.
        3. Main strategic recommendations for long-term optimization and stability.
        4. Potential impact on user experience (e.g., video calls, gaming, browsing).

        Provide a structured response with clear headings and bullet points for each section.
        """
    
    def get_recommendations(self, metrics: Dict) -> Dict:
        """Get AI-powered recommendations for network optimization"""
        try:
            # Check cache first
            cache_key = f"recommendations_{metrics.get('timestamp', '')}"
            if self._get_cached_result(cache_key):
                return self._get_cached_result(cache_key)
            
            # Check daily request limit
            if self.daily_request_count >= self.MAX_DAILY_REQUESTS:
                print("Daily Gemini request limit reached, falling back to rules-based recommendations.")
                return self._get_recommendations_from_rules(metrics)
            
            # Try different prompts in sequence
            prompts = [
                # First attempt: Detailed technical analysis with context and thresholds
                f"""As a senior network optimization expert, analyze these real-time network metrics and provide a highly detailed, actionable analysis with specific recommendations.

Current Network Metrics:
- Packet Loss: {metrics.get('packet_loss', 0)}%
- Latency: {metrics.get('latency', 0)}ms
- Jitter: {metrics.get('jitter', 0)}ms
- Bandwidth: {metrics.get('bandwidth', 0)}Mbps

Context:
- This data comes from a live video call quality monitoring system.
- Recommendations should be prioritized by impact and feasibility.
- Consider both immediate, temporary fixes and long-term, structural solutions.
- Include hardware, software, and configuration advice where relevant.
- Address potential root causes (e.g., congestion, interference, distance, QoS, equipment limitations).

Please structure your response with the following sections:

1.  **Overall Network Health Assessment**:
    -   A concise summary (e.g., "Good", "Fair", "Poor", "Critical").
    -   Primary areas of concern.

2.  **Detailed Metric Analysis & Recommendations**: For each metric, if problematic:
    -   **Packet Loss ({metrics.get('packet_loss', 0)}%)**:
        -   Severity (e.g., Critical (>10%), High (5-10%), Moderate (2-5%)).
        -   Probable Root Causes.
        -   Immediate Actions (e.g., "Check local network cables", "Reduce concurrent streaming").
        -   Long-term Solutions (e.g., "Upgrade router firmware", "Implement QoS for video traffic", "Contact ISP").
        -   Expected Impact if Addressed.
    -   **Latency ({metrics.get('latency', 0)}ms)**:
        -   Severity (e.g., Critical (>200ms), High (100-200ms), Moderate (50-100ms)).
        -   Probable Root Causes.
        -   Immediate Actions.
        -   Long-term Solutions.
        -   Expected Impact if Addressed.
    -   **Jitter ({metrics.get('jitter', 0)}ms)**:
        -   Severity (e.g., Critical (>50ms), High (20-50ms), Moderate (10-20ms)).
        -   Probable Root Causes.
        -   Immediate Actions.
        -   Long-term Solutions.
        -   Expected Impact if Addressed.
    -   **Bandwidth ({metrics.get('bandwidth', 0)}Mbps)**:
        -   Severity (e.g., Critical (<1Mbps), Low (1-2Mbps), Moderate (2-4Mbps)).
        -   Probable Root Causes (e.g., oversaturation, plan limitations, multiple users).
        -   Immediate Actions.
        -   Long-term Solutions.
        -   Expected Impact if Addressed.

3.  **Overall Action Plan & Monitoring**:
    -   Prioritized steps to take.
    -   Key metrics to monitor for improvement.
    -   When to consider professional assistance.

Ensure all recommendations are specific, practical, and directly address the identified metric values. Use clear, concise language.
""",

                # Second attempt: User-focused, simplified recommendations
                f"""Explain the current network issues in simple terms and provide easy-to-follow, actionable recommendations for a typical home user.

Current Network Performance:
- Packet Loss: {metrics.get('packet_loss', 0)}%
- Latency: {metrics.get('latency', 0)}ms
- Jitter: {metrics.get('jitter', 0)}ms
- Bandwidth: {metrics.get('bandwidth', 0)}Mbps

Focus on:
1.  **What's wrong?** (Simple explanation of each problematic metric).
2.  **Why is it happening?** (Common causes).
3.  **Quick fixes you can try now.**
4.  **Things to do for a long-term solution.**
5.  **Overall advice** for a stable connection.

Use bullet points for clarity. Keep it straightforward and avoid technical jargon where possible.
""",

                # Third attempt: Troubleshooting guide
                f"""Act as a network troubleshooting assistant. Based on these metrics, guide the user through a step-by-step troubleshooting process.

Current Metrics:
- Packet Loss: {metrics.get('packet_loss', 0)}%
- Latency: {metrics.get('latency', 0)}ms
- Jitter: {metrics.get('jitter', 0)}ms
- Bandwidth: {metrics.get('bandwidth', 0)}Mbps

Provide:
1.  **Problem Symptoms**: What the user might be experiencing.
2.  **Troubleshooting Steps**:
    -   Basic checks (e.g., "Restart your router").
    -   Intermediate diagnostics (e.g., "Run a speed test").
    -   Advanced solutions (e.g., "Check router settings for QoS").
3.  **When to Escalate**: Advice on when to contact their ISP or a technician.

Structure as a clear troubleshooting flow.
""",

                # Fourth attempt: Performance optimization focus
                f"""As a network performance optimizer, analyze these metrics and suggest ways to improve video call quality specifically.

Current Performance Metrics:
- Packet Loss: {metrics.get('packet_loss', 0)}%
- Latency: {metrics.get('latency', 0)}ms
- Jitter: {metrics.get('jitter', 0)}ms
- Bandwidth: {metrics.get('bandwidth', 0)}Mbps

Focus on optimization for video conferencing. Suggest:
1.  **Immediate Optimization Tips**: Quick changes for better call quality.
2.  **Configuration Adjustments**: Router or device settings.
3.  **Network Hardware/Software Recommendations**: If upgrades are necessary.
4.  **Best Practices**: For consistent video call performance.

Keep recommendations actionable and relevant to video quality.
""",

                # Fifth attempt: Emergency response (more concise)
                f"""You are an emergency network response AI. Based on the following critical network metrics, provide immediate, high-priority actions to stabilize the connection.

Critical Metrics:
- Packet Loss: {metrics.get('packet_loss', 0)}%
- Latency: {metrics.get('latency', 0)}ms
- Jitter: {metrics.get('jitter', 0)}ms
- Bandwidth: {metrics.get('bandwidth', 0)}Mbps

Prioritize actions that will have the quickest and most significant impact on network stability.
Provide a clear, numbered list of actions, followed by brief explanations.
"""
            ]

            # Try each prompt until we get a good response
            for i, prompt in enumerate(prompts):
                try:
                    # Wait to respect rate limits
                    self._wait_for_rate_limit()
                    
                    # Get response from Gemini
                    response = self.model.generate_content(prompt)
                    recommendations = response.text
                    
                    # Enhanced quality check: ensure response is substantial and seems valid
                    if len(recommendations.strip()) > 150 and self._is_valid_recommendation(recommendations): # Increased min length
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
                    else:
                        print(f"Gemini response for prompt {i+1} was too short or invalid. Trying next prompt.")
                    
                except Exception as e:
                    print(f"Error with Gemini prompt attempt {i+1}: {str(e)}")
                    # Continue to next prompt if API call fails
                    continue
            
            # If all prompts fail, fall back to rules
            print("All Gemini attempts failed or returned invalid responses, falling back to rules-based recommendations.")
            return self._get_recommendations_from_rules(metrics)
            
        except Exception as e:
            print(f"An unexpected error occurred in get_recommendations: {str(e)}")
            return self._get_recommendations_from_rules(metrics)

    def _is_valid_recommendation(self, text: str) -> bool:
        """Check if the recommendation text is valid and well-structured"""
        # Check for minimum length (increased to 150 from 100)
        if len(text.strip()) < 150:
            return False
            
        # Check for basic structure (e.g., presence of bullet points, numbers, or section indicators)
        required_elements = ['1.', '2.', '3.', '-', '•', ':']
        if not any(element in text for element in required_elements):
            return False
            
        # Check for metric references to ensure relevance
        metric_terms = ['packet loss', 'latency', 'jitter', 'bandwidth', 'network']
        if not any(term in text.lower() for term in metric_terms):
            return False
            
        # Add a check for repetitive phrases or clearly unhelpful responses (can be refined)
        if "I cannot provide recommendations" in text or "I need more information" in text:
            return False

        return True

    def get_historical_analysis(self, historical_data: List[Dict]) -> Dict:
        """Analyze historical data patterns with caching"""
        try:
            # Calculate averages
            if not historical_data:
                return {
                    'analysis': "No historical data available for analysis.",
                    'average_metrics': {},
                    'data_points': 0,
                    'is_fallback': True
                }

            avg_metrics = {
                'packet_loss': sum(d.get('packet_loss', 0) for d in historical_data) / len(historical_data),
                'latency': sum(d.get('latency', 0) for d in historical_data) / len(historical_data),
                'jitter': sum(d.get('jitter', 0) for d in historical_data) / len(historical_data),
                'bandwidth': sum(d.get('bandwidth', 0) for d in historical_data) / len(historical_data)
            }
            
            # Create cache key
            cache_key = f"historical_{avg_metrics['packet_loss']:.2f}_{avg_metrics['latency']:.2f}_{avg_metrics['jitter']:.2f}_{avg_metrics['bandwidth']:.2f}"
            
            # Check cache first
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                print("Returning cached historical analysis.")
                return cached_result
            
            # Check if we've hit daily limit
            if self.daily_request_count >= self.MAX_DAILY_REQUESTS:
                print("Daily Gemini request limit reached for historical analysis, falling back to rules.")
                return {
                    'analysis': self._get_fallback_recommendations(avg_metrics)['analysis'],
                    'average_metrics': avg_metrics,
                    'data_points': len(historical_data),
                    'is_fallback': True
                }
            
            # Wait for rate limit
            self._wait_for_rate_limit()
            
            # Use the refined _create_prompt for historical analysis
            prompt = self._create_prompt(avg_metrics)
            
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
            print(f"Error getting historical analysis from Gemini: {e}")
            # Fallback to rules-based analysis if API call fails
            return {
                'analysis': self._get_fallback_recommendations(avg_metrics if 'avg_metrics' in locals() else {})['analysis'],
                'average_metrics': avg_metrics if 'avg_metrics' in locals() else {},
                'data_points': len(historical_data),
                'is_fallback': True
            } 