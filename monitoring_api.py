# Grafana Iframe Embedding for Frontend Integration
# No API keys needed - direct iframe embedding in containers

import os


class GrafanaEmbedder:
    """Simple class to generate Grafana iframe embed codes"""

    def __init__(self):
        self.grafana_url = os.getenv('GRAFANA_URL', 'http://grafana:3000')

    def get_dashboard_iframe(self, dashboard_uid: str, dashboard_name: str, height: int = 600, width: str = "100%") -> str:
        """Generate iframe HTML for dashboard embedding
        
        Args:
            dashboard_uid: Grafana dashboard UID
            dashboard_name: Dashboard display name
            height: Iframe height in pixels (default: 600)
            width: Iframe width in CSS units (default: "100%")
        """
        iframe_src = f"{self.grafana_url}/d/{dashboard_uid}/{dashboard_name}?orgId=1&kiosk&refresh=30s&theme=light"

        return f'''<iframe
  src="{iframe_src}"
  width="{width}"
  height="{height}"
  frameborder="0"
  style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"
  allowfullscreen
></iframe>'''


# ============================================
# USAGE FOR FRONTEND TEAM
# ============================================

# In your React/Vue/HTML component:

"""
from monitoring_api import GrafanaEmbedder

embedder = GrafanaEmbedder()

# Full width dashboard
dashboard_html = embedder.get_dashboard_iframe('abc123', 'boq-monitoring')

# Custom width dashboard
dashboard_fixed = embedder.get_dashboard_iframe('abc123', 'boq-monitoring', height=800, width="600px")

# Render in component
<div dangerouslySetInnerHTML={{__html: dashboard_html}} />
"""

# ============================================
# DIRECT USAGE
# ============================================

if __name__ == "__main__":
    embedder = GrafanaEmbedder()
    dashboard_uid = "your-dashboard-uid-here"
    dashboard_name = "boq-monitoring"
    
    print(embedder.get_dashboard_iframe(dashboard_uid, dashboard_name))