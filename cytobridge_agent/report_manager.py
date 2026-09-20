"""
Report Manager for CytoBridge Agent.

Handles report generation, storage, and incremental updates.
Supports section-level editing for multi-turn conversations.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Section IDs and their display order
REPORT_SECTIONS = [
    "header",
    "summary", 
    "preprocessing",
    "model",
    "training",
    "trajectory",
    "drivers",
    "growth",
    "conclusion",
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CytoBridge Analysis Report</title>
    <style>
        :root {{
            --primary: #4f46e5;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        .section {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: var(--primary);
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border);
        }}
        .section-id {{
            font-size: 0.7rem;
            color: var(--text-muted);
            float: right;
            font-family: monospace;
        }}
        .figure-container {{
            text-align: center;
            margin: 1rem 0;
        }}
        .figure-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .figure-caption {{
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
            font-style: italic;
        }}
        .conclusion-box {{
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-left: 4px solid var(--primary);
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        .metric-card {{
            background: var(--bg);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--primary);
        }}
        .metric-label {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}
        ul, ol {{ margin-left: 1.5rem; margin-top: 0.5rem; }}
        li {{ margin-bottom: 0.25rem; }}
        p {{ margin-bottom: 0.75rem; }}
        code {{
            background: var(--bg);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
<div class="container">
{sections}
</div>
</body>
</html>
"""


SECTION_TEMPLATE = """
<section class="section" id="section-{section_id}">
    <h2>{title} <span class="section-id">[{section_id}]</span></h2>
    <div class="section-content">
{content}
    </div>
</section>
"""


class ReportManager:
    """
    Manages report state and enables incremental updates.
    
    The report is stored as a dictionary of sections,
    which can be individually modified.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report state: section_id -> content (HTML string)
        self.sections: Dict[str, str] = {}
        
        # Metadata for tracking
        self.created_at: Optional[str] = None
        self.last_updated: Optional[str] = None
        self.figures: List[Dict[str, str]] = []
        
    def initialize(
        self,
        dataset_name: str,
        n_cells: int,
        n_genes: int,
        user_question: str = "",
    ) -> None:
        """Initialize report with header section."""
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.last_updated = self.created_at
        
        header_content = f"""
    <h1 style="font-size: 1.75rem; margin-bottom: 1rem;">🧬 CytoBridge Analysis Report</h1>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{n_cells:,}</div>
            <div class="metric-label">Cells</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{n_genes:,}</div>
            <div class="metric-label">Genes</div>
        </div>
    </div>
    <p><strong>Dataset:</strong> <code>{dataset_name}</code></p>
    <p><strong>Generated:</strong> {self.created_at}</p>
    <p><strong>Question:</strong> {user_question or 'General analysis'}</p>
"""
        self.sections["header"] = header_content
        
    def set_section(
        self,
        section_id: str,
        content: str,
        title: Optional[str] = None,
    ) -> None:
        """Set or update a section's content."""
        if section_id not in REPORT_SECTIONS and not section_id.startswith("custom_"):
            logger.warning(f"Unknown section ID: {section_id}. Adding as custom section.")
        
        self.sections[section_id] = content
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")
        
    def add_figure(
        self,
        section_id: str,
        figure_path: str,
        caption: str,
        conclusion: str = "",
    ) -> None:
        """Add a figure to a section."""
        # Track figure
        self.figures.append({
            "section": section_id,
            "path": figure_path,
            "caption": caption,
            "conclusion": conclusion,
        })
        
        # Generate figure HTML
        figure_html = f"""
    <div class="figure-container">
        <img src="{figure_path}" alt="{caption}">
        <p class="figure-caption">{caption}</p>
    </div>
"""
        if conclusion:
            figure_html += f"""
    <div class="conclusion-box">
        <strong>Key Finding:</strong> {conclusion}
    </div>
"""
        
        # Append to section
        current = self.sections.get(section_id, "")
        self.sections[section_id] = current + figure_html
        
    def update_section_content(
        self,
        section_id: str,
        new_content: str,
        append: bool = False,
    ) -> str:
        """Update a section's content. Returns status message."""
        if section_id not in self.sections:
            return f"❌ Section '{section_id}' does not exist. Available: {list(self.sections.keys())}"
        
        if append:
            self.sections[section_id] += f"\n{new_content}"
        else:
            self.sections[section_id] = new_content
            
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"✅ Section '{section_id}' updated successfully."
        
    def generate_html(self) -> str:
        """Generate full HTML report from sections."""
        sections_html = ""
        
        # Section titles
        titles = {
            "header": "Report Overview",
            "summary": "Executive Summary",
            "preprocessing": "Data Preprocessing",
            "model": "Model Selection",
            "training": "Training Results",
            "trajectory": "Trajectory & Fate Analysis",
            "drivers": "Driver Gene Analysis",
            "growth": "Growth & Mass Analysis",
            "conclusion": "Conclusions",
        }
        
        # Build sections in order
        for section_id in REPORT_SECTIONS:
            if section_id in self.sections and self.sections[section_id].strip():
                title = titles.get(section_id, section_id.replace("_", " ").title())
                sections_html += SECTION_TEMPLATE.format(
                    section_id=section_id,
                    title=title,
                    content=self.sections[section_id],
                )
        
        # Add any custom sections
        for section_id, content in self.sections.items():
            if section_id not in REPORT_SECTIONS and content.strip():
                title = section_id.replace("_", " ").title()
                sections_html += SECTION_TEMPLATE.format(
                    section_id=section_id,
                    title=title,
                    content=content,
                )
        
        return HTML_TEMPLATE.format(sections=sections_html)
    
    def save(self, filename: str = "report.html") -> Path:
        """Save report to file."""
        html = self.generate_html()
        report_path = self.output_dir / filename
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
            
        logger.info(f"Report saved to {report_path}")
        return report_path
    
    def get_section_list(self) -> str:
        """Get list of current sections for LLM reference."""
        lines = ["Available report sections:"]
        for section_id in REPORT_SECTIONS:
            status = "✅" if section_id in self.sections else "⬜"
            lines.append(f"  {status} {section_id}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for state persistence."""
        return {
            "sections": self.sections,
            "figures": self.figures,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], output_dir: Path) -> "ReportManager":
        """Deserialize from dictionary."""
        manager = cls(output_dir)
        manager.sections = data.get("sections", {})
        manager.figures = data.get("figures", [])
        manager.created_at = data.get("created_at")
        manager.last_updated = data.get("last_updated")
        return manager
