"""
Artifact management tools for CytoBridge Agent.

Provides utilities for saving metrics, generating reports, and managing outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

import yaml
from langchain_core.tools import tool

from ..schemas import (
    DataSummary,
    PlanDecision,
    PilotResult,
    DownstreamResult,
    InsightReport,
    FinalArtifacts,
    ClaimAssessment,
    BiologicalInsight,
    TheoryContext,
)

logger = logging.getLogger(__name__)


def write_artifacts(
    outdir: Path,
    data_summary: DataSummary,
    plan_decision: PlanDecision,
    pilot_results: List[PilotResult],
    final_metrics: Dict[str, Any],
    downstream_results: List[DownstreamResult],
    insight_report: Optional[InsightReport],
    theory_context: Optional[TheoryContext],
    h5ad_path: str,
    config_path: str,
) -> FinalArtifacts:
    """
    Write all artifacts to the output directory.
    
    Args:
        outdir: Output directory path.
        data_summary: Summary of input data.
        plan_decision: LLM planning decision.
        pilot_results: Results from pilot training.
        final_metrics: Metrics from final training.
        downstream_results: Results from downstream analyses.
        insight_report: LLM-generated biological insights.
        theory_context: Retrieved theory chunks.
        h5ad_path: Path to trained h5ad.
        config_path: Path to config YAML.
    
    Returns:
        FinalArtifacts with all output paths.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Write metrics JSON
    metrics_path = outdir / "metrics.json"
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "data_summary": data_summary.model_dump(),
        "plan_decision": plan_decision.model_dump(),
        "pilot_results": [r.model_dump() for r in pilot_results],
        "final_training": final_metrics,
        "downstream_results": [r.model_dump() for r in downstream_results],
    }
    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Write report
    report_path = outdir / "report.md"
    report_content = _generate_report_md(
        data_summary=data_summary,
        plan_decision=plan_decision,
        pilot_results=pilot_results,
        final_metrics=final_metrics,
        downstream_results=downstream_results,
        insight_report=insight_report,
        theory_context=theory_context,
        h5ad_path=h5ad_path,
        config_path=config_path,
    )
    with report_path.open('w', encoding='utf-8') as f:
        f.write(report_content)
    
    # Determine figures directory
    figures_dir = outdir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    return FinalArtifacts(
        config_path=config_path,
        h5ad_path=h5ad_path,
        metrics_path=str(metrics_path),
        report_path=str(report_path),
        figures_dir=str(figures_dir),
        downstream_results=downstream_results,
    )


def _generate_report_md(
    data_summary: DataSummary,
    plan_decision: PlanDecision,
    pilot_results: List[PilotResult],
    final_metrics: Dict[str, Any],
    downstream_results: List[DownstreamResult],
    insight_report: Optional[InsightReport],
    theory_context: Optional[TheoryContext],
    h5ad_path: str,
    config_path: str,
) -> str:
    """Generate the markdown report."""
    sections = []
    
    # Header
    sections.append("# CytoBridge Agent Analysis Report")
    sections.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Data Summary
    sections.append("## 1. Data Summary\n")
    sections.append(f"- **Cells**: {data_summary.n_obs:,}")
    sections.append(f"- **Genes**: {data_summary.n_vars:,}")
    if data_summary.sparsity is not None:
        sections.append(f"- **Sparsity**: {data_summary.sparsity:.1%}")
    if data_summary.time_candidates:
        tc = data_summary.time_candidates[0]
        sections.append(f"- **Time Key**: `{tc.key}` ({len(tc.levels)} time points)")
    if data_summary.label_candidates:
        sections.append(f"- **Label Candidates**: {', '.join(data_summary.label_candidates)}")
    sections.append("")
    
    # Model Selection
    sections.append("## 2. Model Selection\n")
    sections.append(f"**Chosen Model Family**: `{plan_decision.model_family.value}`\n")
    sections.append(f"**Reasoning**: {plan_decision.reasoning}\n")
    if plan_decision.theory_support:
        sections.append(f"**Theory Support**: {plan_decision.theory_support[:500]}...\n")
    sections.append("")
    
    # Pilot Results
    sections.append("## 3. Pilot Training Results\n")
    if pilot_results:
        sections.append("| Candidate | Success | W1 (mean) | TMV (mean) | Runtime |")
        sections.append("|-----------|---------|-----------|------------|---------|")
        for pr in pilot_results:
            w1_str = f"{sum(pr.w1_scores)/len(pr.w1_scores):.4f}" if pr.w1_scores else "N/A"
            tmv_str = f"{sum(pr.tmv_scores)/len(pr.tmv_scores):.4f}" if pr.tmv_scores else "N/A"
            runtime_str = f"{pr.runtime_sec:.1f}s" if pr.runtime_sec else "N/A"
            status = "✓" if pr.success else f"✗ ({pr.error_type or 'error'})"
            sections.append(f"| {pr.candidate_name} | {status} | {w1_str} | {tmv_str} | {runtime_str} |")
    else:
        sections.append("*No pilot runs recorded.*")
    sections.append("")
    
    # Final Training
    sections.append("## 4. Final Training\n")
    if final_metrics.get("error"):
        sections.append(f"**Error**: {final_metrics['error']}\n")
    else:
        if final_metrics.get("w1_scores"):
            w1_mean = sum(final_metrics["w1_scores"]) / len(final_metrics["w1_scores"])
            sections.append(f"- **W1 Score (mean)**: {w1_mean:.4f}")
        if final_metrics.get("tmv_scores"):
            tmv_mean = sum(final_metrics["tmv_scores"]) / len(final_metrics["tmv_scores"])
            sections.append(f"- **TMV Score (mean)**: {tmv_mean:.4f}")
        if final_metrics.get("runtime_sec"):
            sections.append(f"- **Runtime**: {final_metrics['runtime_sec']:.1f}s")
    sections.append(f"\n**Config**: `{Path(config_path).name}`")
    sections.append(f"**Model**: `{Path(h5ad_path).name}`\n")
    
    # Downstream Analyses
    sections.append("## 5. Downstream Analyses\n")
    for dr in downstream_results:
        status = "✓" if dr.success else "✗"
        sections.append(f"### {dr.analysis_type.value} {status}\n")
        sections.append(f"{dr.summary}\n")
        if dr.artifacts:
            sections.append("**Artifacts**:")
            for name, path in dr.artifacts.items():
                # Display filename instead of full absolute path for better portability and to avoid OS-specific path issues
                try:
                    display_path = Path(path).name
                except Exception:
                    display_path = str(path)
                sections.append(f"- `{name}`: `{display_path}`")
        if dr.error:
            sections.append(f"**Error**: {dr.error}")
        sections.append("")
    
    # Biological Insights
    if insight_report:
        sections.append("## 6. Biological Insights\n")
        
        for insight in insight_report.insights:
            sections.append(f"### {insight.title}\n")
            sections.append(f"{insight.description}\n")
            sections.append(f"**Confidence**: {insight.confidence}")
            if insight.theory_support:
                sections.append(f"\n**Theory Support**: {insight.theory_support[:300]}...")
            sections.append("")
        
        # Claim Assessments
        if insight_report.claim_assessments:
            sections.append("### Claim Support Assessment\n")
            for ca in insight_report.claim_assessments:
                icon = {"supported": "✓", "partially_supported": "◐", "not_supported": "✗"}
                sections.append(f"**{ca.claim}**: {icon.get(ca.support_level.value, '?')} {ca.support_level.value}")
                sections.append(f"\n{ca.reasoning}")
                if ca.limitations:
                    sections.append(f"\n*Limitations*: {ca.limitations}")
                sections.append("")
        
        # Overall Limitations
        if insight_report.overall_limitations:
            sections.append("### Overall Limitations\n")
            sections.append(insight_report.overall_limitations)
            sections.append("")
    
    # Theory References
    if theory_context and theory_context.chunks:
        sections.append("## 7. Theory References\n")
        sections.append("Relevant passages from the theoretical foundation:\n")
        for chunk in theory_context.chunks[:5]:
            sections.append(f"**[{chunk.chunk_id}]** (p.{chunk.page}, score={chunk.relevance_score:.2f})")
            sections.append(f"> {chunk.text[:300]}...")
            sections.append("")
    
    return "\n".join(sections)


def generate_html_report(markdown_content: str, outdir: Path) -> str:
    """
    Convert markdown report to HTML.
    
    Args:
        markdown_content: Markdown report content.
        outdir: Output directory.
    
    Returns:
        Path to HTML file.
    """
    try:
        import markdown
        
        html = markdown.markdown(
            markdown_content,
            extensions=['tables', 'fenced_code', 'toc']
        )
        
        # Wrap in basic HTML template
        html_full = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>CytoBridge Agent Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f5f5f5; padding: 15px; overflow-x: auto; border-radius: 5px; }}
        blockquote {{ border-left: 3px solid #ccc; margin: 0; padding-left: 15px; color: #666; }}
        h1, h2, h3 {{ margin-top: 30px; }}
    </style>
</head>
<body>
{html}
</body>
</html>"""
        
        html_path = outdir / "report.html"
        with html_path.open('w', encoding='utf-8') as f:
            f.write(html_full)
        
        return str(html_path)
        
    except ImportError:
        logger.warning("markdown package not available for HTML generation.")
        return ""


# LangChain tool wrapper

@tool
def WriteArtifactsTool(
    output_dir: str,
    data_summary_json: str,
    plan_decision_json: str,
    pilot_results_json: str,
    final_metrics_json: str,
    downstream_results_json: str,
    h5ad_path: str,
    config_path: str,
    insight_report_json: Optional[str] = None,
    theory_context_json: Optional[str] = None,
    generate_html: bool = False,
) -> dict:
    """
    Write all analysis artifacts to the output directory.
    
    Args:
        output_dir: Directory for outputs.
        data_summary_json: JSON of DataSummary.
        plan_decision_json: JSON of PlanDecision.
        pilot_results_json: JSON array of PilotResult.
        final_metrics_json: JSON of final training metrics.
        downstream_results_json: JSON array of DownstreamResult.
        h5ad_path: Path to trained model h5ad.
        config_path: Path to config YAML.
        insight_report_json: Optional JSON of InsightReport.
        theory_context_json: Optional JSON of TheoryContext.
        generate_html: Whether to also generate HTML report.
    
    Returns:
        Dictionary with paths to all artifacts.
    """
    outdir = Path(output_dir)
    
    # Parse JSON inputs
    data_summary = DataSummary(**json.loads(data_summary_json))
    plan_decision = PlanDecision(**json.loads(plan_decision_json))
    pilot_results = [PilotResult(**r) for r in json.loads(pilot_results_json)]
    final_metrics = json.loads(final_metrics_json)
    downstream_results = [DownstreamResult(**r) for r in json.loads(downstream_results_json)]
    
    insight_report = None
    if insight_report_json:
        insight_report = InsightReport(**json.loads(insight_report_json))
    
    theory_context = None
    if theory_context_json:
        theory_context = TheoryContext(**json.loads(theory_context_json))
    
    artifacts = write_artifacts(
        outdir=outdir,
        data_summary=data_summary,
        plan_decision=plan_decision,
        pilot_results=pilot_results,
        final_metrics=final_metrics,
        downstream_results=downstream_results,
        insight_report=insight_report,
        theory_context=theory_context,
        h5ad_path=h5ad_path,
        config_path=config_path,
    )
    
    result = artifacts.model_dump()
    
    # Generate HTML if requested
    if generate_html:
        md_path = Path(artifacts.report_path)
        md_content = md_path.read_text(encoding='utf-8')
        html_path = generate_html_report(md_content, outdir)
        if html_path:
            result["html_report_path"] = html_path
    
    return result

