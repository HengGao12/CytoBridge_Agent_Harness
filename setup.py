"""
Setup script for CytoBridge Agent.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cytobridge-agent",
    version="1.0.0",
    author="CytoBridge Team",
    description="LangGraph-powered intelligent agent for automated single-cell dynamics modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JackkWangzh/CytoBridge-agent",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "cytobridge_agent": [
            "codex_sidecar/package.json",
            "codex_sidecar/package-lock.json",
            "codex_sidecar/server.mjs",
            "codex_sidecar/lib/*.mjs",
            "codex_sidecar/tests/*.mjs",
            "utils/model_context_windows.json",
            "resources/algorithm_benchmarks/*.json",
        ],
    },
    python_requires=">=3.9",
    install_requires=[
        "langgraph>=0.2",
        "langgraph-checkpoint-sqlite>=3.0.1",
        "langchain>=0.3",
        "langchain-openai>=0.2",
        "langchain-community>=0.3",
        "httpx>=0.24",
        "beautifulsoup4>=4.12",
        "lxml>=5.0",
        "trafilatura>=1.9",
        "openai>=1.0",
        "pydantic>=2.0,<3",
        "pyyaml>=6",
        "pypdf>=4.0",
        "pdfplumber>=0.11",
        "PyMuPDF>=1.24",
        "rank-bm25>=0.2",
        "sentence-transformers>=3.0",
        "spacy>=3.8",
        "matplotlib>=3.5",
        "scanpy>=1.9",
        "anndata>=0.8",
        "torch>=2.0",
        "numpy<2",
        "scipy",
        "scikit-learn",
        "tqdm",
        "plotly>=5.0",
        "kaleido>=1.2",
        "zstandard>=0.22",
    ],
    extras_require={
        "enrichment": ["gseapy>=1.0"],
        "html": ["markdown>=3.0"],
        "cellrank": ["cellrank>=2.0"],
    },
    entry_points={
        "console_scripts": [
            "cytobridge-agent=cytobridge_agent.cli:main",
            "cellcompass=cytobridge_agent.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
