"""
GitHub repository builder - extracts GitHub links from papers, clones repositories, and creates indexable code entries
"""
import os
import re
import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import git
except ImportError:
    git = None

from .config import DIRS, FILES, ensure_dir
from .rag_tools import extract_pdf_pages

logger = logging.getLogger(__name__)

# Default ignored directories and files
IGNORE_DIRS = {'.git', '__pycache__', 'node_modules', 'build', 'dist', 'venv', 'env'}
IGNORE_EXTENSIONS = {'.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib', '.exe', '.bin', '.class', '.o'}

# File extensions and content types to extract
CODE_EXTENSIONS = {
    '.py': 'python',
    '.ipynb': 'jupyter',
    '.r': 'r',
    '.r': 'r',
    '.rmd': 'rmarkdown',
    '.c': 'c',
    '.cpp': 'cpp',
    '.h': 'header',
    '.hpp': 'cpp-header',
    '.java': 'java',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.jsx': 'react',
    '.tsx': 'react',
    '.go': 'go',
    '.rs': 'rust',
    '.rb': 'ruby',
    '.php': 'php',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.scala': 'scala',
    '.sh': 'shell',
    '.bash': 'shell',
    '.zsh': 'shell',
    '.pl': 'perl',
    '.pm': 'perl',
    '.jl': 'julia',
    '.lua': 'lua',
    '.m': 'matlab',
    '.nb': 'mathematica',
    '.tex': 'latex',
    '.bib': 'bibtex',
    '.md': 'markdown',
    '.rst': 'restructuredtext',
    '.txt': 'text',
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.toml': 'toml',
    '.xml': 'xml',
    '.html': 'html',
    '.css': 'css',
    '.scss': 'scss',
    '.less': 'less',
    '.sql': 'sql',
    '.rds': 'rds',
    '.rda': 'rdata',
    '.rdata': 'rdata',
    '.rds': 'rds',
}

DOC_EXTENSIONS = {'.md', '.rst', '.txt', '.tex', '.bib', '.ipynb'}

def extract_github_links_from_pdf(pdf_path: Path) -> List[str]:
    """
    Extract all GitHub repository links from a PDF file
    """
    github_links = []
    try:
        # Get PDF text
        pages = extract_pdf_pages(pdf_path)
        if not pages:
            return github_links
        
        full_text = "\n".join([text for _, text in pages])
        
        # Regex pattern for GitHub repository URLs
        # Match https://github.com/owner/repo forms, optionally with .git or /tree/
        github_pattern = r'https?://(?:www\.)?github\.com/[\w\-]+/[\w\-]+(?:\.git)?(?:/[\w\-\./]*)?'
        links = re.findall(github_pattern, full_text)
        
        # Deduplicate and normalize by removing trailing slash and .git
        normalized_links = set()
        for link in links:
            # Remove the .git suffix
            if link.endswith('.git'):
                link = link[:-4]
            # Remove trailing slash
            link = link.rstrip('/')
            normalized_links.add(link)
        
        github_links = list(normalized_links)
        
        if github_links:
            logger.info("RAG status message")
        
    except Exception as e:
        logger.error(f"Failed to extract GitHub links from PDF {pdf_path}: {e}")
    
    return github_links

def clone_or_update_repo(repo_url: str, target_dir: Path, update_existing: bool = False) -> bool:
    """
    Clone a GitHub repository locally, or update/skip when it already exists
    
    Args:
        repo_url: Repository URL
        target_dir: Target directory
        update_existing: Whether to update when the repository already exists
    
    Returns:
        Whether it succeeded
    """
    if git is None:
        logger.error("GitPython is not installed; cannot clone repositories. Install with: pip install GitPython")
        return False
    
    # Extract repository name from URL
    repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
    repo_path = target_dir / repo_name
    
    try:
        if repo_path.exists():
            if update_existing:
                logger.info(f"Repository exists; updating: {repo_path}")
                repo = git.Repo(repo_path)
                # Pull latest changes
                repo.remotes.origin.pull()
                return True
            else:
                logger.info(f"Repository exists; skipping: {repo_path}")
                return True  # Treat as success because it already exists
        else:
            logger.info(f"Cloning repository: {repo_url} -> {repo_path}")
            git.Repo.clone_from(repo_url, repo_path)
            return True
    except Exception as e:
        logger.error(f"Failed to clone/update repository {repo_url}: {e}")
        return False

def is_text_file(file_path: Path) -> bool:
    """
    Determine whether a file is text by extension or read probe
    """
    ext = file_path.suffix.lower()
    if ext in CODE_EXTENSIONS:
        return True
    # Read initial bytes to detect binary content
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            # Null bytes indicate likely binary content
            if b'\x00' in chunk:
                return False
        return True
    except:
        return False

def extract_file_content(file_path: Path) -> Optional[str]:
    """
    Read a text file; return None on failure
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return None

def generate_repo_entry(
    repo_url: str,
    repo_path: Path,
    source_pdf: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Create indexable knowledge entries from a cloned repository
    
    Args:
        repo_url: Repository URL
        repo_path: Local repository path
        source_pdf: Source PDF filename, optional
    
    Returns:
        Entry list, one entry per file or code block
    """
    entries = []
    
    # Extract basic repository information
    repo_name = repo_path.name
    try:
        repo = git.Repo(repo_path)
        last_commit = repo.head.commit.hexsha[:8] if repo.head.is_valid() else "unknown"
        last_commit_msg = repo.head.commit.message.strip() if repo.head.is_valid() else ""
        last_commit_date = repo.head.commit.committed_datetime.isoformat() if repo.head.is_valid() else ""
    except:
        last_commit = "unknown"
        last_commit_msg = ""
        last_commit_date = ""
    
    # Iterate repository files
    for root, dirs, files in os.walk(repo_path):
        # Ignore selected directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]
        
        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(repo_path)
            ext = file_path.suffix.lower()
            
            # Ignore selected extensions
            if ext in IGNORE_EXTENSIONS:
                continue
            
            # Check whether the file is text
            if not is_text_file(file_path):
                continue
            
            # Read content
            content = extract_file_content(file_path)
            if content is None:
                continue
            
            # Determine content type
            if ext in CODE_EXTENSIONS:
                lang = CODE_EXTENSIONS.get(ext, 'code')
                if ext in DOC_EXTENSIONS:
                    content_type = 'documentation'
                else:
                    content_type = 'code'
            else:
                # Treat other text files as documentation
                content_type = 'documentation'
                lang = 'text'
            
            # Generate keywords from the filename
            file_stem = file_path.stem
            keywords = [file_stem] + re.findall(r'[a-zA-Z][a-zA-Z0-9_]*', file_stem)
            
            # For code files, extract function names as keywords using a simple regex
            if content_type == 'code':
                # Extract function definitions with simple def/function matching
                func_matches = re.findall(r'def\s+(\w+)\s*\(', content)
                func_matches += re.findall(r'function\s+(\w+)\s*\(', content)
                keywords.extend(func_matches)
            
            # Create entry
            entry = {
                "metadata": {
                    "title": f"GitHub: {repo_name}/{rel_path}",
                    "source": f"github:{repo_url}/blob/main/{rel_path}",
                    "repo_url": repo_url,
                    "repo_name": repo_name,
                    "file_path": str(rel_path),
                    "content_type": content_type,
                    "language": lang,
                    "source_pdf": source_pdf,
                    "last_commit": last_commit,
                    "last_commit_msg": last_commit_msg[:100] if last_commit_msg else "",
                    "last_commit_date": last_commit_date,
                },
                "content": {
                    "full_text": content,
                },
                "analysis": {
                    "summary": f"GitHub repository {repo_name} file: {rel_path}",
                    "key_methods": [],  # Can be populated with extracted function names
                    "key_objects": keywords[:10],
                    "key_technical_contributions": [],
                    "research_domains": ["code", content_type],
                    "innovation_assessment": {
                        "level": "Medium",
                        "reason": "Code from GitHub repository",
                        "score": 0.5
                    }
                }
            }
            
            entries.append(entry)
    
    logger.info(f"Generated entries from repository {repo_name} generated {len(entries)} entries")
    return entries


class GitHubBuilder:
    """
    RAG helper documentation.
    """
    
    def __init__(self, repos_dir: Optional[Path] = None, update_existing: bool = False):
        """
        Args:
            RAG helper documentation.
            update_existing: Whether to update when the repository already exists
        """
        if repos_dir is None:
            repos_dir = DIRS["knowledge_base"] / "github_repos"
        self.repos_dir = Path(repos_dir)
        ensure_dir(self.repos_dir)
        self.update_existing = update_existing
        self.processed_repos = set()  # Used for deduplication
    
    def process_pdfs(self, pdf_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        RAG helper documentation.
        
        Args:
            RAG helper documentation.
        
        Returns:
            List of all generated entries
        """
        all_entries = []
        repo_to_pdfs = {}  # Record which PDFs each repository came from
        
        # Step 1: extract GitHub links from all PDFs
        for pdf_path in pdf_paths:
            pdf_name = pdf_path.name
            links = extract_github_links_from_pdf(pdf_path)
            for link in links:
                if link not in repo_to_pdfs:
                    repo_to_pdfs[link] = []
                repo_to_pdfs[link].append(pdf_name)
        
        if not repo_to_pdfs:
            logger.info("No GitHub links found")
            return []
        
        logger.info(f"Found {len(repo_to_pdfs)} unique GitHub repositories")
        
        # Step 2: clone or update repositories
        for repo_url, sources in repo_to_pdfs.items():
            success = clone_or_update_repo(repo_url, self.repos_dir, self.update_existing)
            if not success:
                logger.warning(f"Skipping repository {repo_url}，clone failed")
                continue
            
            # Get repository name from URL
            repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
            repo_path = self.repos_dir / repo_name
            
            # Generate entries
            entries = generate_repo_entry(repo_url, repo_path, source_pdf=", ".join(sources))
            for entry in entries:
                # Mark source PDFs
                entry["metadata"]["source_pdfs"] = sources
            all_entries.extend(entries)
        
        logger.info("RAG status message")
        return all_entries
    
    def process_all(self) -> List[Dict[str, Any]]:
        """
        Process all PDFs under the literature directory
        """
        literature_dir = DIRS["literature"]
        pdf_files = list(literature_dir.glob("*.pdf"))
        return self.process_pdfs(pdf_files)