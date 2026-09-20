"""
RAG helper documentation.
"""
import os
import json
import time
import re
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import logging

from .config import DIRS, FILES, KB_CONFIG, ensure_dir, call_llm
from .rag_tools import extract_pdf_text, load_apa_citations

logger = logging.getLogger(__name__)

# RAG helper comment.
class KnowledgeBaseBuilder:
    """RAG helper."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.apa_citations = {}
        self.knowledge_base = []
        
    def build(self, force_rebuild: bool = False, verbose: bool = True) -> Dict[str, Any]:
        """
        RAG helper documentation.
        
        Args:
            RAG helper documentation.
            RAG helper documentation.
            
        Returns:
            RAG helper documentation.
        """
        if verbose:
            print("\n" + "="*80)
            print("📚 LITERATURE KNOWLEDGE BASE BUILDER")
            print("="*80)
        
        # RAG helper comment.
        if not self.llm_client:
            error_msg = "❌ LLM client was not provided"
            if verbose:
                print(error_msg)
            return {"status": "error", "message": error_msg}
        
        # RAG helper comment.
        literature_dir = DIRS["literature"]
        if not literature_dir.exists():
            error_msg = "RAG error"
            if verbose:
                print(error_msg)
            return {"status": "error", "message": error_msg}
        
        # RAG helper comment.
        pdf_files = self._get_pdf_files(literature_dir)
        if verbose:
            print("RAG status message")
            print("RAG status message")
        
        # RAG helper comment.
        self.apa_citations = load_apa_citations()
        if verbose:
            print("RAG status message")
        
        # RAG helper comment.
        existing_entries = self._load_existing_entries()
        
        # RAG helper comment.
        unlearned_pdfs = self._find_unlearned_pdfs(pdf_files, existing_entries)
        
        # RAG helper comment.
        if verbose:
            print("RAG status message")
            print("RAG status message")
            print("RAG status message")
            print("RAG status message")
        
        result = {
            "status": "checked",
            "total_pdfs": len(pdf_files),
            "learned": len(existing_entries),
            "unlearned": len(unlearned_pdfs),
            "needs_update": len(unlearned_pdfs) > 0,
            "kb_file": str(FILES["structured_knowledge_base"])
        }
        
        # RAG helper comment.
        if unlearned_pdfs:
            if verbose:
                print("RAG status message")
            learned_count = self._learn_missing_pdfs(unlearned_pdfs, existing_entries, verbose)
            result["learned_new"] = learned_count
        else:
            if verbose:
                print("RAG status message")
            result["learned_new"] = 0
        
        if verbose:
            print("="*80 + "\n")
        return result
    
    def _get_pdf_files(self, directory: Path) -> List[Path]:
        """RAG helper."""
        pdf_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(Path(os.path.join(root, file)))
        return pdf_files
    
    def _load_existing_entries(self) -> Dict[str, Any]:
        """RAG helper."""
        kb_file = FILES["structured_knowledge_base"]
        existing_entries = {}
        
        if kb_file.exists():
            try:
                with open(kb_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    entries = data.get("knowledge_base", [])
                    for entry in entries:
                        pdf_filename = entry["metadata"]["pdf_filename"]
                        existing_entries[pdf_filename] = entry
                logger.info("RAG status message")
                logger.info("RAG status message")
            except Exception as e:
                logger.warning("RAG status message")
        else:
            logger.info("RAG status message")
        
        return existing_entries
    
    def _find_unlearned_pdfs(self, pdf_files: List[Path], existing_entries: Dict) -> List[Path]:
        """RAG helper."""
        unlearned = []
        for pdf_path in pdf_files:
            pdf_filename = pdf_path.name
            if pdf_filename not in existing_entries:
                unlearned.append(pdf_path)
        return unlearned
    
    def _learn_missing_pdfs(self, pdf_paths: List[Path], existing_entries: Dict, verbose: bool = True) -> int:
        """RAG helper."""
        learned_count = 0
        failed_count = 0
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            filename = pdf_path.name
            if verbose:
                print("RAG status message")
            
            try:
                # RAG helper comment.
                processor = LiteratureProcessor(pdf_path, self.llm_client)
                
                # RAG helper comment.
                entry = processor.process(self.apa_citations)
                
                if entry:
                    # RAG helper comment.
                    existing_entries[filename] = entry
                    learned_count += 1
                    if verbose:
                        print("RAG status message")
                else:
                    failed_count += 1
                
                # RAG helper comment.
                time.sleep(KB_CONFIG["request_delay"])
                
            except Exception as e:
                if verbose:
                    print("RAG status message")
                failed_count += 1
        
        # RAG helper comment.
        if learned_count > 0:
            knowledge_base = list(existing_entries.values())
            self._save_knowledge_base(knowledge_base)
            if verbose:
                print("RAG status message")
        
        return learned_count
    
    def _save_knowledge_base(self, knowledge_base: List[Dict[str, Any]]) -> bool:
        """RAG helper."""
        try:
            kb_file = FILES["structured_knowledge_base"]
            ensure_dir(DIRS["knowledge_base"])
            
            # RAG helper comment.
            with_apa = sum(1 for e in knowledge_base if e["metadata"].get("apa_citation"))
            
            with open(kb_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "total_documents": len(knowledge_base),
                        "processed_date": datetime.now().isoformat(),
                        "description": "Structured knowledge base",
                        "apa_citations_count": with_apa
                    },
                    "knowledge_base": knowledge_base
                }, f, ensure_ascii=False, indent=2)
            
            logger.info("RAG status message")
            return True
            
        except Exception as e:
            logger.error("RAG status message")
            return False


# RAG helper comment.
class LiteratureProcessor:
    """RAG helper."""
    
    def __init__(self, pdf_path: Path, llm_client=None):
        self.pdf_path = Path(pdf_path)
        self.llm_client = llm_client
        self.filename = self.pdf_path.name
        self.title = os.path.splitext(self.filename)[0]
        
    def process(self, apa_citations: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
        """
        RAG helper documentation.
        
        Args:
            RAG helper documentation.
            
        Returns:
            RAG helper documentation.
        """
        try:
            logger.info("RAG status message")
            
            # RAG helper comment.
            full_text = extract_pdf_text(str(self.pdf_path))
            if not full_text or len(full_text) < KB_CONFIG["min_text_length"]:
                logger.warning("RAG status message")
                return None
            
            # RAG helper comment.
            logger.info("RAG status message")
            analysis = self._analyze_literature(full_text)
            
            # RAG helper comment.
            entry = self._create_entry(
                full_text=full_text,
                analysis=analysis,
                apa_citations=apa_citations
            )
            
            logger.info("RAG status message")
            return entry
            
        except Exception as e:
            logger.error(f"Failed to process PDF {self.pdf_path}: {str(e)}")
            return None
    
    def _analyze_literature(self, full_text: str) -> Dict[str, Any]:
        """Analyze paper content with the LLM."""
        required_fields = [
            "key_objects", "key_methods", "key_technical_contributions",
            "key_scientific_conclusions", "research_domains", "innovation_assessment",
            "potential_applications", "method_strengths", "method_limitations", "summary"
        ]
        required_innovation_fields = ["level", "reason", "score"]

        def _extract_and_validate(response_text: str) -> Dict[str, Any]:
            json_match = re.search(r'\{.*\}', response_text or "", re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON object found in LLM response")

            json_str = json_match.group(0)
            print(f"JSON extracted successfully, length: {len(json_str)} characters")

            analysis = json.loads(json_str)
            missing_fields = [f for f in required_fields if f not in analysis]
            if missing_fields:
                raise ValueError(f"JSON is missing required fields: {missing_fields}")

            innovation = analysis.get("innovation_assessment", {})
            missing_innovation = [f for f in required_innovation_fields if f not in innovation]
            if missing_innovation:
                raise ValueError(f"innovation_assessment is missing fields: {missing_innovation}")

            return analysis

        system_prompt = """You are a scientific literature analysis expert. Read the paper content carefully and return the analysis strictly in the following JSON format:

{
    "key_objects": ["main objects, entities, or concepts studied in the paper"],
    "key_methods": ["main methods, algorithms, techniques, or frameworks used"],
    "key_technical_contributions": ["new techniques, improvements, or innovations proposed"],
    "key_scientific_conclusions": ["main findings, conclusions, or significance"],
    "research_domains": ["research domains involved, such as 'bioinformatics' or 'single-cell analysis'"],
    "innovation_assessment": {
        "level": "High/Medium/Low",
        "reason": "assessment rationale",
        "score": "numeric value between 0.0 and 1.0"
    },
    "potential_applications": ["potential applications of the technique"],
    "method_strengths": ["method strengths"],
    "method_limitations": ["method limitations"],
    "summary": "concise summary of the paper within 200 words"
}

Important requirements:
1. Return valid JSON.
2. All fields must be present.
3. List fields may be empty arrays but must not be null.
4. Innovation score ranges: High (0.8-1.0), Medium (0.5-0.7), Low (0.0-0.4).
5. Fill all content in English.
6. Return only the JSON object and no extra text.
"""

        prompt = f"""Analyze the following paper and return the result strictly as JSON:

Paper title: {self.title}

Paper content, first 3000 characters:
{full_text[:3000]}

Return the complete JSON analysis:"""

        print(f"\nCalling LLM to analyze paper: {self.title}")
        
        try:
            response_text = call_llm(
                llm_client=self.llm_client,
                user_prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2
            )
            
            print(f"LLM response status: {'success' if response_text else 'failure'}")
            
            if not response_text:
                print("LLM returned an empty response")
                raise ValueError("LLM call failed with an empty response")
                
            print(f"   Response length: {len(response_text)} characters")
            print(f"   Response preview: {response_text[:200]}...")
            
        except Exception as e:
            print(f"LLM call error: {str(e)}")
            raise
        
        try:
            analysis = _extract_and_validate(response_text)
        except Exception as first_error:
            print(f"Structured literature analysis parse failed; retrying once: {first_error}")
            print(f"   Raw response: {(response_text or '')[:500]}")
            repair_prompt = f"""The previous literature-analysis output violated the structured JSON contract. Error:
{first_error}

Invalid output excerpt:
{(response_text or '')[:4000]}

Return the complete JSON again. Return only a JSON object, with no Markdown or explanation.
The JSON must include these top-level fields:
{json.dumps(required_fields, ensure_ascii=False)}
`innovation_assessment` must include these fields:
{json.dumps(required_innovation_fields, ensure_ascii=False)}

Paper title: {self.title}
"""
            repair_response = call_llm(
                llm_client=self.llm_client,
                user_prompt=repair_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
            )
            analysis = _extract_and_validate(repair_response)
        
        # Add the title field.
        analysis["title"] = self.title
        
        print(f"Successfully analyzed paper: {self.title}")
        print(f"   Methods: {analysis.get('key_methods', [])[:3]}")
        print(f"   Innovation score: {analysis.get('innovation_assessment', {}).get('score', 'N/A')}")
        
        return analysis
    
    def _create_entry(self, full_text: str, analysis: Dict[str, Any], apa_citations: Dict[str, str] = None) -> Dict[str, Any]:
        """Create a structured knowledge entry."""
        # RAG helper comment.
        apa_citation = ""
        if apa_citations:
            apa_citation = apa_citations.get(self.filename, "")
            if apa_citation:
                logger.info("RAG status message")
            else:
                logger.warning("RAG status message")
        
        # RAG helper comment.
        abstract = analysis.get("summary", "")
        
        entry = {
            "metadata": {
                "title": self.title,
                "pdf_filename": self.filename,
                "apa_citation": apa_citation
            },
            "content": {
                "abstract": abstract,
                "full_text": full_text[:5000] if full_text else ""
            },
            "analysis": analysis,
            "search_metadata": {
                "keywords": analysis.get("key_objects", []) + analysis.get("key_methods", []),
                "domains": analysis.get("research_domains", []),
                "innovation_score": analysis.get("innovation_assessment", {}).get("score", 0.5)
            }
        }
        
        return entry
