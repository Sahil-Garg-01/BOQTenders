"""
Consistency checking service for evaluating BOQ extraction reliability.
"""
from typing import List, Dict, Any
from difflib import SequenceMatcher
from loguru import logger
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config.settings import settings
from services.boq_extractor import BOQExtractor


class ConsistencyChecker:
    """
    Checks consistency of BOQ extractions across multiple runs.
    
    Example:
        checker = ConsistencyChecker()
        result = checker.check(chunks, vector_store, runs=4)
        print(f"Consistency: {result['consistency_score']}%")
    """
    
    def __init__(self, boq_extractor: BOQExtractor, default_runs: int = None, low_threshold: float = None):
        """
        Initialize consistency checker.
        
        Args:
            boq_extractor: BOQ extractor instance. Required.
            default_runs: Default number of extraction runs. Defaults to config value.
            low_threshold: Threshold for low consistency warning. Defaults to config value.
        """
        self.boq_extractor = boq_extractor
        self.default_runs = default_runs or settings.consistency.default_runs
        self.low_threshold = low_threshold or settings.consistency.low_consistency_threshold
    
    def _calculate_similarity(self, results: List[str]) -> float:
        """
        Calculate average pairwise similarity between results.
        
        Args:
            results: List of BOQ extraction results.
        
        Returns:
            Average similarity score (0.0 to 1.0).
        """
        similarities = []
        
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if results[i] and results[j]:
                    sim = SequenceMatcher(None, results[i], results[j]).ratio()
                    similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0
    
    def _extract_confidence_scores(self, boq: str) -> List[float]:
        """
        Extract confidence scores from BOQ output.
        
        Args:
            boq: Formatted BOQ output string.
        
        Returns:
            List of confidence score values.
        """
        if not boq:
            return []
        
        lines = boq.split('\n')
        confidence_idx = None
        confidences = []
        
        # Find confidence column index from header
        for line in lines:
            line = line.strip()
            if '|' in line and 'Confidence' in line and not line.startswith('| ---'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                confidence_idx = next(
                    (i for i, p in enumerate(parts) if 'Confidence' in p),
                    None
                )
                if confidence_idx is not None:
                    break
        
        if confidence_idx is None:
            return []
        
        # Extract confidence values from data rows
        for line in lines:
            if '|' in line and not line.startswith('| ---') and 'Confidence' not in line:
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) > confidence_idx:
                    try:
                        conf_str = parts[confidence_idx]
                        if conf_str and conf_str != 'NA' and conf_str != 'N/A':
                            conf_str = conf_str.rstrip('%')
                            conf = float(conf_str)
                            confidences.append(conf)
                    except (ValueError, IndexError):
                        pass
        
        return confidences
    
    def check(self, chunks: List[Document], vector_store: FAISS, runs: int = None) -> Dict[str, Any]:
        """
        Run multiple BOQ extractions and compute consistency metrics.
        
        Args:
            chunks: Document chunks to extract from.
            vector_store: Vector store (passed to extractor).
            runs: Number of extraction runs. Defaults to config value.
        
        Returns:
            Dictionary with consistency metrics:
                - consistency_score: Overall consistency percentage
                - runs: Number of runs attempted
                - successful_runs: Number of successful runs
                - avg_similarity: Average pairwise similarity
                - avg_confidence: Average confidence score
                - total_confidence_scores: Number of confidence scores found
                - is_low_consistency: Whether consistency is below threshold
        """
        runs = runs or self.default_runs
        logger.info(f'Starting consistency check with {runs} runs')
        
        results = []
        for run_num in range(runs):
            try:
                logger.info(f'Consistency run {run_num + 1}/{runs}')
                boq = self.boq_extractor.extract(chunks, vector_store)
                results.append(boq)
            except Exception as e:
                logger.warning(f"Consistency run {run_num + 1} failed: {e}")
                results.append("")
        
        # Calculate similarity
        avg_similarity = self._calculate_similarity(results)
        consistency_score = avg_similarity * 100
        
        # Extract and average confidence scores
        all_confidences = []
        for boq in results:
            confidences = self._extract_confidence_scores(boq)
            all_confidences.extend(confidences)
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        successful_runs = len([r for r in results if r])
        
        result = {
            "consistency_score": round(consistency_score, 2),
            "runs": runs,
            "successful_runs": successful_runs,
            "avg_similarity": round(avg_similarity, 2),
            "avg_confidence": round(avg_confidence, 2),
            "total_confidence_scores": len(all_confidences),
            "is_low_consistency": consistency_score < self.low_threshold
        }
        
        logger.info(f'Consistency check completed: {result}')
        return result
