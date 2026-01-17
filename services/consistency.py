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
    Checks consistency of BOQ extractions from outputs.
    
    Example:
        checker = ConsistencyChecker()
        result = checker.check_from_outputs(outputs)
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
    
    def check_from_outputs(self, outputs: List[str]) -> Dict[str, Any]:
        """
        Compute consistency metrics from a list of BOQ outputs.
        
        Args:
            outputs: List of BOQ output strings.
        
        Returns:
            Dictionary with consistency metrics.
        """
        logger.info(f'Computing consistency from {len(outputs)} outputs')
        
        # Calculate similarity
        avg_similarity = self._calculate_similarity(outputs)
        consistency_score = avg_similarity * 100
        
        # Extract and average confidence scores
        all_confidences = []
        for boq in outputs:
            confidences = self._extract_confidence_scores(boq)
            all_confidences.extend(confidences)
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        successful_runs = len([r for r in outputs if r])
        
        result = {
            "consistency_score": round(consistency_score, 2),
            "runs": len(outputs),
            "successful_runs": successful_runs,
            "avg_similarity": round(avg_similarity, 2),
            "avg_confidence": round(avg_confidence, 2),
            "total_confidence_scores": len(all_confidences)
        }
        
        logger.info(f'Consistency check from outputs completed: {result}')
        return result
