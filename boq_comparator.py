from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import re

def parse_boq_table(boq_text: str) -> List[Dict[str, str]]:
    """
    Parse a markdown BOQ table into a list of dictionaries.
    Assumes table format: | Item No/Code | Description | Quantity | Unit | Rate | Amount |
    """
    lines = boq_text.strip().split('\n')
    if len(lines) < 3:
        return []
    
    # Find table start (after header)
    table_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('|') and '---' in line:
            table_start = i + 1
            break
    if table_start == -1 or table_start >= len(lines):
        return []
    
    items = []
    for line in lines[table_start:]:
        line = line.strip()
        if not line or not line.startswith('|'):
            continue
        parts = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove outer |
        if len(parts) >= 6:
            item = {
                'code': parts[0],
                'desc': parts[1],
                'qty': parts[2],
                'unit': parts[3],
                'rate': parts[4],
                'amt': parts[5]
            }
            items.append(item)
    return items

def match_items(extracted: List[Dict[str, str]], original: List[Dict[str, str]], threshold: float = 0.8) -> List[Tuple[Dict[str, str], Dict[str, str], float]]:
    """
    Match extracted items to original items based on description similarity.
    Returns list of (extracted_item, original_item, similarity_score) for matches above threshold.
    """
    matches = []
    used_original = set()
    
    for ext_item in extracted:
        ext_desc = ext_item['desc'].lower().strip()
        best_match = None
        best_score = 0.0
        
        for idx, orig_item in enumerate(original):
            if idx in used_original:
                continue
            orig_desc = orig_item['desc'].lower().strip()
            score = SequenceMatcher(None, ext_desc, orig_desc).ratio()
            if score > best_score and score >= threshold:
                best_match = orig_item
                best_score = score
        
        if best_match:
            matches.append((ext_item, best_match, best_score))
            used_original.add(original.index(best_match))
    
    return matches

def compare_fields(item1: Dict[str, str], item2: Dict[str, str]) -> Dict[str, bool]:
    """
    Compare fields between two items.
    Returns dict with match flags for each field.
    """
    def normalize_num(s: str) -> float:
        s = re.sub(r'[^\d.]', '', s)
        try:
            return float(s) if s else 0.0
        except ValueError:
            return 0.0
    
    def num_close(a: str, b: str, tol: float = 0.05) -> bool:
        na, nb = normalize_num(a), normalize_num(b)
        if na == 0 and nb == 0:
            return True
        if na == 0 or nb == 0:
            return False
        return abs(na - nb) / max(na, nb) <= tol
    
    return {
        'code_match': item1['code'].strip().lower() == item2['code'].strip().lower(),
        'qty_match': num_close(item1['qty'], item2['qty']),
        'unit_match': item1['unit'].strip().lower() == item2['unit'].strip().lower(),
        'rate_match': num_close(item1['rate'], item2['rate']),
        'amt_match': num_close(item1['amt'], item2['amt'])
    }

def calculate_metrics(matched: int, total_extracted: int, total_original: int) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score.
    """
    precision = matched / total_extracted if total_extracted > 0 else 0.0
    recall = matched / total_original if total_original > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1': round(f1 * 100, 2)
    }

def compare_boq_texts(extracted_boq: str, original_boq: str) -> str:
    """
    Compare extracted BOQ text to original BOQ text.
    Returns a concise markdown report with accuracy metrics and key differences.
    """
    extracted_items = parse_boq_table(extracted_boq)
    original_items = parse_boq_table(original_boq)
    
    if not extracted_items or not original_items:
        return "## BOQ Comparison Report\n\nError: Could not parse one or both BOQ tables. Ensure they are in markdown table format."
    
    matches = match_items(extracted_items, original_items)
    metrics = calculate_metrics(len(matches), len(extracted_items), len(original_items))
    
    report = f"""## BOQ Comparison Report

### Quick Metrics
- **Precision**: {metrics['precision']}% (correctly extracted items)
- **Recall**: {metrics['recall']}% (original items found)
- **F1 Score**: {metrics['f1']}% (overall accuracy)
- **Matched**: {len(matches)}/{len(extracted_items)} extracted, {len(matches)}/{len(original_items)} original

### Key Differences
"""
    
    unmatched_ext = [item for item in extracted_items if not any(item == m[0] for m in matches)]
    if unmatched_ext:
        report += "\n#### Extra in Extracted (not in original)\n"
        for item in unmatched_ext:
            report += f"- Item {item['code']}: {item['desc'][:60]}... (Qty: {item['qty']}, Unit: {item['unit']})\n"
    
    unmatched_orig = [item for item in original_items if not any(item == m[1] for m in matches)]
    if unmatched_orig:
        report += "\n#### Missing from Extracted (in original)\n"
        for item in unmatched_orig:
            report += f"- Item {item['code']}: {item['desc'][:60]}... (Qty: {item['qty']}, Unit: {item['unit']})\n"
    
    # Add field mismatches for matched items
    field_mismatches = []
    for ext_item, orig_item, score in matches:
        field_comp = compare_fields(ext_item, orig_item)
        mismatches = [k for k, v in field_comp.items() if not v]
        if mismatches:
            field_mismatches.append((ext_item, orig_item, mismatches))
    
    if field_mismatches:
        report += "\n#### Field Mismatches in Matched Items\n"
        for ext_item, orig_item, mismatches in field_mismatches:
            mismatch_str = ", ".join([f"{m.replace('_match', '').upper()}: Ext='{ext_item[m.replace('_match', '')]}' vs Orig='{orig_item[m.replace('_match', '')]}'" for m in mismatches])
            report += f"- Item {ext_item['code']}: {mismatch_str}\n"
    
    if not unmatched_ext and not unmatched_orig and not field_mismatches:
        report += "\n✅ No major differences found. All items matched successfully."
    
    return report

# Placeholders for manual input
extracted_boq = """
| Item No/Code | Description | Quantity | Unit | Rate | Amount |
|--------------|-------------|----------|------|------|--------|
| 1 | Supplying and making end termination with brass compression gland and aluminium  | 2 | Nos |  |  |
| 2 | Supply of following 3.5C 185 sq.mm size 1.1 KV grade XLPE insulated, PVC sheathe | 150 | Mtrs |  |  |
| 3 | Laying and fixing of one number PVC insulated and PVC sheathed / XLPE power cabl | 50 | Mtrs |  |  |
| 4 | Laying and fixing of one number PVC insulated and PVC sheathed / XLPE power cabl | 100 | Mtrs |  |  |
| 5 | Fabrication, supply and installation of 150 MM WIDTH X 50 MM DEPTH X 2MM THICK s | 200 | Mtrs |  |  |
| 6 | Wiring for light point / exhaust fan point / with 1.5 sq.mm FRLS" now comapre the accuracy? items can be suffled |  |  |  |  |
"""

original_boq = """
| Item No/Code | Description | Quantity | Unit | Rate | Amount |
|--------------|-------------|----------|------|------|--------|
| 1 | Supplying and making end termination with brass compression gland and aluminium  | 2 | Nos |  2000 |  |
| 2 | Supply of following 3.5C 185 sq.mm size 1.1 KV grade XLPE insulated, PVC sheathe | 150 | Mtrs | 2000 |  |
| 3 | Laying and fixing of one number PVC insulated and PVC sheathed / XLPE power cabl | 50 | Mtrs |   2000 |  |
| 4 | Laying and fixing of one number PVC insulated and PVC sheathed / XLPE power cabl | 100 | Mtrs | 2000 |  |
| 5 | Fabrication, supply and installation of 150 MM WIDTH X 50 MM DEPTH X 2MM THICK s | 200 | Mtrs | 2000 |  |
| 6 | Wiring for light point / exhaust fan point / with 1.5 sq.mm FRLS" now comapre the accuracy? items can be suffled | 2000 |  |  |  |
"""

if __name__ == "__main__":
    report = compare_boq_texts(extracted_boq, original_boq)
    print(report)