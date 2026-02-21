"""
BOQ extraction service for extracting Bill of Quantities from documents.
"""
import re
from typing import List, Optional, Tuple
from loguru import logger
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config.settings import settings
from core.llm import LLMClient
from prompts.get_prompts import (
    METADATA_EXTRACTION_TEMPLATE,
    BOQ_EXTRACTION_TEMPLATE,
    BOQ_IMPROVEMENT_TEMPLATE,
    BOQ_COLUMN_HEADERS,
)

# Page marker pattern for detecting source pages
PAGE_MARKER_PATTERN = r"(?i)(?:---\s*)?page\s+(\d+)(?:\s*---)?"


class BOQExtractor:
    """
    Extracts Bill of Quantities (BOQ) data from document chunks.
    
    Example:
        extractor = BOQExtractor(llm_client)
        boq_output = extractor.extract(chunks)
    """
    
    def __init__(self, llm_client: LLMClient, batch_size: int = None, max_prompt_length: int = None, page_search_length: int = None):
        """
        Initialize BOQ extractor.
        
        Args:
            llm_client: LLM client instance. Required.
            batch_size: Number of chunks per batch. Defaults to config value.
            max_prompt_length: Max chars in prompt. Defaults to config value.
            page_search_length: Chars for page detection. Defaults to config value.
        """
        self.llm_client = llm_client
        self.batch_size = batch_size or settings.boq.batch_size
        self.max_prompt_length = max_prompt_length or settings.boq.max_prompt_length
        self.page_search_length = page_search_length or settings.boq.page_search_length
        self.source_max_length = settings.boq.source_max_length
    
    def _batch_chunks(self, chunks: List[Document]) -> List[List[Document]]:
        """Split chunks into batches."""
        return [
            chunks[i:i + self.batch_size]
            for i in range(0, len(chunks), self.batch_size)
        ]
    
    def _extract_metadata(self, chunks: List[Document]) -> str:
        """Extract document metadata from first few chunks."""
        metadata_text = '\n\n'.join([chunk.page_content for chunk in chunks[:3]])
        prompt = METADATA_EXTRACTION_TEMPLATE.format(
            document_text=metadata_text[:2000]
        )
        
        logger.info('Invoking LLM for metadata extraction...')
        result = self.llm_client.invoke(prompt)
        logger.info('Metadata extraction completed')
        return result
    
    def _detect_page_source(self, desc: str, batch_text: str) -> str:
        """
        Detect the page number for a BOQ item based on its description.
        
        Args:
            desc: Item description.
            batch_text: Full batch text to search in.
        
        Returns:
            Page source string (e.g., "Page 5") or "Unknown".
        """
        search_str = desc[:self.page_search_length].strip().lower()
        batch_text_lower = batch_text.lower()
        pos = batch_text_lower.rfind(search_str)
        
        if pos != -1:
            matches = list(re.finditer(PAGE_MARKER_PATTERN, batch_text[:pos]))
            if matches:
                page = matches[-1].group(1)
                return f"Page {page}"
        
        return "Unknown"
    
    def _parse_boq_line(self, line: str, batch_text: str) -> Optional[str]:
        """
        Parse a single BOQ line from LLM output.
        
        Args:
            line: Raw line from LLM output.
            batch_text: Full batch text for page detection.
        
        Returns:
            Formatted BOQ item string or None if invalid.
        """
        line = line.strip()
        if not line or '|' not in line:
            return None
        
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 9:
            parts += ['NA'] * (9 - len(parts))
        
        # Add source column
        parts.append("Unknown")
        
        # Detect page source (description is now at position 3)
        desc = parts[3]
        parts[9] = self._detect_page_source(desc, batch_text)
        
        return '|'.join(parts[:10])
    
    def _extract_from_batch(self, batch_text: str, batch_num: int, previous_boq: Optional[str] = None, boq_mode: list = None, specific_boq: str = None) -> List[str]:
        """
        Extract BOQ items from a single batch.
        
        Args:
            batch_text: Combined text from batch chunks.
            batch_num: Batch number for logging.
            previous_boq: Previous BOQ output for improvement (optional).
            boq_mode: List of modes ["default", "specific BOQ"].
            specific_boq: Specific BOQ string if mode includes "specific BOQ".
        
        Returns:
            List of BOQ item strings.
        """
        prompt_text = batch_text[:self.max_prompt_length]
        
        # Determine extraction instruction based on mode
        # If both default and specific BOQ are selected, extract all
        # If only specific BOQ is selected, apply constraint
        if boq_mode and "specific BOQ" in boq_mode and "default" not in boq_mode and specific_boq:
            extraction_instruction = f"Extract only BOQ items that are related to or match the following: {specific_boq}. If no matching items are found, return NO_BOQ_ITEMS."
            specific_constraint = f"\n**IMPORTANT CONSTRAINT:** Only extract items related to: {specific_boq}. Ignore all other items."
        else:
            extraction_instruction = "Extract all BOQ items present in the text."
            specific_constraint = ""
        
        if previous_boq:
            base_prompt = BOQ_IMPROVEMENT_TEMPLATE
            # For specific BOQ mode, add constraint to improvement template
            if specific_constraint:
                base_prompt = base_prompt.replace("**Your task:**", f"{specific_constraint}\n\n**Your task:**")
            prompt = base_prompt.format(previous_boq=previous_boq, batch_text=prompt_text)
        else:
            base_prompt = BOQ_EXTRACTION_TEMPLATE
            # Insert instruction before "Text to analyze:"
            base_prompt = base_prompt.replace("Text to analyze:", f"{extraction_instruction}\n\nText to analyze:")
            prompt = base_prompt.format(batch_text=prompt_text)
        
        try:
            logger.info(f'Invoking LLM for BOQ extraction on batch {batch_num}...')
            result = self.llm_client.invoke(prompt)
            logger.info(f'LLM response received for batch {batch_num}')
            
            if 'NO_BOQ_ITEMS' in result:
                logger.info(f'No BOQ items found in batch {batch_num}')
                return []
            
            boq_items = []
            lines = result.strip().split('\n')
            
            for line in lines:
                parsed = self._parse_boq_line(line, batch_text)
                if parsed:
                    boq_items.append(parsed)
            
            logger.info(f'Extracted {len(boq_items)} BOQ items from batch {batch_num}')
            return boq_items
            
        except Exception as e:
            logger.warning(f'Error processing batch {batch_num}: {e}')
            return []
    
    def _format_output(self, unique_items: List[str], metadata_result: str) -> str:
        """
        Format extracted BOQ items into markdown output.
        
        Args:
            unique_items: List of unique BOQ item strings.
            metadata_result: Document metadata.
        
        Returns:
            Formatted markdown string.
        """
        logger.info('Formatting BOQ output...')
        
        if not unique_items:
            logger.info('No BOQ items to format')
            return f'''## DOCUMENT SUMMARY
{metadata_result}

## DETAILED BILL OF QUANTITIES
No BOQ items were found in this document.'''
        
        # Filter out header rows and separator rows
        filtered_items = []
        for item in unique_items:
            item_str = str(item).strip()
            # Skip header rows and separator rows
            if 'Item No' in item_str or 'Item Code' in item_str or '---' in item_str:
                logger.debug('Skipping header/separator row')
                continue
            if item_str:
                filtered_items.append(item)
        
        if not filtered_items:
            logger.info('No valid BOQ items found after filtering headers')
            return f'''## DOCUMENT SUMMARY
{metadata_result}

## DETAILED BILL OF QUANTITIES
No BOQ items were found in this document.'''
        
        # Determine which columns have data
        cols_present = [False] * 10
        normalized_items = []
        
        for item in filtered_items:
            parts = [p.strip() for p in item.split('|')]
            if len(parts) < 10:
                parts += ['NA'] * (10 - len(parts))
            normalized_items.append(parts[:10])
            
            for i in range(10):
                if parts[i] and parts[i].upper() != 'NA':
                    cols_present[i] = True
        
        # Build column indices (always include item code, category, description, unit rate, and total amount)
        col_indices = [i for i, present in enumerate(cols_present) if present]
        if 0 not in col_indices:
            col_indices.append(0)
        if 2 not in col_indices:
            col_indices.append(2)
        if 3 not in col_indices:
            col_indices.append(3)
        if 6 not in col_indices:
            col_indices.append(6)
        if 7 not in col_indices:
            col_indices.append(7)
        col_indices.sort()  # Maintain column order
        
        # Build header and separator rows
        header_row = '| ' + ' | '.join([BOQ_COLUMN_HEADERS[i] for i in col_indices]) + ' |\n'
        sep_row = '|' + '|'.join(['-' * (len(BOQ_COLUMN_HEADERS[i]) + 2) for i in col_indices]) + '|\n'
        
        formatted_boq = f'''## DOCUMENT SUMMARY
{metadata_result}

## DETAILED BILL OF QUANTITIES
**Total Items Found:** {len(normalized_items)}

{header_row}{sep_row}'''
        
        # Add data rows
        for parts in normalized_items:
            # Truncate source if too long
            parts[9] = parts[9][:self.source_max_length] if len(parts[9]) > self.source_max_length else parts[9]
            row_vals = [parts[i] for i in col_indices]
            
            # Format confidence score (now at position 8)
            if 8 in col_indices:
                conf_idx = col_indices.index(8)
                if row_vals[conf_idx] != 'NA':
                    row_vals[conf_idx] = row_vals[conf_idx].rstrip('%') + '%'
                    if parts[9] == "Unknown":
                        row_vals[conf_idx] = "N/A"
            
            # Replace NA with dash for display
            row_vals = ['-' if val == 'NA' else val for val in row_vals]
            
            formatted_boq += '| ' + ' | '.join(row_vals) + ' |\n'
        
        # Clean up formatting
        try:
            s = formatted_boq.replace('\r\n', '\n').replace('\r', '\n')
            lines = [ln.lstrip() for ln in s.split('\n')]
            header_idx = next((i for i, ln in enumerate(lines) if ln.startswith('| ')), None)
            
            if header_idx and header_idx > 0 and lines[header_idx - 1].strip():
                lines.insert(header_idx, '')
            
            if header_idx:
                sep_idx = header_idx + 1
                if not (sep_idx < len(lines) and re.match(r'^\|\s*-+', lines[sep_idx])):
                    cols = [c for c in lines[header_idx].split('|') if c.strip()]
                    sep = '|' + '|'.join(['---' for _ in cols]) + '|'
                    lines.insert(sep_idx, sep)
            
            formatted_boq = '\n'.join(lines).strip() + '\n\n'
        except Exception:
            pass
        
        return formatted_boq
    
    def _merge_multiple_runs(self, all_outputs: List[str]) -> str:
        """
        Merge BOQ outputs from multiple runs by extracting and deduplicating items.
        Preserves the best metadata and combines unique items from all runs.
        
        Args:
            all_outputs: List of BOQ output strings from each run.
        
        Returns:
            Merged BOQ output as markdown string.
        """
        if not all_outputs:
            return ""
        
        if len(all_outputs) == 1:
            return all_outputs[0]
        
        logger.info(f'Merging {len(all_outputs)} run outputs...')
        
        # Extract all items from all runs
        all_items = []
        metadata_lines = []
        
        for run_idx, output in enumerate(all_outputs, 1):
            if not output or output.strip() == "NO_BOQ_ITEMS":
                logger.info(f'Run {run_idx}: No items to merge (empty or NO_BOQ_ITEMS)')
                continue
            
            lines = output.strip().split('\n')
            items_in_run = []
            
            # Extract metadata from first run
            if run_idx == 1:
                for line in lines:
                    if line.startswith('**Project Name:**') or \
                       line.startswith('**Tender No:**') or \
                       line.startswith('**Contract:**') or \
                       line.startswith('**Date:**'):
                        metadata_lines.append(line)
            
            # Extract table rows (skip header and separator)
            in_table = False
            header_found = False
            for line_idx, line in enumerate(lines):
                stripped = line.strip()
                
                # Detect table header - support multiple formats
                if stripped.startswith('|') and any(keyword in stripped for keyword in ['S.No', 'Item Code', 'Item No/Code', 'Item Name']):
                    in_table = True
                    header_found = True
                    continue
                
                # Skip separator line
                if in_table and '|---' in stripped:
                    continue
                
                # Extract data rows
                if in_table and stripped.startswith('|') and stripped.endswith('|'):
                    # Make sure it's not the header again (check for header keywords)
                    if not any(keyword in stripped for keyword in ['S.No', 'Item Code', 'Item No/Code', 'Item Name', 'Item No.', 'Category', 'Item Description', 'Quantity', 'Unit', 'Confidence Score']):
                        items_in_run.append(stripped)
            
            logger.info(f'Run {run_idx}: Header found={header_found}, Extracted {len(items_in_run)} items')
            if not header_found and run_idx == 1:
                logger.warning(f'Run {run_idx}: No table header found! Output might be malformed.')
            
            all_items.extend(items_in_run)
        
        # Deduplicate items based on item code and description
        unique_items_map = {}
        for item_line in all_items:
            parts = [p.strip() for p in item_line.split('|') if p.strip()]
            if len(parts) >= 3:  # Need at least S.No, Code, Description
                # Create key from code + description (case-insensitive)
                code = parts[1] if len(parts) > 1 else ""
                desc = parts[2] if len(parts) > 2 else ""
                key = f"{code.lower()}||{desc.lower()}"
                
                # Keep first occurrence (from earlier run)
                if key not in unique_items_map:
                    unique_items_map[key] = item_line
        
        unique_items = list(unique_items_map.values())
        logger.info(f'After deduplication: {len(unique_items)} unique items')
        
        # If no items were found in any run, return the first non-empty output
        if not unique_items:
            logger.warning('No items found in any run after parsing. Returning first non-empty output.')
            for output in all_outputs:
                if output and output.strip() and output.strip() != "NO_BOQ_ITEMS":
                    return output
            return all_outputs[0] if all_outputs else ""
        
        # Rebuild the BOQ with merged items
        output_lines = []
        
        # Add metadata
        if metadata_lines:
            output_lines.append('## BOQ METADATA')
            output_lines.extend(metadata_lines)
            output_lines.append('')
        
        # Add table
        output_lines.append('## DETAILED BILL OF QUANTITIES')
        output_lines.append(f'**Total Items Found:** {len(unique_items)}')
        output_lines.append('')
        output_lines.append('| Item No. | Item Name | Category | Description | Quantity | Unit | Unit Rate (₹) | Total Amount (₹) | Confidence Score | Source |')
        output_lines.append('|---------|-----------|----------|-------------|----------|------|-----------------|---------------------|-----------------|--------|')
        
        # Re-number items
        for idx, item_line in enumerate(unique_items, 1):
            parts = [p.strip() for p in item_line.split('|') if p.strip()]
            if parts:
                parts[0] = str(idx)  # Update item number
                # Replace NA with dash for display
                parts = ['-' if p == 'NA' else p for p in parts]
                output_lines.append('| ' + ' | '.join(parts) + ' |')
        
        merged_output = '\n'.join(output_lines) + '\n\n'
        logger.info(f'Merge complete: {len(unique_items)} items in final BOQ')
        
        return merged_output
    
    def extract(self, chunks: List[Document], vector_store: FAISS = None, previous_boq: Optional[str] = None, boq_mode: list = None, specific_boq: str = None) -> str:
        """
        Extract BOQ from document chunks.
        
        Args:
            chunks: List of Document chunks.
            vector_store: Optional vector store (not used currently).
            previous_boq: Previous BOQ output for improvement (optional).
            boq_mode: List of modes ["default", "specific BOQ"].
            specific_boq: Specific BOQ string if mode includes "specific BOQ".
        
        Returns:
            Formatted BOQ output as markdown string.
        """
        try:
            logger.info(f'Starting comprehensive BOQ extraction from {len(chunks)} chunks')
            
            # Extract metadata
            logger.info('Extracting document metadata...')
            metadata_result = self._extract_metadata(chunks)
            
            # Create batches
            logger.info('Creating batches...')
            batches = self._batch_chunks(chunks)
            logger.info(f'Created {len(batches)} batches')
            
            # Extract from each batch
            boq_items = []
            for batch_num, batch_chunks in enumerate(batches, 1):
                logger.info(f'Processing batch {batch_num}/{len(batches)} ({len(batch_chunks)} chunks)')
                
                chunk_texts = [chunk.page_content for chunk in batch_chunks]
                batch_text = '\n\n'.join(chunk_texts)
                logger.info(f'Batch text length: {len(batch_text)}')
                
                batch_items = self._extract_from_batch(batch_text, batch_num, previous_boq, boq_mode, specific_boq)
                boq_items.extend(batch_items)
                logger.info(f'Batch {batch_num} yielded {len(batch_items)} items')
            
            # Deduplicate
            unique_items = list(dict.fromkeys(boq_items))
            logger.info(f'Found {len(unique_items)} unique BOQ items after deduplication')
            
            # Format output
            logger.info('Formatting BOQ output...')
            formatted_boq = self._format_output(unique_items, metadata_result)
            
            logger.info('Comprehensive BOQ extraction completed successfully')
            return formatted_boq
            
        except Exception as e:
            logger.error(f'Error in comprehensive BOQ extraction: {e}')
            raise
    
    def extract_iterative(self, chunks: List[Document], vector_store: FAISS, runs: int, boq_mode: list = None, specific_boq: str = None) -> Tuple[str, List[str]]:
        """
        Extract BOQ iteratively, improving with each run.
        
        Args:
            chunks: Document chunks.
            vector_store: Vector store.
            runs: Number of runs (1-5).
            boq_mode: List of modes ["default", "specific BOQ"].
            specific_boq: Specific BOQ string if mode includes "specific BOQ".
        
        Returns:
            Tuple of (final_boq, list_of_all_outputs).
        """
        logger.info(f'BOQ extraction mode: {boq_mode}, Specific BOQ: {specific_boq}')
        all_outputs = []
        previous_boq = None
        
        # Handle single run case
        if runs == 1:
            logger.info('Starting single BOQ extraction (runs=1)')
            try:
                current_output = self.extract(chunks, vector_store, previous_boq, boq_mode, specific_boq)
                logger.info('Single extraction completed')
            except Exception as e:
                logger.error(f'Single extraction failed: {e}')
                raise
            all_outputs.append(current_output)
            return current_output, all_outputs
        
        # Handle multiple runs case
        for run in range(runs):
            try:
                logger.info(f'Starting iterative run {run + 1}/{runs}')
                current_output = self.extract(chunks, vector_store, previous_boq, boq_mode, specific_boq)
                logger.info(f'Iterative run {run + 1} completed')
            except Exception as e:
                logger.warning(f'Iterative run {run + 1} failed: {e}, using previous output')
                current_output = previous_boq if previous_boq else ""
            
            all_outputs.append(current_output)
            # For next iteration, pass the full previous output for improvement
            # This allows the improvement template to see all previously extracted items
            if current_output and run < runs - 1:
                # Pass the complete previous output for context improvement
                previous_boq = current_output
            else:
                previous_boq = None
        
        # Merge all runs by extracting and deduplicating BOQ items
        merged_boq = self._merge_multiple_runs(all_outputs)
        logger.info(f'Merged {len(all_outputs)} runs into final BOQ')
        
        return merged_boq, all_outputs
