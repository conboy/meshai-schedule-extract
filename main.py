#!/usr/bin/env python3
"""
Main script to process all PDF and PNG files in the data/ directory 
using optimized async batch processing.
"""

# Standard library imports
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

# Local imports
sys.path.append(str(Path(__file__).parent))
from extract_shifts import close_session, process_single_schedule_file, setup_logging


# Configuration constants
class Config:
    DATA_DIR = "data"
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    SUPPORTED_EXTENSIONS = ['.pdf', '.png', '.jpg', '.jpeg']


def find_schedule_files(data_dir: str) -> List[Path]:
    """Find all supported schedule files in the data directory."""
    logger = logging.getLogger(__name__)
    data_path = Path(data_dir)
    
    logger.info(f"Scanning directory: {data_dir}")
    
    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return []
    
    # Find all supported files
    all_files = []
    for extension in Config.SUPPORTED_EXTENSIONS:
        pattern = f"*{extension}"
        files = list(data_path.glob(pattern))
        all_files.extend(files)
    
    # Count files by type
    file_counts = {}
    for file in all_files:
        ext = file.suffix.lower()
        file_counts[ext] = file_counts.get(ext, 0) + 1
    
    # Log counts
    count_strs = [f"{count} {ext.upper()}" for ext, count in file_counts.items()]
    logger.info(f"Found {len(all_files)} files: {', '.join(count_strs)}")
    
    if logger.isEnabledFor(logging.DEBUG):
        for file in all_files:
            logger.debug(f"Found file: {file}")
    
    return sorted(all_files)

async def process_file(file_path: Path, api_key: str) -> bool:
    """Process a single file using direct function calls."""
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    try:
        result = await process_single_schedule_file(file_path, api_key)
        processing_time = time.time() - start_time
        
        if result:
            logger.info(f"Successfully processed {file_path.name} in {processing_time:.2f}s")
            return True
        else:
            logger.error(f"Failed to process {file_path.name}")
            return False
            
    except asyncio.TimeoutError:
        processing_time = time.time() - start_time
        logger.error(f"Timeout processing {file_path.name} (exceeded timeout, processed for {processing_time:.2f}s)")
        return False
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Exception processing {file_path.name} after {processing_time:.2f}s: {e}", exc_info=True)
        return False

async def main():
    """Main function to process all files in the data directory with async batch processing."""
    log_file = f'schedule_extract_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logger = setup_logging(log_file)
    
    logger.info("Starting optimized batch processing of schedule files")
    
    # Check if API key is set
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set")
        logger.info("Please set your OpenRouter API key: export OPENROUTER_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Set up paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / Config.DATA_DIR
    
    logger.debug(f"Script directory: {script_dir}")
    logger.debug(f"Data directory: {data_dir}")
    
    # Find all files to process
    files_to_process = find_schedule_files(data_dir)
    
    if not files_to_process:
        logger.error("No supported files found in data/ directory")
        logger.info("Supported formats: PDF, PNG, JPG, JPEG")
        sys.exit(1)
    
    logger.info(f"Processing {len(files_to_process)} files concurrently")
    for i, file_path in enumerate(files_to_process, 1):
        logger.debug(f"  {i}. {file_path.name}")
    
    # Process files concurrently without limits
    start_time = time.time()
    
    # Create tasks for all files
    tasks = [process_file(file_path, api_key) for file_path in files_to_process]
    
    # Process all files concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count results
    successful = sum(1 for result in results if result is True)
    failed = len(results) - successful
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info("=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files: {len(files_to_process)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {duration:.2f} seconds")
    logger.info(f"Average time per file: {duration/len(files_to_process):.2f} seconds")
    
    # List generated JSON files
    json_files = list(script_dir.glob("*.json"))
    if json_files:
        logger.info(f"Generated {len(json_files)} JSON output files:")
        for json_file in sorted(json_files):
            logger.info(f"  - {json_file.name}")
    
    if failed > 0:
        logger.warning(f"{failed} files failed to process. Check the log for details.")
        await close_session()
        sys.exit(1)
    
    logger.info("All files processed successfully!")
    await close_session()

if __name__ == "__main__":
    asyncio.run(main())