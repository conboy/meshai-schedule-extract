#!/usr/bin/env python3
"""
Script to extract shift information from PDF or PNG files using OpenRouter API.
Optimized with connection pooling, image compression, and concurrent processing.
"""

# Standard library imports
import asyncio
import base64
import io
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Third-party imports
import aiofiles
import aiohttp
from PIL import Image


# Configuration constants
class Config:
    # API settings
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    API_TIMEOUT = 300
    
    # Connection pooling
    CONNECTION_POOL_SIZE = 10
    CONNECTIONS_PER_HOST = 5
    DNS_CACHE_TTL = 300
    
    # Image compression
    IMAGE_QUALITY = 85
    MAX_IMAGE_SIZE = (1920, 1080)
    SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg']
    
    # PDF conversion
    PDF_DPI = 150
    
    # Batch processing
    BATCH_SIZE = 5
    MAX_CONCURRENT_REQUESTS = 3
    
    # File extensions
    SUPPORTED_EXTENSIONS = ['.pdf', '.png', '.jpg', '.jpeg']
    
    # Logging
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Global session for connection pooling
_session = None

async def get_session():
    """Get or create a shared aiohttp session with connection pooling."""
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=Config.API_TIMEOUT)
        connector = aiohttp.TCPConnector(
            limit=Config.CONNECTION_POOL_SIZE,
            limit_per_host=Config.CONNECTIONS_PER_HOST,
            ttl_dns_cache=Config.DNS_CACHE_TTL,
            use_dns_cache=True,
        )
        _session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": "schedule-extract/1.0"}
        )
    return _session

async def close_session():
    """Close the global session."""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None

def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration with optional file output."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=Config.LOG_FORMAT,
        datefmt=Config.DATE_FORMAT,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

try:
    from pdf2image import convert_from_path
except ImportError:
    logging.warning("pdf2image not installed. Install with: uv add pdf2image")
    convert_from_path = None


async def compress_image(image_path: str, quality: int = Config.IMAGE_QUALITY, max_size: Tuple[int, int] = Config.MAX_IMAGE_SIZE) -> bytes:
    """Compress image to reduce file size while maintaining quality."""
    logger = logging.getLogger(__name__)
    
    try:
        # Open and compress image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Resize if too large
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                logger.debug(f"Resized image from original to {img.size}")
            
            # Compress to memory
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            compressed_data = buffer.getvalue()
            
            original_size = os.path.getsize(image_path)
            compression_ratio = len(compressed_data) / original_size
            logger.debug(f"Compressed image: {original_size} → {len(compressed_data)} bytes ({compression_ratio:.2%})")
            
            return compressed_data
    except Exception as e:
        logger.warning(f"Failed to compress image {image_path}: {e}, using original")
        # Fall back to original file
        async with aiofiles.open(image_path, 'rb') as f:
            return await f.read()

async def read_file_as_base64(file_path: str) -> str:
    """Read file, compress if image, and encode as base64."""
    logger = logging.getLogger(__name__)
    try:
        logger.debug(f"Reading file for base64 encoding: {file_path}")
        
        # Check if it's an image file that can be compressed
        if Path(file_path).suffix.lower() in Config.SUPPORTED_IMAGE_FORMATS:
            data = await compress_image(file_path)
        else:
            # For PDFs and other files, read normally
            async with aiofiles.open(file_path, 'rb') as f:
                data = await f.read()
        
        logger.debug(f"File size: {len(data)} bytes")
        return base64.b64encode(data).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise


async def convert_pdf_to_png(pdf_path: str, output_dir: Optional[str] = None, dpi: int = Config.PDF_DPI) -> List[str]:
    """Convert PDF to PNG files using pdf2image with caching and optimization."""
    logger = logging.getLogger(__name__)
    
    if convert_from_path is None:
        logger.error("pdf2image package not available. Install with: uv add pdf2image")
        return []
    
    if output_dir is None:
        output_dir = Path(pdf_path).parent / "preprocessing"
    
    output_prefix = Path(pdf_path).stem
    output_dir = Path(output_dir)
    
    logger.info(f"Converting PDF to PNG: {pdf_path}")
    logger.debug(f"Output directory: {output_dir}")
    logger.debug(f"Output prefix: {output_prefix}")
    
    # Create preprocessing directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    logger.debug(f"Created/verified output directory: {output_dir}")
    
    try:
        # Convert PDF to images in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=2) as executor:
            logger.debug(f"Starting PDF to image conversion (DPI: {dpi})")
            images = await loop.run_in_executor(executor, convert_from_path, pdf_path, dpi)
            logger.info(f"PDF converted to {len(images)} pages")
            
            # Save images concurrently
            async def save_image(image, page_num):
                output_file = output_dir / f"{output_prefix}-{page_num}.png"
                logger.debug(f"Saving page {page_num} to: {output_file}")
                await loop.run_in_executor(executor, image.save, output_file, 'PNG')
                return output_file
            
            # Process all pages concurrently
            save_tasks = [save_image(image, i) for i, image in enumerate(images, 1)]
            png_files = await asyncio.gather(*save_tasks)
        
        logger.info(f"Successfully converted {pdf_path} to {len(png_files)} PNG files")
        return sorted(png_files)
        
    except Exception as e:
        logger.error(f"Error converting PDF to PNG: {e}", exc_info=True)
        return []

def get_mime_type(file_path):
    """Get MIME type based on file extension."""
    ext = Path(file_path).suffix.lower()
    mime_types = {
        '.pdf': 'application/pdf',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
    }
    return mime_types.get(ext, 'application/octet-stream')

async def extract_shifts_from_file(file_path, api_key, model="google/gemini-2.5-flash-preview-05-20"):
    """Extract shift information from an image file using OpenRouter API."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Extracting shifts from: {file_path}")
    logger.debug(f"Using model: {model}")
    
    # Read and encode file
    file_base64 = await read_file_as_base64(file_path)
    mime_type = get_mime_type(file_path)
    logger.debug(f"File MIME type: {mime_type}")
    
    # Create data URL
    data_url = f"data:{mime_type};base64,{file_base64}"
    logger.debug(f"Created data URL (length: {len(data_url)} chars)")
    
    # Prepare the prompt
    prompt = """Extract ALL shifts from this schedule. Return ONLY a JSON array with these fields for each shift: Start Date, Start Time, End Date, End Time, Job, Member, Worked Hours, Note. Use null for missing fields. Be concise but complete - extract EVERY shift shown."""
    logger.debug(f"Using prompt: {prompt}")
    
    # Prepare request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Use image_url format for all files (PDFs are converted to images first)
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ],
        "response_format": {"type": "json_object"},
        "structured_outputs": True
    }
    
    logger.debug("Sending request to OpenRouter API")
    
    # Make request using shared session
    try:
        session = await get_session()
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        ) as response:
            
            logger.debug(f"API response status: {response.status}")
            
            if response.status == 200:
                result = await response.json()
                content = result['choices'][0]['message']['content']
                logger.info(f"Successfully extracted shifts (response length: {len(content)} chars)")
                logger.debug(f"API response content: {content[:200]}...")
                return content
            else:
                error_text = await response.text()
                logger.error(f"API request failed: {response.status}")
                logger.error(f"Response text: {error_text}")
                raise Exception(f"API request failed: {response.status} - {error_text}")
            
    except asyncio.TimeoutError:
        logger.error("API request timed out after 5 minutes")
        raise Exception("API request timed out")
    except aiohttp.ClientError as e:
        logger.error(f"Request error: {e}")
        raise Exception(f"Request error: {e}")

async def save_json_result(result, output_file):
    """Save API result to JSON file with error handling."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Saving results to: {output_file}")
    async with aiofiles.open(output_file, 'w') as f:
        try:
            json_result = json.loads(result)
            await f.write(json.dumps(json_result, indent=2))
            logger.debug("Successfully parsed and saved JSON result")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}, attempting to fix")
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', result, re.DOTALL)
            if json_match:
                logger.debug("Found JSON in markdown code block, extracting")
                try:
                    json_result = json.loads(json_match.group(1))
                    await f.write(json.dumps(json_result, indent=2))
                    logger.info("Successfully extracted and saved JSON from markdown")
                except json.JSONDecodeError:
                    logger.warning("Extracted JSON still invalid, wrapping response")
                    json_result = {
                        "raw_response": result,
                        "note": "API response was not valid JSON, wrapped in this structure"
                    }
                    await f.write(json.dumps(json_result, indent=2))
            else:
                logger.debug("No markdown JSON found, attempting to fix truncated JSON")
                try:
                    if result.strip().startswith('[') and not result.strip().endswith(']'):
                        logger.debug("Detected truncated JSON array, attempting to fix")
                        last_complete = result.rfind('},')
                        if last_complete != -1:
                            fixed_json = result[:last_complete + 1] + '\n]'
                            json_result = json.loads(fixed_json)
                            await f.write(json.dumps(json_result, indent=2))
                            logger.info("Successfully fixed truncated JSON array")
                        else:
                            logger.warning("Could not fix truncated JSON, wrapping response")
                            json_result = {
                                "raw_response": result,
                                "note": "API response was truncated and could not be fixed"
                            }
                            await f.write(json.dumps(json_result, indent=2))
                    else:
                        logger.warning("Response not a valid JSON array, wrapping")
                        json_result = {
                            "raw_response": result,
                            "note": "API response was not valid JSON, wrapped in this structure"
                        }
                        await f.write(json.dumps(json_result, indent=2))
                except Exception as fix_error:
                    logger.error(f"Error while trying to fix JSON: {fix_error}")
                    json_result = {
                        "raw_response": result,
                        "note": "API response could not be parsed as JSON"
                    }
                    await f.write(json.dumps(json_result, indent=2))

async def batch_process_small_images(png_files, api_key, max_batch_size=3):
    """Process ALL PNG files concurrently - no deferred processing."""
    logger = logging.getLogger(__name__)
    
    # Process ALL files concurrently at once
    logger.info(f"Processing ALL {len(png_files)} PNG files concurrently")
    tasks = [process_single_png_file(png_file, api_key) for png_file in png_files]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Exception processing {png_files[i]}: {result}")
            final_results.append((png_files[i], None))
        else:
            final_results.append(result)
    
    return final_results

async def process_single_png_file(png_file, api_key):
    """Process a single PNG file."""
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"Processing: {png_file}")
        result = await extract_shifts_from_file(png_file, api_key)
        output_file = f"{Path(png_file).stem}.json"
        await save_json_result(result, output_file)
        logger.info(f"Results saved to: {output_file}")
        return (png_file, result)
    except Exception as e:
        logger.error(f"Failed to process {png_file}: {e}")
        return (png_file, None)

async def process_multiple_files(png_files, api_key):
    """Process multiple PNG files with batching optimization."""
    return await batch_process_small_images(png_files, api_key)

async def process_single_schedule_file(file_path, api_key):
    """Process a single schedule file (PDF or image). Used by main.py."""
    logger = logging.getLogger(__name__)
    
    try:
        file_ext = Path(file_path).suffix.lower()
        logger.debug(f"Processing {file_ext} file: {file_path}")
        
        # Handle PDF files: convert to PNG first
        if file_ext == '.pdf':
            logger.info("PDF file detected, converting to PNG first")
            png_files = await convert_pdf_to_png(file_path)
            
            if not png_files:
                logger.error("Failed to convert PDF to PNG files")
                return False
            
            logger.info(f"Successfully converted to {len(png_files)} PNG files")
            
            # Process all PNG files
            logger.info(f"Processing {len(png_files)} PNG files")
            results = await process_multiple_files(png_files, api_key)
            
            successful_results = [r for r in results if r[1] is not None]
            logger.info(f"Successfully processed {len(successful_results)}/{len(png_files)} files")
            return len(successful_results) > 0
        
        # Regular processing for image files
        else:
            logger.info("Processing image file directly")
            result = await extract_shifts_from_file(file_path, api_key)
            
            # Save result to file
            output_file = f"{Path(file_path).stem}.json"
            await save_json_result(result, output_file)
            logger.info(f"Results saved to: {output_file}")
            return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)
        return False

async def main():
    logger = setup_logging()
    
    if len(sys.argv) != 2:
        logger.error("Incorrect usage")
        print("Usage: python extract_shifts.py <file_path>")
        print("Supported formats: PDF, PNG, JPG, JPEG")
        print("Note: PDF files are automatically converted to PNG files first")
        sys.exit(1)
    
    file_path = sys.argv[1]
    logger.info(f"Processing file: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        sys.exit(1)
    
    # Get API key from environment
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)
    
    logger.debug("API key found and verified")
    
    try:
        file_ext = Path(file_path).suffix.lower()
        logger.debug(f"File extension: {file_ext}")
        
        # Handle PDF files: convert to PNG first
        if file_ext == '.pdf':
            logger.info("PDF file detected, converting to PNG first")
            png_files = await convert_pdf_to_png(file_path)
            
            if not png_files:
                logger.error("Failed to convert PDF to PNG files")
                sys.exit(1)
            
            logger.info(f"Successfully converted to {len(png_files)} PNG files")
            
            # Process all PNG files concurrently
            logger.info(f"Processing {len(png_files)} PNG files concurrently")
            results = await process_multiple_files(png_files, api_key)
            
            successful_results = [r for r in results if r[1] is not None]
            logger.info(f"Completed processing all {len(png_files)} PNG files from PDF conversion")
            logger.info(f"Successfully processed {len(successful_results)}/{len(png_files)} files")
            return
        
        # Regular processing for image files
        else:
            logger.info("Processing image file directly")
            result = await extract_shifts_from_file(file_path, api_key)
            
            # Save result to file
            output_file = f"{Path(file_path).stem}.json"
            await save_json_result(result, output_file)
            logger.info(f"Results saved to: {output_file}")
            logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up the global session
        await close_session()

if __name__ == "__main__":
    asyncio.run(main())
