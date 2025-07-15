import pandas as pd
import base64
from typing import Optional
from logger import get_logger
from config import OPENROUTER_API_KEY
from session import get_session

async def extract_shifts_from_csv_with_png(csv_path: str, png_path: str, api_key: str = None, model: str = "google/gemini-2.5-flash-preview-05-20") -> Optional[str]:
    """Extract shift information from CSV content using OpenRouter API with PNG attachment."""
    logger = get_logger(__name__)
    
    if api_key is None:
        api_key = OPENROUTER_API_KEY
        if not api_key:
            logger.error("No API key provided and OPENROUTER_API_KEY not set in environment")
            return None
    
    logger.info(f"Extracting shifts from CSV: {csv_path} with PNG: {png_path}")
    
    try:
        # Read CSV content
        df = pd.read_csv(csv_path)
        csv_content = df.to_string(index=False)
        
        # Read and encode PNG
        with open(png_path, 'rb') as f:
            png_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Prepare the prompt
        prompt = f"""Below is a CSV extracted from an Excel file using pandas, along with the original PNG image of the schedule. Your task is to extract ALL shifts from this CSV schedule data. Return ONLY a JSON array with these fields for each shift: Start Date, Start Time, End Date, End Time, Job, Member, Worked Hours, Note. Use null for missing fields. Be concise but complete - extract EVERY shift shown.

CSV Data:
{csv_content}"""
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
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
                                "url": f"data:image/png;base64,{png_data}"
                            }
                        }
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "structured_outputs": True
        }
        
        # Make request using shared session
        session = await get_session()
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        ) as response:
            
            if response.status == 200:
                result = await response.json()
                content = result['choices'][0]['message']['content']
                logger.info(f"Successfully extracted shifts from CSV with PNG")
                return content
            else:
                error_text = await response.text()
                logger.error(f"API request failed: {response.status} - {error_text}")
                return None
                
    except Exception as e:
        logger.error(f"Error extracting shifts from CSV with PNG: {e}")
        return None