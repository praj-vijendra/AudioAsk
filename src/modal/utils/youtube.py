import yt_dlp
import os
import logging
from typing import Optional, Dict, Any, Union
from ..config.settings import YOUTUBE_DL_OPTIONS, MAX_FILE_SIZE

logger = logging.getLogger(__name__)

class YouTubeExtractor:
    def __init__(self, output_dir: str, youtube_token: Optional[str] = None, cookies_path: Optional[str] = None, browser_name: Optional[str] = None):
        self.output_dir = output_dir
        self.ydl_opts = YOUTUBE_DL_OPTIONS.copy()
        self.ydl_opts['outtmpl'] = os.path.join(output_dir, '%(id)s.%(ext)s')
        
        # Set up authentication if provided
        if youtube_token or cookies_path or browser_name:
            self._setup_authentication(youtube_token, cookies_path, browser_name)

    def _setup_authentication(self, token: Optional[str] = None, cookies_path: Optional[str] = None, browser_name: Optional[str] = None):
        """
        Setup YouTube authentication using one of the provided methods:
        1. Direct cookies file path
        2. Browser cookies export
        3. Token method
        """
        try:
            # Method 1: Direct cookie file path
            if cookies_path and os.path.exists(cookies_path):
                self.ydl_opts['cookiefile'] = cookies_path
                logger.info(f"Using provided cookies file for YouTube authentication")
                return
                
            # Method 2: Extract cookies from browser
            if browser_name:
                # Browser should be one of: chrome, chromium, firefox, opera, edge, safari
                if browser_name in ['chrome', 'chromium', 'firefox', 'opera', 'edge', 'safari']:
                    self.ydl_opts['cookiesfrombrowser'] = (browser_name, None, None, None)
                    logger.info(f"Using {browser_name} browser cookies for YouTube authentication")
                    return
                else:
                    logger.warning(f"Unsupported browser: {browser_name}. Using fallback authentication method.")
            
            # Method 3: Legacy token method
            if token:

                self.ydl_opts['cookiesfrombrowser'] = None  # Clear any browser cookie settings
                self.ydl_opts['cookies'] = token
                logger.info("Using YouTube authentication token")
                return
                
            logger.warning("No YouTube authentication method provided")
            
        except Exception as e:
            logger.warning(f"Failed to setup YouTube authentication: {str(e)}")

    def extract_audio(self, url: str) -> Optional[str]:
        """
        Extract audio from a YouTube URL.
        
        Args:
            url: YouTube video URL
            
        Returns:
            str: Path to the extracted audio file
        """
        output_path = None
        
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Get video info first to check duration
                info = ydl.extract_info(url, download=False)
                if info.get('duration', 0) > 3600:  # 1 hour limit
                    raise ValueError("Video duration exceeds maximum allowed duration of 1 hour")

                # Download the audio
                ydl.download([url])
                
                # Get the output file path
                video_id = info['id']
                output_path = os.path.join(self.output_dir, f"{video_id}.mp3")
                
                # Check file size
                if os.path.getsize(output_path) > MAX_FILE_SIZE:
                    os.remove(output_path)
                    raise ValueError("Extracted audio file exceeds maximum allowed size of 500MB")
                
                return output_path

        except Exception as e:
            logger.error(f"Error extracting audio from YouTube: {str(e)}")
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
            raise ValueError(f"Failed to extract audio from YouTube: {str(e)}")

    def cleanup(self, file_path: str):
        """Clean up the extracted audio file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            # Also clean up cookie file if it exists
            cookie_file = os.path.join(self.output_dir, 'youtube_cookies.txt')
            if os.path.exists(cookie_file):
                os.remove(cookie_file)
        except Exception as e:
            logger.warning(f"Failed to clean up file {file_path}: {str(e)}")