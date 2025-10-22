"""
Voice Processing for Human-Centered AI Assistant
Accessibility-focused speech-to-text and text-to-speech
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
try:
    import faster_whisper
    WHISPER_AVAILABLE = True
except ImportError:
    try:
        import whisper
        WHISPER_AVAILABLE = True
        WHISPER_VARIANT = "openai"
    except ImportError:
        WHISPER_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

class VoiceProcessor:
    """
    Accessibility-focused voice processing
    
    Human-Centered Design:
    - Local processing for privacy
    - Graceful degradation when deps missing
    - Clear error messages
    - Multiple TTS options
    """
    
    def __init__(self):
        self.whisper_model = None
        self.tts_engine = None
        
        # Initialize Whisper for transcription
        if WHISPER_AVAILABLE:
            self._initialize_whisper()
        
        # Initialize TTS
        if PYTTSX3_AVAILABLE:
            self._initialize_tts()
    
    def _initialize_whisper(self):
        """Initialize Whisper model for transcription"""
        try:
            if "faster_whisper" in globals():
                # Use faster-whisper (more efficient)
                self.whisper_model = faster_whisper.WhisperModel(
                    "base",  # Good balance of speed/accuracy
                    device="cpu",
                    compute_type="int8"
                )
                self.whisper_variant = "faster"
            else:
                # Use openai-whisper (fallback)
                self.whisper_model = whisper.load_model("base")
                self.whisper_variant = "openai"
            
            logger.info(f"Whisper initialized: {self.whisper_variant}")
            
        except Exception as e:
            logger.error(f"Error initializing Whisper: {e}")
            self.whisper_model = None
    
    def _initialize_tts(self):
        """Initialize TTS engine"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure voice settings for accessibility
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer female voice if available (often clearer)
                for voice in voices:
                    if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            # Set reasonable speaking rate
            self.tts_engine.setProperty('rate', 180)  # words per minute
            self.tts_engine.setProperty('volume', 0.9)
            
            logger.info("TTS engine initialized")
            
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}")
            self.tts_engine = None
    
    async def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        if not self.whisper_model:
            raise Exception("Speech recognition not available. Install: pip install faster-whisper")
        
        try:
            if self.whisper_variant == "faster":
                # faster-whisper
                segments, _ = self.whisper_model.transcribe(audio_path)
                text = " ".join([segment.text for segment in segments])
            else:
                # openai-whisper
                result = self.whisper_model.transcribe(audio_path)
                text = result["text"]
            
            # Clean up transcription
            text = text.strip()
            
            logger.info(f"Transcribed audio: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise Exception(f"Failed to transcribe audio: {str(e)}")
    
    async def text_to_speech(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Convert text to speech audio file
        
        Args:
            text: Text to speak
            output_path: Optional output file path
            
        Returns:
            Path to generated audio file
        """
        if not self.tts_engine:
            raise Exception("Text-to-speech not available. Install: pip install pyttsx3")
        
        try:
            # Create output path if not provided
            if not output_path:
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=".wav",
                    dir="/tmp"
                )
                temp_file.close()
                output_path = temp_file.name
            
            # Generate speech
            self.tts_engine.save_to_file(text, output_path)
            self.tts_engine.runAndWait()
            
            # Verify file was created
            if not os.path.exists(output_path):
                raise Exception("TTS file generation failed")
            
            logger.info(f"Generated TTS audio: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            raise Exception(f"Failed to generate speech: {str(e)}")
    
    def is_available(self) -> dict:
        """Check availability of voice features"""
        return {
            "transcription": self.whisper_model is not None,
            "text_to_speech": self.tts_engine is not None,
            "whisper_variant": getattr(self, "whisper_variant", None),
            "dependencies": {
                "whisper": WHISPER_AVAILABLE,
                "pyttsx3": PYTTSX3_AVAILABLE
            }
        }

# Global instance
voice_processor = VoiceProcessor()