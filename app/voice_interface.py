"""
Voice Interface for SLM Personal Agent
Enables voice input and text-to-speech output for hands-free interaction
"""

import asyncio
import io
import logging
import tempfile
import os
from typing import Optional, Dict, Any
from fastapi import UploadFile
import speech_recognition as sr
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class VoiceRequest(BaseModel):
    text: str
    voice: str = "default"  # Voice selection for TTS
    speed: float = 1.0      # Speech speed
    pitch: float = 1.0      # Speech pitch

class VoiceResponse(BaseModel):
    audio_url: str
    text: str
    duration: Optional[float] = None

class VoiceInterface:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_enabled = self._check_tts_availability()
        self.speech_recognition_enabled = self._check_speech_recognition()
        
    def _check_tts_availability(self) -> bool:
        """Check if text-to-speech is available"""
        try:
            # Try importing pyttsx3 for cross-platform TTS
            import pyttsx3
            return True
        except ImportError:
            try:
                # Try system TTS on macOS
                import subprocess
                subprocess.run(['which', 'say'], check=True, capture_output=True)
                return True
            except:
                logger.warning("No TTS engine available. Text-to-speech will be disabled.")
                return False
    
    def _check_speech_recognition(self) -> bool:
        """Check if speech recognition is available"""
        try:
            # Basic check for microphone availability for live recording
            with sr.Microphone() as source:
                pass
            logger.info("Microphone available for live speech recognition")
            return True
        except:
            logger.warning("No microphone available for live recording, but file transcription will still work")
            # Still return True because we can transcribe uploaded audio files
            return True
    
    async def transcribe_audio(self, audio_file: UploadFile) -> str:
        """Transcribe audio file to text using speech recognition"""
        # Note: We allow file transcription even if live microphone is not available
        
        try:
            # Save uploaded file temporarily with original extension
            file_extension = ".webm"  # Default for browser recordings
            if audio_file.filename:
                file_extension = os.path.splitext(audio_file.filename)[1] or ".webm"
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_input_file:
                content = await audio_file.read()
                temp_input_file.write(content)
                temp_input_path = temp_input_file.name
            
            # Convert to WAV format for speech recognition
            wav_path = temp_input_path.replace(file_extension, ".wav")
            
            try:
                # Try to convert using pydub
                try:
                    from pydub import AudioSegment
                    
                    # Load audio file (supports WebM, MP3, etc.)
                    if file_extension.lower() == ".webm":
                        # WebM might need special handling
                        audio = AudioSegment.from_file(temp_input_path, format="webm")
                    else:
                        audio = AudioSegment.from_file(temp_input_path)
                    
                    # Convert to WAV format suitable for speech recognition
                    audio = audio.set_frame_rate(16000).set_channels(1)  # Mono, 16kHz
                    audio.export(wav_path, format="wav")
                    logger.info(f"Converted {file_extension} to WAV for transcription")
                    
                except Exception as conv_error:
                    logger.warning(f"Audio conversion failed: {conv_error}. Trying direct recognition...")
                    # If conversion fails, try using the original file
                    wav_path = temp_input_path
                
                # Use speech recognition to transcribe
                with sr.AudioFile(wav_path) as source:
                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = self.recognizer.record(source)
                
                # Try multiple recognition engines
                try:
                    # Try Google Speech Recognition (free tier)
                    text = self.recognizer.recognize_google(audio_data)
                    logger.info(f"Transcribed with Google: '{text}'")
                    return text
                except sr.RequestError as e:
                    logger.warning(f"Google Speech Recognition request failed: {e}")
                    # Fallback to offline recognition if available
                    try:
                        text = self.recognizer.recognize_sphinx(audio_data)
                        logger.info(f"Transcribed with Sphinx: '{text}'")
                        return text
                    except Exception as sphinx_error:
                        logger.warning(f"Sphinx recognition failed: {sphinx_error}")
                        raise Exception("No speech recognition service available")
                except sr.UnknownValueError:
                    raise Exception("Could not understand audio - please speak more clearly")
                    
            finally:
                # Clean up temporary files
                if os.path.exists(temp_input_path):
                    os.unlink(temp_input_path)
                if wav_path != temp_input_path and os.path.exists(wav_path):
                    os.unlink(wav_path)
                
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            raise Exception(f"Failed to transcribe audio: {e}")
    
    async def text_to_speech(self, request: VoiceRequest) -> str:
        """Convert text to speech and return audio file path"""
        if not self.tts_enabled:
            raise Exception("Text-to-speech not available")
        
        try:
            # Create temporary file for audio output
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.close()
            audio_path = temp_file.name
            
            # Try different TTS engines
            success = False
            
            # Try system TTS (macOS) first for better web compatibility
            try:
                import subprocess
                cmd = ['say', '-o', audio_path, '--data-format=LEI16@22050']
                
                # Add voice if specified
                if request.voice != "default":
                    cmd.extend(['-v', request.voice])
                
                # Add speech rate if specified
                if request.speed != 1.0:
                    cmd.extend(['-r', str(int(200 * request.speed))])
                
                # Add text to speak
                cmd.append(request.text)
                
                subprocess.run(cmd, check=True)
                success = True
                logger.info(f"Generated TTS with macOS say: {audio_path}")
                
            except Exception as e:
                logger.warning(f"macOS say failed: {e}")
                
            # Fallback to pyttsx3 (cross-platform)
            if not success:
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    
                    # Configure voice settings
                    voices = engine.getProperty('voices')
                    if voices and request.voice != "default":
                        # Try to find requested voice
                        for voice in voices:
                            if request.voice.lower() in voice.name.lower():
                                engine.setProperty('voice', voice.id)
                                break
                    
                    # Set speech rate and volume
                    rate = engine.getProperty('rate')
                    engine.setProperty('rate', int(rate * request.speed))
                    
                    # Generate speech
                    engine.save_to_file(request.text, audio_path)
                    engine.runAndWait()
                    success = True
                    logger.info(f"Generated TTS with pyttsx3: {audio_path}")
                    
                except Exception as e:
                    logger.warning(f"pyttsx3 failed: {e}")
                
            
            if not success:
                raise Exception("No TTS engine succeeded")
            
            return audio_path
            
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            raise Exception(f"Failed to generate speech: {e}")
    
    async def get_available_voices(self) -> Dict[str, Any]:
        """Get list of available TTS voices"""
        voices = {"system_voices": [], "pyttsx3_voices": []}
        
        # Get pyttsx3 voices
        try:
            import pyttsx3
            engine = pyttsx3.init()
            pyttsx3_voices = engine.getProperty('voices')
            
            for voice in pyttsx3_voices or []:
                voices["pyttsx3_voices"].append({
                    "id": voice.id,
                    "name": voice.name,
                    "languages": getattr(voice, 'languages', []),
                    "gender": getattr(voice, 'gender', 'unknown')
                })
        except Exception as e:
            logger.warning(f"Could not get pyttsx3 voices: {e}")
        
        # Get system voices (macOS)
        try:
            import subprocess
            result = subprocess.run(['say', '-v', '?'], capture_output=True, text=True)
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        voice_name = parts[0]
                        language = parts[1] if len(parts) > 1 else "unknown"
                        voices["system_voices"].append({
                            "name": voice_name,
                            "language": language,
                            "description": ' '.join(parts[2:]) if len(parts) > 2 else ""
                        })
        except Exception as e:
            logger.warning(f"Could not get system voices: {e}")
        
        return voices
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get voice interface capabilities"""
        return {
            "speech_recognition": self.speech_recognition_enabled,
            "text_to_speech": self.tts_enabled,
            "voice_commands": self.speech_recognition_enabled,
            "audio_response": self.tts_enabled
        }

# Global voice interface instance
voice_interface = VoiceInterface()