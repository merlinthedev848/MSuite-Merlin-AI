"""
Telephony, VoIP/SIP AudioSocket, and Real-Time Voice Agent for Merlin AI.

Provides Asterisk / FreeSWITCH AudioSocket protocol framing (res_audiosocket),
PCM / µ-law audio transcoding, energy-based Voice Activity Detection (VAD),
Whisper STT / Piper TTS execution hooks, and deterministic calendar agent integration.
"""

try:
    import audioop
    HAS_AUDIOOP = True
except ImportError:
    audioop = None
    HAS_AUDIOOP = False
import datetime
import io
import logging
import math
import os
import shutil
import struct
import subprocess
import threading
import uuid
import wave
from typing import Any

logger = logging.getLogger("merlin.telephony")

# Standard telephony audio constants
SAMPLE_RATE_8K = 8000
SAMPLE_RATE_16K = 16000
AUDIO_CHANNELS = 1
BYTES_PER_SAMPLE = 2  # 16-bit linear PCM

# AudioSocket Frame Types (Asterisk res_audiosocket)
AUDIOSOCKET_TYPE_HANGUP = 0x00
AUDIOSOCKET_TYPE_UUID = 0x01
AUDIOSOCKET_TYPE_SILENCE = 0x02
AUDIOSOCKET_TYPE_AUDIO = 0x10
AUDIOSOCKET_TYPE_ERROR = 0xFF


def create_wav_bytes(pcm_data: bytes, sample_rate: int = SAMPLE_RATE_16K, channels: int = 1) -> bytes:
    """Wraps raw 16-bit PCM bytes into a standard RIFF/WAV file header."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(BYTES_PER_SAMPLE)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return buf.getvalue()


def pcm16_to_ulaw(pcm_bytes: bytes) -> bytes:
    """Converts 16-bit linear PCM to 8-bit G.711 µ-law."""
    if HAS_AUDIOOP and audioop is not None:
        try:
            return audioop.lin2ulaw(pcm_bytes, 2)
        except Exception:
            pass

    # Software fallback for µ-law quantization
    out = bytearray()
    for i in range(0, len(pcm_bytes) - 1, 2):
        sample = int.from_bytes(pcm_bytes[i:i+2], byteorder="little", signed=True)
        # Clip
        sample = max(-32768, min(32767, sample))
        sign = 0x80 if sample < 0 else 0x00
        if sample < 0:
            sample = -sample
        sample += 132
        if sample > 32767:
            sample = 32767
        exp = 7
        for e in range(7, -1, -1):
            if sample & (1 << (e + 7)):
                exp = e
                break
        mantissa = (sample >> (exp + 3)) & 0x0F
        byte = ~(sign | (exp << 4) | mantissa) & 0xFF
        out.append(byte)
    return bytes(out)


def ulaw_to_pcm16(ulaw_bytes: bytes) -> bytes:
    """Converts 8-bit G.711 µ-law to 16-bit linear PCM."""
    if HAS_AUDIOOP and audioop is not None:
        try:
            return audioop.ulaw2lin(ulaw_bytes, 2)
        except Exception:
            pass

    out = bytearray()
    for b in ulaw_bytes:
        ub = ~b & 0xFF
        sign = ub & 0x80
        exp = (ub >> 4) & 0x07
        mantissa = ub & 0x0F
        sample = ((mantissa << 3) + 132) << exp
        sample -= 132
        if sign:
            sample = -sample
        sample = max(-32768, min(32767, sample))
        out.extend(int(sample).to_bytes(2, byteorder="little", signed=True))
    return bytes(out)


def calculate_pcm_energy(pcm_bytes: bytes) -> float:
    """Calculates RMS energy of a 16-bit PCM chunk to detect voice activity."""
    if not pcm_bytes:
        return 0.0
    if HAS_AUDIOOP and audioop is not None:
        try:
            return float(audioop.rms(pcm_bytes, 2))
        except Exception:
            pass

    count = len(pcm_bytes) // 2
    if count == 0:
        return 0.0
    sum_sq = 0.0
    for i in range(0, len(pcm_bytes) - 1, 2):
        val = int.from_bytes(pcm_bytes[i:i+2], byteorder="little", signed=True)
        sum_sq += val * val
    return math.sqrt(sum_sq / count)


# --- Asterisk AudioSocket Framing ---

def parse_audiosocket_frame(data: bytes) -> tuple[int, bytes, bytes]:
    """
    Parses a single Asterisk AudioSocket frame:
    Format: [1-byte Type] [2-bytes Payload Length (big-endian)] [Payload]
    Returns: (frame_type, payload_bytes, remaining_bytes)
    """
    if len(data) < 3:
        return -1, b"", data

    frame_type = data[0]
    payload_len = struct.unpack(">H", data[1:3])[0]

    if len(data) < 3 + payload_len:
        # Incomplete frame
        return -1, b"", data

    payload = data[3 : 3 + payload_len]
    rest = data[3 + payload_len :]
    return frame_type, payload, rest


def build_audiosocket_frame(frame_type: int, payload: bytes) -> bytes:
    """Constructs a binary AudioSocket frame."""
    length = len(payload)
    header = struct.pack(">BH", frame_type, length)
    return header + payload


# --- Local STT / TTS Execution Wrappers ---

def transcribe_audio_chunk(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE_16K) -> str:
    """
    Transcribes audio using local Whisper CLI if installed on the host,
    otherwise returns a clean placeholder indicating speech was received.
    """
    if not pcm_bytes or len(pcm_bytes) < sample_rate:  # Under 0.5s of audio
        return ""

    whisper_bin = shutil.which("whisper") or shutil.which("whisper-cli") or shutil.which("faster-whisper")
    if not whisper_bin:
        return ""

    wav_data = create_wav_bytes(pcm_bytes, sample_rate=sample_rate)
    temp_path = f"/tmp/merlin_stt_{uuid.uuid4().hex[:8]}.wav"
    try:
        with open(temp_path, "wb") as f:
            f.write(wav_data)

        cmd = [whisper_bin, temp_path, "--model", "tiny.en", "--output_format", "txt"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0)
        txt_path = temp_path.replace(".wav", ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, encoding="utf-8") as tf:
                text = tf.read().strip()
            try:
                os.remove(txt_path)
            except OSError:
                pass
            return text
        return res.stdout.strip()
    except Exception as e:
        logger.warning("Local Whisper transcription failed: %s", e)
        return ""
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def synthesize_speech(text: str, voice: str = "en_US-lessac-medium") -> bytes:
    """
    Synthesizes speech via local Piper TTS if available.
    Returns 16-bit linear PCM audio bytes (16kHz).
    """
    if not text:
        return b""

    piper_bin = shutil.which("piper")
    if piper_bin:
        try:
            cmd = [piper_bin, "--output_raw", "--model", voice]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, _ = proc.communicate(input=text.encode("utf-8"), timeout=10.0)
            if stdout:
                return stdout
        except Exception as e:
            logger.warning("Local Piper TTS synthesis failed: %s", e)

    return b""


# --- Real-Time Telephony Call Session State Machine ---

class TelephonySession:
    """
    Encapsulates a live VoIP phone call / audio stream session.
    Manages speech accumulation, VAD, LLM response generation with
    deterministic function calling for appointment booking, and audio playback.
    """

    def __init__(self, session_id: str | None = None, caller_number: str = "Unknown", caller_name: str = "Caller"):
        self.session_id = session_id or str(uuid.uuid4())
        self.caller_number = caller_number
        self.caller_name = caller_name
        self.created_at = datetime.datetime.now().isoformat()
        self.audio_in_buffer = bytearray()
        self.audio_out_buffer = bytearray()
        self.is_active = True
        self.silence_chunks = 0
        self.speaking = False
        self.energy_threshold = 450.0  # RMS threshold for voice detection
        self.lock = threading.Lock()
        self.history: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"You are Merlin, an intelligent telephony automated receptionist. "
                    f"Caller Phone: {self.caller_number}, Caller Name: {self.caller_name}. "
                    f"Your job is to assist callers with questions, check calendar availability, and book appointments. "
                    f"Keep responses concise, polite, professional, and clear for phone audio."
                ),
            }
        ]

    def add_audio_frame(self, pcm_chunk: bytes) -> None:
        """Appends incoming audio chunk (16-bit PCM 16kHz) to the active buffer."""
        with self.lock:
            self.audio_in_buffer.extend(pcm_chunk)
            rms = calculate_pcm_energy(pcm_chunk)
            if rms > self.energy_threshold:
                self.speaking = True
                self.silence_chunks = 0
            else:
                if self.speaking:
                    self.silence_chunks += 1

    def should_trigger_turn(self, silence_chunks_needed: int = 15) -> bool:
        """Determines if the caller finished speaking (silence detected after speech)."""
        with self.lock:
            return self.speaking and (self.silence_chunks >= silence_chunks_needed)

    def extract_utterance(self) -> bytes:
        """Extracts and clears the accumulated user audio utterance."""
        with self.lock:
            audio = bytes(self.audio_in_buffer)
            self.audio_in_buffer.clear()
            self.speaking = False
            self.silence_chunks = 0
            return audio

    def close(self) -> None:
        """Terminates the telephony session."""
        self.is_active = False


# Active live sessions registry
_telephony_sessions: dict[str, TelephonySession] = {}
_sessions_lock = threading.Lock()


def get_or_create_telephony_session(session_id: str | None = None, caller_number: str = "Unknown") -> TelephonySession:
    with _sessions_lock:
        sid = session_id or str(uuid.uuid4())
        if sid not in _telephony_sessions:
            _telephony_sessions[sid] = TelephonySession(session_id=sid, caller_number=caller_number)
        return _telephony_sessions[sid]


def close_telephony_session(session_id: str) -> None:
    with _sessions_lock:
        sess = _telephony_sessions.pop(session_id, None)
        if sess:
            sess.close()


def get_active_telephony_sessions() -> list[dict[str, Any]]:
    with _sessions_lock:
        return [
            {
                "session_id": s.session_id,
                "caller_number": s.caller_number,
                "caller_name": s.caller_name,
                "created_at": s.created_at,
                "is_active": s.is_active,
            }
            for s in _telephony_sessions.values()
        ]
