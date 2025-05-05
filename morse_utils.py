from werkzeug.utils import secure_filename
import librosa
import numpy as np





MORSE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..',  
    '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
    '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----',
    ',': '--..--', '.': '.-.-.-', '?': '..--..', "'": '.----.',
    '!': '-.-.--', '/': '-..-.', '(': '-.--.', ')': '-.--.-',
    '&': '.-...', ':': '---...', ';': '-.-.-.', '=': '-...-',
    '+': '.-.-.', '-': '-....-', '_': '..--.-', '"': '.-..-.',
    '$': '...-..-', '@': '.--.-.', ' ': '/'
}

# Reverse dictionary to decode Morse back to text
MORSE_REVERSE_DICT = {v: k for k, v in MORSE_DICT.items()}

def detect_beeps_with_gaps(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    hop_length = 512
    energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = np.arange(len(energy)) * (hop_length / sr)
    threshold = np.mean(energy) * 1.2

    beeps = []
    start_time = None
    is_sound = False

    for i in range(len(energy)):
        if energy[i] > threshold and not is_sound:
            start_time = times[i]
            is_sound = True
        elif energy[i] <= threshold and is_sound:
            end_time = times[i]
            duration = end_time - start_time
            beeps.append((start_time, end_time, duration))
            is_sound = False
    
    return beeps

SHORT_BEEP = 0.15
LONG_BEEP = 0.3
SHORT_GAP = 0.3
LONG_GAP = 0.9

def classify_morse(beeps):
    morse_sequence = []
    for i, (start, end, duration) in enumerate(beeps):
        if duration < SHORT_BEEP:
            morse_sequence.append(".")
        elif duration > LONG_BEEP:
            morse_sequence.append("-")
        if i < len(beeps) - 1:
            next_start = beeps[i + 1][0]
            gap = next_start - end
            if gap > LONG_GAP:
                morse_sequence.append("   ")
            elif gap > SHORT_GAP:
                morse_sequence.append(" ")
    return "".join(morse_sequence)

def morse_to_text(morse_code):
    words = morse_code.split("   ")
    decoded_text = []
    for word in words:
        letters = word.split(" ")
        decoded_word = "".join([MORSE_REVERSE_DICT.get(letter, "?") for letter in letters])
        decoded_text.append(decoded_word)
    return " ".join(decoded_text)
