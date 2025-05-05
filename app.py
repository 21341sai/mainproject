from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import time
from werkzeug.utils import secure_filename
import os


from morse_utils import detect_beeps_with_gaps, morse_to_text, classify_morse

app = Flask(__name__)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()
face_mesh = mp_face_mesh.FaceMesh()

# Global variables
morse_code = ''
hand_present = False
blink_start_time = None
decoded_text = ""

# Constants
BLINK_THRESHOLD = 0.2
DOT_TIME = 0.2
DASH_TIME = 0.5

MORSE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..',
    '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
    '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----',
    ' ': '/'
}
MORSE_REVERSE_DICT = {v: k for k, v in MORSE_DICT.items()}

def classify_hand_state(hand_landmarks):
    """Classify hand state based on finger positions."""
    tip_ids = [4, 8, 12, 16, 20]
    folded_fingers = sum(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[tip - 2].y for tip in tip_ids[1:])
    return "fist" if folded_fingers >= 3 else "palm"

def calculate_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio (EAR) for blink detection."""
    left_x = landmarks[eye_indices[0]].x
    right_x = landmarks[eye_indices[1]].x
    top_y = (landmarks[eye_indices[2]].y + landmarks[eye_indices[3]].y) / 2
    bottom_y = (landmarks[eye_indices[4]].y + landmarks[eye_indices[5]].y) / 2
    return (bottom_y - top_y) / (right_x - left_x)

def generate_frames(mode):
    """Generate live video frames with hand or eye tracking for Morse detection."""
    global morse_code, hand_present, blink_start_time
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if mode == "hand":
            hand_results = hands.process(rgb_frame)
            gesture = None
            current_hand_present = False
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    gesture = classify_hand_state(hand_landmarks)
                    current_hand_present = True
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if not hand_present and current_hand_present:
                morse_code += "." if gesture == "fist" else "-" if gesture == "palm" else ""
            hand_present = current_hand_present
        elif mode == "eye":
            face_results = face_mesh.process(rgb_frame)
            current_time = time.time()
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    right_eye_indices = [33, 133, 160, 158, 153, 144]
                    left_eye_indices = [362, 263, 387, 385, 380, 373]
                    ear = (calculate_ear(face_landmarks.landmark, right_eye_indices) +
                           calculate_ear(face_landmarks.landmark, left_eye_indices)) / 2
                    if ear < BLINK_THRESHOLD:
                        if blink_start_time is None:
                            blink_start_time = current_time
                    else:
                        if blink_start_time is not None:
                            blink_duration = current_time - blink_start_time
                            blink_start_time = None
                            morse_code += "." if blink_duration < DOT_TIME else "-" if blink_duration < DASH_TIME else ""
        cv2.putText(frame, f"Morse: {morse_code}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/decode_morse', methods=['POST'])
def decode_morse_endpoint():
    global morse_code, decoded_text
    words = morse_code.strip().split("   ")
    decoded_words = []
    
    for word in words:
        characters = word.split(" ")
        decoded_word = ""
        for char in characters:
            decoded_word += MORSE_REVERSE_DICT.get(char, "?")
        decoded_words.append(decoded_word)
    
    decoded_text = " ".join(decoded_words)
    
    return jsonify({
        "decoded": decoded_text,
        "morse_code": morse_code 
    })

@app.route('/video_feed')
def video_feed():
    mode = request.args.get('mode', 'hand')
    return Response(generate_frames(mode), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video', methods=['POST'])
def stop_detection():
    return jsonify({"status": "Detection stopped", "morse_code": morse_code})

@app.route('/clear_morse', methods=['POST'])
def clear_morse():
    global morse_code, decoded_text
    morse_code, decoded_text = "", ""
    return '', 204

@app.route('/get_morse', methods=['GET'])
def get_morse():
    return jsonify({"morse_code": morse_code})

@app.route('/add_morse', methods=['POST'])
def add_morse():
    global morse_code
    data = request.json
    char = data.get('char')
    if char:
        morse_code += char
    return jsonify({"status": "success", "morse_code": morse_code})

@app.route('/add_space', methods=['POST'])
def add_space():
    global morse_code
    morse_code += " "
    return jsonify({"status": "success", "morse_code": morse_code})

@app.route('/add_word_space', methods=['POST'])
def add_word_space():
    global morse_code
    morse_code += "   "
    return jsonify({"status": "success", "morse_code": morse_code})

@app.route('/predict', methods=['POST'])
def predict_morse():
    global morse_code, decoded_text 
    
    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    morse_code = classify_morse(detect_beeps_with_gaps(filepath)) 
    decoded_text = morse_to_text(morse_code) 
    
    return jsonify({
        "morse_code": morse_code,
        "decoded_text": decoded_text
    })

if __name__ == "__main__":
    app.run(debug=True)