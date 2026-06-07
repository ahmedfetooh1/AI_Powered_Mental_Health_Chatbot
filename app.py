import os
import json
import random
import torch
import re
import string
from typing import Literal, Optional
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from deep_translator import GoogleTranslator

# --- 1. الذاكرة (Memory) ---
chat_history = []

# --- Configuration ---
MODEL_PATH = "./final_model"
ID2LABEL = None

try:
    with open(os.path.join(MODEL_PATH, "labels.json"), "r", encoding="utf-8") as f:
        labels_map = json.load(f)
        ID2LABEL = {int(k): v for k, v in labels_map.items()}
    NewLabels = Literal[tuple(ID2LABEL.values())]
except FileNotFoundError:
    print("Warning: labels.json not found.")
    NewLabels = Literal["normal", "depression", "suicidal", "anxiety", "bipolar", "stress"]

SafetyLevel = Literal["safe", "flag", "crisis"]

# --- Helper Functions ---
def is_arabic(text: str) -> bool:
    return bool(re.search(r'[\u0600-\u06FF]', text))

def translate_text(text: str, target: str) -> str:
    try:
        return GoogleTranslator(source='auto', target=target).translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# --- Bot Data ---
NORMAL_RESPONSES = [
    "Thank you for sharing your thoughts with me. As your mental health assistant, I'm glad to see that your mental state seems stable right now.",
    "I appreciate you checking in! From what you've shared, things seem fairly balanced. I'm here to support you in maintaining this positive state.",
    "It's wonderful that you're expressing yourself. Your situation feels stable, and I encourage you to keep taking excellent care of yourself."
]

DEPRESSION_RESPONSES = [
   "I want you to know that I am here for you. It sounds like you might be going through a period of depression. Please be gentle with yourself.",
   "I'm so sorry you're feeling this heavy weight. Loss of interest and deep sadness are incredibly hard. Let's focus on small, supportive self-care steps."
]
DEPRESSION_ADVICE = [
    "1- Be kind to yourself and try to maintain a gentle, consistent daily routine.",
    "2- Focus on the basics: prioritize sleep and nourishing meals.",
    "3- If you can, engage in light physical activity like a short walk.",
    "4- Keep a journal to safely release your thoughts.",
    "5- Try reaching out to a trusted friend or family member, even just for a chat."
]

ANXIETY_RESPONSES = [
    "I hear how overwhelming this is for you. Anxiety is a heavy burden, but it is manageable. Let's try to gently organize your thoughts and find some calm together."
]
ANXIETY_ADVICE = [
    "1- Try practicing deep breathing or guided relaxation exercises.",
    "2- Channel the nervous energy into a daily walk or light exercise.",
    "3- Focus on the present moment—what is one thing you can gently control right now?"
]

SUICIDAL_RESPONSES = [
    "Please know that your life is incredibly valuable and I am here listening to you. The thoughts you're experiencing are serious, and you do not have to carry this alone."
]
SUICIDAL_ADVICE = [
    "1- Please reach out to a trusted mental health professional right away.",
    "2- Make sure you are not alone; contact someone you trust and tell them how you feel.",
    "3- Call or text your local emergency services or a crisis helpline immediately. You are worth saving."
]

BIPOLAR_RESPONSES = [
    "Thank you for being open. It sounds like you're experiencing some intense fluctuations in mood and energy. I am here to help you navigate this safely."
]
BIPOLAR_ADVICE = [
    "1- Focus heavily on getting consistent, quality sleep every night.",
    "2- Gently pace your daily tasks; avoid overloading yourself when your energy spikes.",
    "3- Keep track of your mood shifts and discuss them with a professional."
]

STRESS_RESPONSES = [
    "I can tell that you are carrying an incredible amount of stress right now. As your assistant, I want to remind you that it's okay to put your well-being first."
]
STRESS_ADVICE = [
    "1- Try to categorize your tasks and focus only on the absolute priorities today.",
    "2- Give yourself permission to take short, mindful breaks.",
    "3- Set firm, healthy boundaries between your responsibilities and your personal rest."
]

RESPONSES_MAP = {
    "normal": NORMAL_RESPONSES,
    "depression": DEPRESSION_RESPONSES,
    "anxiety": ANXIETY_RESPONSES,
    "suicidal": SUICIDAL_RESPONSES,
    "bipolar": BIPOLAR_RESPONSES,
    "stress": STRESS_RESPONSES
}

ADVICE_MAP = {
    "depression": DEPRESSION_ADVICE,
    "anxiety": ANXIETY_ADVICE,
    "suicidal": SUICIDAL_ADVICE,
    "bipolar": BIPOLAR_ADVICE,
    "stress": STRESS_ADVICE,
    "normal": []
}

# --- Load Model & Tokenizer ---
model = None
tok = None
try:
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    if ID2LABEL:
        model.config.id2label = ID2LABEL
except Exception as e:
    print(f"Error loading model: {e}")

def detect_safety(text: str) -> SafetyLevel:
    crisis_keywords = ["kill myself", "suicide", "انتحار", "اقتل نفسي", "انهي حياتي"]
    for kw in crisis_keywords:
        if kw in text.lower():
            return "crisis"
    return "safe"

# --- API Configuration ---
app = Flask(__name__)
CORS(app)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User Database Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    is_user = db.Column(db.Boolean, default=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('messages', lazy=True))

with app.app_context():
    db.create_all()

# --- 3. Endpoints ---
@app.route("/", endpoint="index")
def index():
    return render_template("chat.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        
        user = User.query.filter_by(email=email).first()
        if user and user.password == password:
            return redirect(url_for("index"))
        else:
            return render_template("auth.html", error="Invalid email or password.")
            
    return render_template("auth.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        
        if password != confirm_password:
            return render_template("auth.html", error="Passwords do not match.")
            
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template("auth.html", error="Email already exists.")
            
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for("index"))
        
    return render_template("auth.html")

@app.route("/messages", methods=["GET"])
def get_messages():
    username = request.args.get('username')
    if not username:
        return jsonify([])
    user = User.query.filter((User.username == username) | (User.email.startswith(username))).first()
    if not user:
        return jsonify([])
    
    results = [{"text": m.text, "is_user": m.is_user} for m in user.messages]
    return jsonify(results)

@app.route("/classify", methods=["POST"])
def classify_text():
    data = request.get_json()
    raw_input = data.get("text", "").strip() if data else ""
    username = data.get("username", "guest")
    if not raw_input:
        return jsonify({"label": "normal", "score": 0.0, "safety": "safe", "message": "Empty input", "response": "Please say something."})

    user = User.query.filter((User.username == username) | (User.email.startswith(username))).first()

    if user:
        user_msg = Message(text=raw_input, is_user=True, user_id=user.id)
        db.session.add(user_msg)
        db.session.commit()

    user_is_ar = is_arabic(raw_input)
    
    # ا- نظام الذاكرة: جلب السياق القديم
    context = ""
    if chat_history:
        context = "Previous context: " + chat_history[-1] + ". "
    
    # 1. تحويل النص للإنجليزي للمعالجة (سواء كان عربي أو إنجليزي)
    english_text = translate_text(raw_input, 'en') if user_is_ar else raw_input
    
    # دمج السياق مع المدخلات الحالية للموديل
    full_text_for_model = context + english_text
    clean_text = english_text.lower().translate(str.maketrans('', '', string.punctuation))
    clean_raw = raw_input.lower().translate(str.maketrans('', '', string.punctuation))

    # --- Gibberish Filter ---
    has_letters = bool(re.search(r'[a-zA-Z\u0600-\u06FF]', raw_input))
    is_repeated = bool(re.search(r'(.)\1{4,}', raw_input))
    
    is_gibberish = False
    if not has_letters or is_repeated or len(clean_text.replace(" ", "")) < 2:
        is_gibberish = True

    # --- فحص الترحيب ---
    GREETINGS_EN = ["hi", "hello", "hey", "greetings", "good morning", "good evening", "howdy"]
    GREETINGS_AR = ["مرحبا", "مرحباً", "اهلا", "أهلا", "السلام عليكم", "هلا", "صباح الخير", "مساء الخير"]
    
    is_greeting = False
    if any(re.search(rf"\b{w}\b", clean_text) for w in GREETINGS_EN) or any(w in clean_raw for w in GREETINGS_AR):
        is_greeting = True

    if is_greeting:
        res = f"Hello {username}! I am your dedicated Mental Health Assistant. I'm here to gently listen and support you in a safe space. How are you feeling today?"
        final_response = translate_text(res, 'ar') if user_is_ar else res
        if user:
            bot_msg = Message(text=final_response, is_user=False, user_id=user.id)
            db.session.add(bot_msg)
            db.session.commit()
        return jsonify({
            "label": "normal", "score": 1.0, "safety": "safe", "message": "Greeting", 
            "response": final_response
        })

    if is_gibberish and not is_greeting:
        gib_res = "I'm sorry, I didn't quite understand that. Could you please clarify or share your thoughts a bit more clearly? I'm here to carefully listen."
        final_response = translate_text(gib_res, 'ar') if user_is_ar else gib_res
        if user:
            bot_msg = Message(text=final_response, is_user=False, user_id=user.id)
            db.session.add(bot_msg)
            db.session.commit()
        return jsonify({
            "label": "normal", "score": 0.0, "safety": "safe", "message": "Gibberish", 
            "response": final_response
        })
    # --- المعالجة بالموديل ---
    if model is None:
        return jsonify({"label": "normal", "score": 0.0, "safety": "safe", "message": "Model error", "response": "Model unavailable."})

    inputs = tok(full_text_for_model, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        score, pred = torch.max(probs, dim=1)

    label = model.config.id2label.get(int(pred), "normal").lower()
    safety = detect_safety(english_text)

    # --- بناء الرد الكامل ---
    base_reply = random.choice(RESPONSES_MAP.get(label, NORMAL_RESPONSES))
    advice_list = ADVICE_MAP.get(label, [])
    
    full_response_en = f"Your condition is <b>{label}</b>.<br>{base_reply}"
    if advice_list:
        advice_html = "<br>".join(advice_list)
        full_response_en += f"<br><br><b>Suggested Advice:</b><br>{advice_html}"

    # ب- حفظ المدخل الحالي في الذاكرة
    chat_history.append(raw_input)
    if len(chat_history) > 5:
        chat_history.pop(0) # حفظ آخر 5 رسائل

    # --- 2. الرد بنفس لغة المستخدم ---
    if user_is_ar:
        final_response = translate_text(full_response_en, 'ar')
        final_msg = "هذا ليس بديلاً عن الرعاية الطبية المهنية."
        if safety == "crisis": final_msg = "إذا كنت في خطر، اتصل بالطوارئ فوراً."
    else:
        final_response = full_response_en
        final_msg = "This is not a substitute for medical care."
        if safety == "crisis": final_msg = "Please contact emergency services immediately."

    if user:
        bot_msg = Message(text=final_response, is_user=False, user_id=user.id)
        db.session.add(bot_msg)
        db.session.commit()

    return jsonify({
        "label": label,
        "score": float(score),
        "safety": safety,
        "message": final_msg,
        "response": final_response
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)