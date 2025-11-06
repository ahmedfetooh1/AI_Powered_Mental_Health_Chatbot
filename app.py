import re
from typing import Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import random

SafetyLevel = Literal["safe", "flag", "crisis"]

class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    label: Literal["normal","concern"]
    score: float
    safety: SafetyLevel
    message: str
    response: str  # إضافة رد الروبوت

app = FastAPI(title="Mental Health Chatbot API", version="0.1.0")

# إضافة CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# إضافة endpoint للصفحة الرئيسية
@app.get("/")
def read_root():
    return {"message": "Mental Health Chatbot API is running!"}

# ردود الروبوت حسب التصنيف
NORMAL_RESPONSES = [
    "أفهم ما تشعر به. شكراً لمشاركتك معي.",
    "أقدر ثقتك في التحدث معي. كيف يمكنني مساعدتك؟",
    "شكراً لك على مشاركة مشاعرك. أنا هنا لمساعدتك.",
    "أقدر صراحتك. هل تريد التحدث عن شيء محدد؟",
    "شكراً لك على الثقة. كيف يمكنني دعمك اليوم؟"
]

CONCERN_RESPONSES = [
    "أفهم أنك تمر بوقت صعب. أنا هنا لمساعدتك.",
    "أقدر أنك تشاركني مشاعرك. هذا يتطلب شجاعة.",
    "أفهم ما تشعر به. دعنا نتحدث عن هذا أكثر.",
    "أقدر ثقتك في التحدث معي. كيف يمكنني مساعدتك؟",
    "أفهم أنك تشعر بالضيق. أنا هنا لمساعدتك."
]

CRISIS_RESPONSES = [
    "أفهم أنك تمر بوقت صعب جداً. من المهم أن تطلب مساعدة فورية.",
    "أقدر أنك تشاركني هذا. من المهم أن تتحدث مع مختص فوراً.",
    "أفهم أنك تشعر باليأس. من المهم أن تطلب مساعدة فورية.",
    "أقدر ثقتك في التحدث معي. من المهم أن تتحدث مع مختص فوراً.",
    "أفهم أنك تمر بوقت صعب جداً. من المهم أن تطلب مساعدة فورية."
]

# تحميل النموذج والـ tokenizer
try:
    tok = AutoTokenizer.from_pretrained("model")
    model = AutoModelForSequenceClassification.from_pretrained("model")
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    tok = None
    model = None

CRISIS_PATTERNS = [
    r"\b(?:انتحار|أنتحر|أؤذي نفسي|أقتل نفسي)\b",
    r"\b(?:قتل|أقتل|أؤذي الآخرين)\b"
]
CRISIS_RE = re.compile("|".join(CRISIS_PATTERNS), re.IGNORECASE)

def detect_safety(text: str) -> SafetyLevel:
    if CRISIS_RE.search(text):
        return "crisis"
    if any(w in text for w in ["حزين جدًا","أشعر بيأس","لا فائدة"]):
        return "flag"
    return "safe"

def softmax(logits: torch.Tensor) -> torch.Tensor:
    e = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    return e / e.sum(dim=-1, keepdim=True)

def get_bot_response(label: str, safety: SafetyLevel) -> str:
    if safety == "crisis":
        return random.choice(CRISIS_RESPONSES)
    elif label == "concern":
        return random.choice(CONCERN_RESPONSES)
    else:
        return random.choice(NORMAL_RESPONSES)

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if model is None or tok is None:
        return ClassifyResponse(
            label="normal", 
            score=0.5, 
            safety="safe", 
            message="النموذج غير متاح حالياً",
            response="عذراً، النموذج غير متاح حالياً."
        )
    
    safety = detect_safety(req.text)
    with torch.no_grad():
        tokens = tok(req.text, return_tensors="pt", truncation=True)
        logits = model(**tokens).logits
        probs = softmax(logits)[0]
        score, pred = torch.max(probs, dim=-1)
    label = model.config.id2label[int(pred)]
    
    # الحصول على رد الروبوت حسب التصنيف
    bot_response = get_bot_response(label, safety)
    
    msg = "هذا ليس بديلاً عن الرعاية الطبية. اطلب مساعدة مختصة عند الحاجة."
    if safety == "crisis":
        msg = "إذا كنت في خطر فوري، تواصل فوراً مع خدمات الطوارئ أو خط المساندة المحلي."
    elif safety == "flag":
        msg = "قد يفيدك التحدث مع مختص. هل ترغب بإرشادات موارد دعم محلية؟"
    
    return ClassifyResponse(
        label=label, 
        score=float(score), 
        safety=safety, 
        message=msg,
        response=bot_response
    )