import json

# افتح ملف البيانات
with open('intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# قائمة النوايا المقلقة
concern_intents = ['sad', 'stressed', 'worthless', 'depressed', 'anxious', 'lonely', 'hopeless', 'overwhelmed']

# قائمة النوايا العادية  
normal_intents = ['greeting', 'morning', 'afternoon', 'evening', 'night', 'goodbye', 'thanks', 'about', 'skill', 'creation', 'name', 'help']

training_data = []

# معالجة كل نية
for intent in data['intents']:
    tag = intent['tag']
    patterns = intent['patterns']
    
    # تحديد التصنيف
    if tag in concern_intents:
        label = 'concern'
    elif tag in normal_intents:
        label = 'normal'
    else:
        label = 'normal'  # افتراضي
    
    # إضافة كل نمط
    for pattern in patterns:
        if pattern.strip():
            training_data.append({
                'text': pattern.strip(),
                'label': label
            })

# كتابة ملف التدريب
with open('train.jsonl', 'w', encoding='utf-8') as f:
    for item in training_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"تم تحويل {len(training_data)} مثال")