import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # استيراد دالة التقسيم

# قائمة التصنيفات التي تريد استخدامها (لتكون هي القيمة في عمود 'label' في ملف train.jsonl)
# لاحظ أننا سنقوم بتحويل أسماء التصنيفات في البيانات إلى حروف صغيرة للمقارنة
TARGET_LABELS = ['normal', 'depression', 'suicidal', 'anxiety', 'bipolar', 'stress', 'personality disorder']

all_data = [] # تم تغيير الاسم ليحتوي على كل البيانات قبل التقسيم

# 1. قراءة البيانات من ملف CSV
csv_file_path = 'dataset.csv'

try:
    df = pd.read_csv(csv_file_path, encoding='utf-8')
except FileNotFoundError:
    print(f"خطأ: ملف '{csv_file_path}' غير موجود.")
    exit()

# 2. معالجة وتصنيف كل صف في DataFrame

for index, row in df.iterrows():
    pattern = str(row['statement']).strip()
    tag = str(row['status']).strip().lower().replace(" ", "_")
    
    if not pattern or pd.isna(row['statement']):
        continue
    
    label = 'normal' 
    
    # البحث عن تطابق مباشر
    if tag in TARGET_LABELS:
        label = tag
    
    # === منطق إضافي للتعامل مع التسميات غير الواضحة ===
    elif 'suicide' in tag:
        label = 'suicidal'
    elif 'depress' in tag:
        label = 'depression'
    elif 'anxiet' in tag:
        label = 'anxiety'
    elif 'bipolar' in tag:
        label = 'bipolar'
    elif 'stress' in tag:
        label = 'stress'
    elif 'normal' in tag:
        label = 'normal'
    
    # إضافة النمط المصنف إلى قائمة البيانات الكاملة
    all_data.append({
        'text': pattern,
        'label': label
    })

# 3. تقسيم البيانات (Data Splitting)
# تقسيم البيانات إلى مجموعتي تدريب (70%) وتقييم (30%)
# stratify=y يضمن أن نسبة التصنيفات في مجموعتي train و eval هي نفسها الموجودة في البيانات الأصلية.

if not all_data:
    print("لم يتم العثور على أي بيانات صالحة للمعالجة.")
    exit()

# استخراج التصنيفات (Labels) لـ stratify
labels = [item['label'] for item in all_data]

# التقسيم
train_data, eval_data = train_test_split(
    all_data, 
    test_size=0.3,         # 30% للتقييم
    random_state=42,       # لتكرار نفس التقسيم في كل مرة
    stratify=labels        # لضمان التوزيع المتساوي للتصنيفات
)

# 4. كتابة ملف التدريب بصيغة JSONL
with open('train.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 5. كتابة ملف التقييم بصيغة JSONL
with open('eval.jsonl', 'w', encoding='utf-8') as f:
    for item in eval_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"✅ تم بنجاح تحويل ومعالجة البيانات.")
print(f"  - حجم مجموعة التدريب (70%): {len(train_data)} مثال (تم حفظها في train.jsonl)")
print(f"  - حجم مجموعة التقييم (30%): {len(eval_data)} مثال (تم حفظها في eval.jsonl)")
print(f"تم التقسيم بناءً على القائمة: {TARGET_LABELS}")