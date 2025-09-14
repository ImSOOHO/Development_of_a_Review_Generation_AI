# 기본 세팅
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import pandas as pd
import torch
import random
import sentence_transformers
from transformers import pipeline
from transformers import GPT2LMHeadModel
from konlpy.tag import Okt
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from kss import split_sentences
import random

okt = Okt()

# 1. 기존 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "KingKDB/koGPTV3_contextbasedV4"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# 2. 데이터 로드(사용 DATA: 전처리V5(음식대분류추가))
df = pd.read_csv("C:/Users/DMQA/OneDrive/바탕 화면/노트북 파일 모음/포트폴리오/nlp 프젝/데이터/전처리V5(음식대분류추가).csv")
df_filtered = df[['store_name', 'store_genre', 'store_category', 'content_clean', 'sentiment']]

# 3. 샘플링 비율 계산 및 샘플링
genre_counts = df_filtered['store_genre'].value_counts()
total_samples = 3000
genre_ratios = (genre_counts / genre_counts.sum()) * total_samples
genre_sample_counts = genre_ratios.round().astype(int)
diff = total_samples - genre_sample_counts.sum()
if diff != 0:
    genre_sample_counts[genre_sample_counts.idxmax()] += diff

sampled_list = []
for genre, count in genre_sample_counts.items():
    genre_df = df_filtered[df_filtered['store_genre'] == genre]
    sampled_df = genre_df.sample(n=count, random_state=42)
    sampled_list.append(sampled_df)

df_sampled = pd.concat(sampled_list, ignore_index=True)

# 4. 학습용 텍스트 생성
sentiment_map = {2: "긍정", 0: "부정"}
df_sampled['sentiment_label'] = df_sampled['sentiment'].map(sentiment_map)

# 텍스트 구성 예: "[긍정] 고기마을 한식 고기 리뷰: 정말 맛있었어요!" -> 감성 토큰 형태로 학습
df_sampled['text'] = df_sampled.apply(
    lambda row: f"[{row['sentiment_label']}] {row['store_name']} {row['store_genre']} {row['store_category']} 리뷰: {row['content_clean']}",
    axis=1
)

# 5. Hugging Face Dataset 변환 및 토크나이징
train_dataset = Dataset.from_pandas(df_sampled[['text']])

def tokenize_function(examples):
    return tokenizer(
        examples["text"],                   
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # Trainer가 인식 못하는 컬럼 제거
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. 학습 설정 (GPU + fp16 사용)
training_args = TrainingArguments(
    output_dir="./koGPT_finetuned_v2",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=1,
    logging_steps=100,              # 로그 출력 간격 (100 step마다)
    logging_strategy="steps",       # 로그 출력 방식: step 단위로 지정
    #evaluation_strategy="no",       # 평가는 하지 않음
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    report_to="none"
)

# 7. Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 8. 2차 파인튜닝 실행
trainer.train()

# 9. 리뷰 생성
# 총 생성할 리뷰 수
total_reviews = 600
half_reviews = total_reviews // 2  # 긍정 300, 부정 300

# 음식점 장르별 비율에 따라 리뷰 개수 계산
genre_counts = df['store_genre'].value_counts(normalize=True)
genre_review_counts = (genre_counts * half_reviews).round().astype(int)

# 총합이 정확히 300이 안 될 수 있으므로 보정
diff = half_reviews - genre_review_counts.sum()
if diff != 0:
    genre_review_counts[genre_review_counts.idxmax()] += diff

generated_reviews = []

def clean_review(text):
    # 반복 문자 줄이기 (ex. 하하하하 → 하하)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)

    # 한글 + 영어 붙은 경우 분리
    text = re.sub(r'([가-힣])([A-Za-z])', r'\1 \2', text)

    # 문장 분리
    sentences = split_sentences(text)

    # 종결 표현 정의
    ending_patterns = (
        r'(감사합니다|맛있어요|맛있습니다|추천합니다|또 올게요|최고예요|잘 먹었습니다|'
        r'겠습니다|었습니다|았음|할께요|할게요|습니다|합니다|합니당|입니다|'
        r'잘먹었습니다|좋아요|좋아용|바라요|바래요|해주셨어요|드셨어요|다녀왔어요)(?=\s|$|[.!?])'
    )

    # 부자연스러운 문장 시작 표현 리스트
    unnatural_starts = ['요', '서', '고', '데', '그리고', '그래서', '하지만', '또한', '야', '할 일이 있어']

    # 출력 대상 문장 모으기
    collected = []
    for sentence in sentences:
        # 조사 등으로 시작하면 해당 표현만 제거
        for starter in unnatural_starts:
            if sentence.strip().startswith(starter):
                sentence = sentence.strip()[len(starter):].lstrip()
                break

        collected.append(sentence.strip())
        if re.search(ending_patterns, sentence):
            break  # 종결 표현 문장까지 수집 후 종료

    return ' '.join(collected).strip()

#text = "할 일이 있어 친절하게 처리해주셨어요. 정말 감사합니다. 또 올게요!"
#print(clean_review(text))


def generate_review(prompt, max_new_tokens=150, top_k=50, top_p=0.95, max_chars=None):
    # 프롬프트를 포함한 입력 토큰 생성
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    # max_new_tokens로 프롬프트 제외한 새 토큰 수 제한
    sample_output = model.generate(
        input_ids,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    # 생성된 텍스트에서 프롬프트 부분 제거
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    # max_chars가 지정된 경우에만 후처리
    if max_chars is not None and len(generated_text) > max_chars:
        truncated = generated_text[:max_chars]
        m = re.search(r'[.!?](?!.*[.!?])', truncated)
        if m:
            end = m.end()
            return truncated[:end].strip()
        else:
            return truncated.strip()
    return generated_text

for sentiment_label in ["긍정", "부정"]:
    for genre, count in genre_review_counts.items():
        df_genre = df[df['store_genre'] == genre]
        store_names = df_genre['store_name'].unique()
        if len(store_names) == 0:
            print(f"'{genre}' 장르에 해당하는 store_name 데이터가 없습니다.")
            continue

        for i in range(count):
            store_name = random.choice(store_names)
            category = df_genre[df_genre['store_name'] == store_name]['store_category'].values[0]
            prompt = f"[{sentiment_label}]{store_name} {genre} {category} 리뷰:"  # 변경된 프롬프트

            review_raw = generate_review(prompt, max_new_tokens=200)
            review = clean_review(review_raw)

            generated_reviews.append({
                "store_genre": genre,
                "store_name": store_name,
                "store_category": category,
                "target_sentiment": sentiment_label,
                "prompt": prompt,
                "generated_review": review
            })
            print(f"[{genre}][{store_name}][{sentiment_label}] 리뷰 생성 {i+1}/{count} 완료")


# 결과 저장
df_generated = pd.DataFrame(generated_reviews)
print("총 생성된 리뷰 개수:", len(df_generated))
print(df_generated['target_sentiment'].value_counts())

# 생성된 리뷰 저장
df_generated.to_csv("generated_reviews_600.csv_v6.csv", index=False, encoding="utf-8-sig")
