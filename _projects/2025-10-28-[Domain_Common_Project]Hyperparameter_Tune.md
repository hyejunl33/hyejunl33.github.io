# [모델최적화]하이퍼파라미터 튜닝

# Introduction

![image](/assets/images/2025-10-28-10-28-11.png)

이전모델에서 FocalLoss를 추가하고 Learning rate scheduler를 “cosine”으로 바꾸고, `label_smoothing`을 적용하고, `weight_decay`값을 키워봤다. regularizastion방법을 썼기떄문에, 에폭을 늘려서 학습해도 될것같아서, 에폭을 기존 5에서 8로 늘렸다.

![image](/assets/images/2025-10-28-10-28-25.png)

그 결과 TrainingLoss는 줄어들지면 ValidationLoss는 줄어들지 않는 과적합이 일어났다. 우선 `EarlystoppingPatience`를 3으로 설정했지만 기준을 Accuracy로 잡아서 validation Acc가 증가하는 이상 Early stopping은 적용되지 않았다. 따라서 이번에는 Early stopping기준을 Validation Loss로 잡을 예정이다. 이를 통해 Overfitting을 막을 생각이다.

그리고 하이퍼 파라미터 튜닝을 통해 최적의 `Weight_decay`값, `label_smoothing`값, 그리고 이전에는 적용하지 않았던 `drop_out` 비율까지 찾을 예정이다. 그 외에도 최적의 에폭 수, `Learningrate_scheduler_type` 등 다양한 하이퍼파라미터를 config에 넣어서 최적의 하이퍼 파라미터를 찾을 예정이다.

하이퍼파라미터 튜닝은 Evaluation/Loss를 기준으로 하기때문에, 최대한 Overfitting을 막는 방향으로 하이퍼파라미터 튜닝을 진행할 예정이다.

그리고 이후에  Back Translation을 통한 데이터 증강, 확률적 가중치 평균 (Stochastic Weight Averaging - SWA) 기법을 이용해서 조금더 일반화 성능을 늘려볼 예정이다. 

그리고 `kykim/bert-kor-base` 모델이 단일모델로서는 성능이 가장 잘 나온다고 한다. 이 모델을 이용해서 학습을 돌려볼 예정이다.

# 하이퍼파라미터 튜닝

```python

# === 훈련 함수 정의 ===
def train():
    # --- 1. W&B Run 초기화 ---
    os.environ["WANDB_SILENT"] = "true"
    run = wandb.init(project="klue-bert-sweep-movie-review")
    config = wandb.config

    # --- 2. 시드 설정 ---
    set_seed(RANDOM_STATE)

    # --- 3. 데이터 전처리 (train 함수 내부에서 실행) ---
    print(f"Run {run.name}: Preprocessor 생성 (희귀 단어 제거: {config.num_rare_words_to_remove}개)")
    # 각 run마다 config 값으로 preprocessor 생성
    preprocessor = TextPreprocessingPipeline(num_rare_words_to_remove=int(config.num_rare_words_to_remove))

    # 원본 훈련 데이터로 fit_transform 수행
    print(f"Run {run.name}: 훈련 데이터 전처리 (fit & transform)...")
    X_train_processed = preprocessor.fit_transform(X_train_orig.tolist(), y_train_orig.tolist())

    # 원본 검증 데이터로 transform 수행
    print(f"Run {run.name}: 검증 데이터 전처리 (transform)...")
    X_val_processed = preprocessor.transform(X_val_orig.tolist())

    # DataFrame 형태로 변환 (Dataset 클래스 입력용)
    train_data = pd.DataFrame(
        {"ID": ids_train_orig.values, "review": X_train_processed, "label": y_train_orig.values}
    )
    val_data = pd.DataFrame(
        {"ID": ids_val_orig.values, "review": X_val_processed, "label": y_val_orig.values}
    )
    # print(f"Run {run.name}: 데이터 전처리 완료. 훈련: {len(train_data)}, 검증: {len(val_data)}") # Sweep 중 출력 최소화

    # --- 4. 모델 및 토크나이저 로드 ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    num_added_toks = tokenizer.add_tokens(NEW_SPECIAL_TOKENS)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        hidden_dropout_prob=config.dropout_rate,
        attention_probs_dropout_prob=config.dropout_rate
    )
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- 5. 데이터셋 생성 ---
    train_dataset = ReviewDataset(
        train_data["review"], # 전처리된 텍스트 사용
        train_data["label"],
        tokenizer,
        CHOSEN_MAX_LENGTH,
    )
    val_dataset = ReviewDataset(
        val_data["review"], # 전처리된 텍스트 사용
        val_data["label"],
        tokenizer,
        CHOSEN_MAX_LENGTH,
    )

    # --- 6. TrainingArguments 설정 ---
    output_dir = os.path.join("./results", run.name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(config.num_train_epochs),
        per_device_train_batch_size=int(config.per_device_train_batch_size),
        per_device_eval_batch_size=128, # 평가 배치는 고정 또는 Sweep 파라미터로 추가
        warmup_steps=int(config.warmup_steps),
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        label_smoothing_factor=config.label_smoothing_factor,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", # 검증 손실 기준
        greater_is_better=False,        # 손실은 낮을수록 좋음
        save_total_limit=1,
        report_to="wandb",
        run_name=run.name,
        seed=RANDOM_STATE,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,      # DataLoader 워커 오류 방지 위해 0으로 설정
        remove_unused_columns=False,
        lr_scheduler_type=config.lr_scheduler_type, # Sweep 값 사용
        gradient_accumulation_steps=int(config.gradient_accumulation_steps), # Sweep 값 사용
        logging_first_step=True,
        disable_tqdm=True,             # 진행률 표시줄 비활성화
    )

    # --- 7. EarlyStopping 설정 ---
    # TrainingArguments에서 patience를 직접 설정할 수 없으므로 Callback 사용
    # Sweep config에서 patience 값을 가져오도록 수정
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3 # 고정값 사용 (또는 sweep_config에 추가하고 config.early_stopping_patience 사용)
        # early_stopping_threshold=0.001 # 필요시 설정
    )

    # --- 8. Trainer 초기화 ---
    trainer = CustomTrainerWithFocalLoss(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        num_classes=NUM_CLASSES,
        focal_loss_alpha=[0.11, 0.45, 0.12, 0.32], # 알파는 고정
        focal_loss_gamma=config.focal_loss_gamma, # Sweep 값 사용
        callbacks=[early_stopping_callback],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # --- 9. 훈련 실행 ---
    try:
        # print(f"Run {run.name}: 훈련 시작...") # Sweep 중 출력 최소화
        trainer.train()
        # print(f"Run {run.name}: 훈련 완료.") # Sweep 중 출력 최소화
    except Exception as e:
        print(f"Run {run.name}: 훈련 중 오류 발생: {e}")
        wandb.finish(exit_code=1)
        raise

    # --- 10. W&B Run 종료 ---
    wandb.finish()

# === Sweep 설정 ===
sweep_config = {
    'method': 'bayes',
    'metric': { 'name': 'eval/loss', 'goal': 'minimize' },
    'parameters': {
        'learning_rate': { 'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 5e-5 },
        'num_train_epochs': { 'values': [5, 8, 10, 12] },
        'weight_decay': { 'distribution': 'uniform', 'min': 0.0, 'max': 0.1 },
        'per_device_train_batch_size': { 'values': [64, 128, 256] },
        'warmup_steps': { 'distribution': 'q_uniform', 'min': 100, 'max': 1000, 'q': 50 },
        'focal_loss_gamma': { 'distribution': 'uniform', 'min': 1.0, 'max': 3.0 },
        'label_smoothing_factor': { 'distribution': 'uniform', 'min': 0.0, 'max': 0.2 },
        'dropout_rate': { 'distribution': 'uniform', 'min': 0.05, 'max': 0.3 },
        'lr_scheduler_type': { 'values': ['cosine', 'linear', 'constant'] },
        'gradient_accumulation_steps': { 'values': [1, 2, 4] },
        'num_rare_words_to_remove': { 'values': [0, 3, 5, 10] } # 추가됨
    }
}

# === W&B Sweep 초기화 및 에이전트 실행 ===
# wandb 로그인 필요
# !wandb login your_api_key

sweep_id = wandb.sweep(sweep_config, project="klue-bert-sweep-movie-review")
print(f"Sweep ID: {sweep_id}")

# 에이전트 실행 (20번 시도)
wandb.agent(sweep_id, function=train, count=20)

print("W&B Sweep 완료!")
```

```python
sweep_config = {
    'method': 'bayes',
    'metric': { 'name': 'eval/loss', 'goal': 'minimize' },
    'parameters': {
        'learning_rate': { 'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 5e-5 },
        'num_train_epochs': { 'values': [5, 8, 10, 12] },
        'weight_decay': { 'distribution': 'uniform', 'min': 0.0, 'max': 0.1 },
        'per_device_train_batch_size': { 'values': [64, 128, 256] },
        'warmup_steps': { 'distribution': 'q_uniform', 'min': 100, 'max': 1000, 'q': 50 },
        'focal_loss_gamma': { 'distribution': 'uniform', 'min': 1.0, 'max': 3.0 },
        'label_smoothing_factor': { 'distribution': 'uniform', 'min': 0.0, 'max': 0.2 },
        'dropout_rate': { 'distribution': 'uniform', 'min': 0.05, 'max': 0.3 },
        'lr_scheduler_type': { 'values': ['cosine', 'linear', 'constant'] },
        'gradient_accumulation_steps': { 'values': [1, 2, 4] },
        'num_rare_words_to_remove': { 'values': [0, 3, 5, 10] } # 추가됨
    }
}
```

`config` 를 보면 알 수 있듯이 베이지안 방식으로 learning rate, 에폭 수, weight_decay, 배치사이즈, warmup_step등등을 하이퍼파라미터로 넣어주었다.

이때 이번 실험에서의 목표가 과적합을 막는것이었으므로 Validation/loss를 metric으로 삼고, 이를 minimize하는것을 goal로 설정했다. 따라서 validation loss가 earlystopping patience를 넘는다면 바로 조기종료 될 것이다.

하이퍼파라미터 튜닝을 16번의 step을 거쳐서 진행했다. 원래는 20번으로 실행했는데, 28시간째에, 다른실험도 해봐야해서 일단 Interupt를 했다..

![image](/assets/images/2025-10-28-10-29-03.png)

eval/accuracy를 기준으로 상위 5개를 시각화했다. 보라색이 가장 acc가 높아보이지만,validation loss가 불안정하게 커지므로, 과적합되었다고 볼 수 있다. 따라서 그 다음으로 acc가 높은 ancient-sweep-13 하이퍼파라미터를 사용하기로 했다.

최적의 하이퍼파라미터는 모델의 아키텍처마다 다르다고 한다. 따라서 원래는 5개의 모델을 모두 앙상블할 예정이었지만, `klue/bert-base` 모델로 하이퍼파라미터 튜닝을 했으므로, BERT기반 3개로만 앙상블해서 추론한 결과와 5개 모델 전부 같은 하이퍼파라미터를 이용해서 학습한 후 추론한 결과를 비교해볼 예정이다.

# 모델 앙상블

모델 앙상블은 학습을 각각의 모델 따로따로 진행해준 후, 추론파트에서 각각의 모델을 불러와서 softVoting방식으로 레이블을 판단한다.

```python
# === Ensemble Training Block ===
# Replace the original training cell (ID: 176174ee) with this code.

# --- Model Training Settings ---
SAVE_MODEL = True
USE_WANDB = True # Set to True if you want to use Weights & Biases logging for each model
from transformers import EarlyStoppingCallback 
# List of models for ensemble
model_names = [
    # "klue/roberta-base",
    "klue/bert-base",
    "kykim/bert-kor-base",
    "beomi/kcbert-base"
    # "monologg/koelectra-base-v3-discriminator"
]

# --- Loop through each model for training ---
all_training_results = {}

for model_name in model_names:
    model_short_name = model_name.split('/')[-1].replace('-', '_') # Create a short name for directories/logging
    output_dir = f"./results_{model_short_name}"
    run_name = f"{model_short_name}-movie-review"

    print("\n" + "=" * 50)
    print(f"모델 훈련 시작: {model_name}")
    print(f"결과 저장 경로: {output_dir}")
    print("=" * 50)

    # --- Load Model and Tokenizer for the current model ---
    try:
        current_tokenizer = AutoTokenizer.from_pretrained(model_name)
        current_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=NUM_CLASSES,
        )
        current_model.to(device) # Move model to GPU

        # --- Add new special tokens and resize embeddings ---
        num_added_toks = current_tokenizer.add_tokens(NEW_SPECIAL_TOKENS)
        if num_added_toks > 0:
            print(f"Added {num_added_toks} new special tokens for {model_name}.")
            current_model.resize_token_embeddings(len(current_tokenizer))
            print("Resized model token embeddings.")

    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Skipping this model.")
        continue # Skip to the next model if loading fails

    # --- Recreate Datasets with the current tokenizer ---
    # This is necessary because different models might have different tokenization
    train_dataset_current = ReviewDataset(
        train_data["review"],
        train_data["label"],
        current_tokenizer,
        CHOSEN_MAX_LENGTH,
    )
    val_dataset_current = ReviewDataset(
        val_data["review"],
        val_data["label"],
        current_tokenizer,
        CHOSEN_MAX_LENGTH,
    )

    # --- Training Arguments for the current model ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=BATCH_SIZE_EVAL,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch" if SAVE_MODEL else "no",
        load_best_model_at_end=SAVE_MODEL,
        metric_for_best_model= "eval_loss" if SAVE_MODEL else None,
        greater_is_better=False,
        save_total_limit=1 if SAVE_MODEL else 0, # Save only the best model
        label_smoothing_factor=0.18547040498752376,
        report_to="wandb" if USE_WANDB else "none",
        run_name=run_name if USE_WANDB else None,
        seed=RANDOM_STATE,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        remove_unused_columns=False,
        push_to_hub=False,
        gradient_accumulation_steps=4,
        logging_first_step=True,
        save_safetensors=SAVE_MODEL,
        lr_scheduler_type= "constant",
        # early_stopping_patience=3, # Keep early stopping if desired
        # early_stopping_threshold=0.001,
    )
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3) # <--- 원하는 patience 값
    # --- Initialize Trainer for the current model ---
    trainer = CustomTrainerWithFocalLoss(
        model=current_model,
        args=training_args,
        train_dataset=train_dataset_current,
        eval_dataset=val_dataset_current,
        tokenizer=current_tokenizer,
        num_classes=NUM_CLASSES,
        callbacks=[early_stopping_callback],
        focal_loss_alpha=[0.11, 0.45, 0.12, 0.32], # Adjust alpha if needed
        focal_loss_gamma=2.965201490877045,
        data_collator=DataCollatorWithPadding(tokenizer=current_tokenizer),
        compute_metrics=compute_metrics,
    )

    # --- Training Execution ---
    print(f"훈련 정보 ({model_name}):")
    print(f"  훈련 샘플: {len(train_dataset_current):,}개")
    print(f"  검증 샘플: {len(val_dataset_current):,}개")
    print(f"  훈련 에포크: {training_args.num_train_epochs}회")
    print(f"  배치 크기: {BATCH_SIZE_TRAIN} (훈련) / {BATCH_SIZE_EVAL} (검증)")
    print(f"  학습률: {LEARNING_RATE}")
    print(f"  시드값: {RANDOM_STATE}")
    print(f"  디바이스: {device}")
    print(f"  wandb 사용: {USE_WANDB}")

    try:
        training_result = trainer.train()
        print(f"\n{model_name} 훈련 완료")
        print(f"  최종 훈련 손실: {training_result.training_loss:.4f}")
        all_training_results[model_name] = training_result

        # --- Save the final best model ---
        if SAVE_MODEL:
            best_model_path = f"{output_dir}/best_model_{model_short_name}"
            print(f"최종 베스트 모델 저장 중: {best_model_path}")
            trainer.save_model(best_model_path)
            current_tokenizer.save_pretrained(best_model_path)
            print(f"최종 베스트 모델 저장 완료: {best_model_path}")

    except KeyboardInterrupt:
        print(f"\n사용자에 의해 {model_name} 훈련이 중단되었습니다.")
        # Optionally break the loop if one model training is interrupted
        # break
    except Exception as e:
        print(f"\n{model_name} 훈련 중 오류 발생: {str(e)}")
        # Optionally log the error and continue with the next model
        # continue

    # --- Clean up GPU memory before next model ---
    del current_model
    del current_tokenizer
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n" + "=" * 50)
print("모든 모델 훈련 완료!")
print("=" * 50)

```

훈련코드는 각각의 훈련과정에서 데이터누수를 막기위해 토크나이저와 모델을 따로 불러온 후, fit과 transform을 따로 진행해준다. 그리고 이전의 baseline code에서처럼 학습을 진행해준다.

```python
# === Ensemble Inference Block ===
# Replace the original inference cell (ID: 079c8f40) with this code.

import torch
import numpy as np
import os
from tqdm.auto import tqdm # Import tqdm for progress bar
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

print(f"현재 디바이스: {device}")

# --- Load Test Data ---
df_test = pd.read_csv("data/test.csv")
print(f"테스트 데이터 로드 완료: {len(df_test)} 샘플")

# --- Apply Preprocessing ---
# Use the 'preprocessor' instance fitted during training
print("테스트 데이터에 전처리 파이프라인 적용 중...")
test_texts = df_test["review"].tolist()
# Use transform only, assumes preprocessor is already fitted
test_processed = preprocessor.transform(test_texts)
print("테스트 데이터 전처리 완료")

# --- List of Trained Model Paths ---
# Ensure these paths match where models were saved in the training block
model_paths = [
    f"./results_{model_name.split('/')[-1].replace('-', '_')}/best_model_{model_name.split('/')[-1].replace('-', '_')}"
    for model_name in model_names # Use the same model_names list from training
]

# --- Ensemble Inference ---
all_logits = []
print("\n" + "=" * 50)
print("앙상블 추론 시작...")
print("=" * 50)

for model_path in tqdm(model_paths, desc="모델별 추론 진행"):
    if not os.path.exists(model_path):
        print(f"Warning: 모델 경로를 찾을 수 없습니다: {model_path}. 이 모델을 건너뜁니다.")
        continue

    print(f"\n모델 로딩: {model_path}")
    try:
        # Load tokenizer and model for the current path
        current_tokenizer = AutoTokenizer.from_pretrained(model_path)
        current_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        current_model.to(device)
        current_model.eval()

        # Create test dataset with the current tokenizer
        test_dataset_current = ReviewDataset(
            pd.Series(test_processed), # Pass processed text as a Series
            None, # No labels for test set
            current_tokenizer,
            CHOSEN_MAX_LENGTH
        )

        # Define minimal TrainingArguments for prediction
        temp_output_dir = f"./temp_prediction_{os.path.basename(model_path)}"
        predict_args = TrainingArguments(
            output_dir=temp_output_dir,
            per_device_eval_batch_size=BATCH_SIZE_EVAL * 2, # Increase batch size for faster inference
            dataloader_num_workers=2,
            remove_unused_columns=False, # Keep columns needed by dataset
            fp16=torch.cuda.is_available(),
            logging_steps=len(test_dataset_current) // (BATCH_SIZE_EVAL * 2) + 1 # Reduce logging
        )

        # Initialize Trainer for prediction
        pred_trainer = Trainer(
            model=current_model,
            args=predict_args,
            tokenizer=current_tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=current_tokenizer),
        )

        # Get predictions (logits)
        print(f"모델 추론 중: {model_path}")
        predictions = pred_trainer.predict(test_dataset_current)
        logits = predictions.predictions # Logits are in predictions.predictions
        all_logits.append(logits)
        print(f"모델 추론 완료: {model_path}")

        # Clean up memory
        del current_model
        del current_tokenizer
        del pred_trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_output_dir):
             shutil.rmtree(temp_output_dir)

    except Exception as e:
        print(f"Error during inference for model {model_path}: {e}")
        # Decide whether to continue or stop if one model fails
        # continue

# --- Averaging Logits ---
if not all_logits:
    raise ValueError("추론 결과가 없습니다. 모델 경로 또는 추론 과정을 확인하세요.")

print("\n로짓(logits) 평균 계산 중...")
# Convert list of numpy arrays to a numpy array for easier calculation
all_logits_np = np.array(all_logits)

# Calculate the average logits across models (axis=0)
average_logits = np.mean(all_logits_np, axis=0)
print(f"평균 로짓 계산 완료. 형태: {average_logits.shape}")

# --- Final Prediction ---
predicted_labels = np.argmax(average_logits, axis=1)
print(f"최종 예측 레이블 생성 완료: {len(predicted_labels)}개")

# --- Add predictions to the test DataFrame ---
df_test["pred"] = predicted_labels
print(f"\ndf_test에 pred 컬럼이 추가되었습니다. 형태: {df_test.shape}")

# --- Analyze Prediction Distribution ---
unique_predictions, counts = np.unique(predicted_labels, return_counts=True)
print("\n최종 예측 분포:")
for pred, count in zip(unique_predictions, counts):
    percentage = (count / len(predicted_labels)) * 100
    class_name = LABEL_MAPPING.get(pred, f"클래스 {pred}")
    print(f"   {class_name} ({pred}): {count:,}개 ({percentage:.1f}%)")

print("\n" + "=" * 50)
print("앙상블 추론 완료!")
print("=" * 50)

# GPU 메모리 정리
if torch.cuda.is_available():
    torch.cuda.empty_cache()

```

이후 진행하는 inferece단계에서는 각 모델에서의 logit을 계산한 후 soft voting방식으로 레이블을 예측한다.

# 결과

### BERT기반 모델 3개를 앙상블한 결과

![image](/assets/images/![image.png](image%203.png).png)

BERT기반 모델 3개를 앙상블한결과를 SWA-1,2,3에 따라 나눈 결과이다. 총 에폭수가 5였으므로 3은 너무많고, 1은 너무적어서 SWA는 2가 최적이었던것 같다.

다만 지금까지 모델에 여러가지 튜닝을 해왔음에도 test/acc성능이 나오지 않는것은, 이유를 생각해봐야겠다.

![image](/assets/images/2025-10-28-10-30-09.png)

![image](/assets/images/2025-10-28-10-30-14.png)

추정컨데, 모델에 적용한, 앙상블, 하이퍼파라미터 튜닝을 통한 정규화, Focal Loss는 일반적으로 모델의 성능을 Rubust하게 만들어준다. 하지만 test/acc만이 평가 metric인 이상, 결국에 test/acc성능이 좋아야한다. 지금까지의 코드는 과적합을 막기위해 bestmodel선정을 valid/loss가 최소화된 모델을 선택했다. 그런데, 결국에는 acc성능이 평가 metric이기 때문에, acc를 기준으로 체크포인트를 선택해서 SWA를 해야하는게 아닌가 생각이 든다. 물론 valid/loss가 폭발하지 않는 선에서, 즉 과적합 되지 않는 선에서 정리를 해야한다.

그리고 Focal Loss의 $\alpha$가중치가 소수클래스에 대해 너무 큰 가중치를 주는게 아닌가 생각이 든다. test 데이터의 분포는 알 수 없지만, 일반적으로 80% acc를 갖는 모델의 예측분포는 전체 데이터셋의 분포와 유사한 분포를 띠고있다. 따라서 stratify된 분포를 갖지 않을까 추측해본다. 만약 전체 데이터셋의 분포와 같다면 강한 부정, 약한 긍정(다수클래스)은 훨씬 덜 선택하고 있는거고, 강한긍정, 약한부정(소수클래스) 는 오히려 정답레이블보다 많이 선택하고 있음을 확인할 수 있다. 따라서 Focal Loss의 가중치를 아예 1로 두고 실험을 해보고, test데이터셋의 분포를 더 추론해보고 acc의 성능을 파악해봐야겠다.

# 이후 실험에서 다뤄야 할것들

- base모델 바꿔서 하이퍼파라미터 튜닝
- SWA 구현해서 SWA횟수를 하이퍼파라미터로 넣기

- 데이터증강 실험해보고 효과 있으면 이것도 하이퍼파라미터로 넣기
- TAPT적용
- Boosting-adaboost staking구현 → 기존 코드에 implementation

- acc만이 평가지표로 활용된다. acc를 올릴 수 있는방법.. → 너무 많은 정규화를 하면 안되는걸까?

