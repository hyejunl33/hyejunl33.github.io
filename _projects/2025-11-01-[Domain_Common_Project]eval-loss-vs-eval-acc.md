---
layout: modern-single
title: "[영화리뷰 감성분류][모델최적화]-eval/loss vs eval/acc"
date: 2025-11-01
tags:
  - Domain_Common_Project
  - study
  - ModelOptimization
excerpt: "[모델최적화]-eval/loss vs eval/acc"
math: true
---



# Introduction

해당 test의 평가 지표는 test/acc이다. 지금까지의 실험 가설은 eval/loss를 줄여서 과적합을 막으면 자동으로 test/acc도 올라갈것이라는 가정이 있었다. 하지만 eval/loss를 metric으로 모델을 선택해서 test를 해본 결과, 성능이 오히려 하락하는 문제가 있었다.

![image](/assets/images/2025-11-01-13-30-04.png)

![image](/assets/images/2025-11-01-13-30-13.png)

그리고 두번째로, 모델이 과하게 소수클래스를 중요하게 보는 문제가 있었다. test데이터셋의 분포는 알 수 없지만, 일반적으로 전체데이터셋과 같다고 생각할때, Focal Loss의 $$\alpha$$의 가중치로 인해 소수클래스만을 너무 중요하게 여겨서, 전체 데이터셋의 분포와 예측 분포를 볼때, 소수클래스의 비율이 과장되게 예측됨을 확인할 수 있었다.

따라서 이번실험에서는 평가지표를 eval/acc로 바꿔본 후, Focal Loss의 가중치를 모두 1로 두고 성능을 관찰해본다. 시간관계상 앙상블한 모델이 아닌 단일모델로 실험을 한다. 평가지표를 acc로 바꿨기 때문에 eval/acc만 상승하고 eval/loss는 폭발하는 상황을 막기 위해서 기존에 8이었던 에폭을 5로 설정한다.

만약 유의미한 성능향상이 있을 경우 이 모델을 기준으로, 하이퍼파라미터 튜닝을 한 후 앙상블을 해서 성능을 실험해 볼 예정이다.

# Eval/accuracy를 메트릭으로 이용하자..

기존에는 너무 결과가 과적합되는것 같아서 eval/loss를 메트릭으로 사용해서 모델을 선정했다. 따라서 acc성능이 잘 나오지 않은것으로 추측된다. 따라서 이번에는 eval/accuracy를 메트릭으로 사용한다.

```python
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
        metric_for_best_model= "eval_accuracy" if SAVE_MODEL else None,
        greater_is_better=True,
        save_total_limit=SAVE_TOTAL_LIMIT_FOR_SWA if SAVE_MODEL else 0, # 수정: SWA 위해 늘림
        label_smoothing_factor=0.18547040498752376,
        report_to="wandb" if USE_WANDB else "none",
        run_name=run_name if USE_WANDB else None,
        seed=current_seed,
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
```

`metric_for_best_model= "eval_accuracy"` 를 인자로 줘서 eval_accuracy를 메트릭으로 사용한다. 그리고 loss와는 다르게 크면 클수록 좋은것이므로  `greater_is_better=True` 인자를 True로 바꿔줬다. 이와 더불어서 Trainer를 호출할때 `focal_loss_alpha=[1, 1, 1, 1],`  를 이용해서 클래스간 가중치를 아예 없애버렸다.

# 결과

![image](/assets/images/2025-11-01-13-30-26.png)

기존 실험들 중 가장 높은 성능을 보인 0.8211의 acc가 나왔다. 이제 확실한건, focal_loss의 alpha가중치를 신중하게 수정해야 한다는점, 그리고 모델을 선정할때는 최종 test를 위해서 accuracy를 기준으로 선정해야 한다는점이 확실해졌다.

![image](/assets/images/2025-11-01-13-30-32.png)

긍정적인 부분은 alpha가중치를 모두 1로 뒀음에도 f1이 accuracy와 비슷한 정도의 score를 갖는다는것이다. 따라서 가중치가 모두 같음에도 recall성능이 나쁘지 않다는것으로, alpha가중치를 이전에 백분율의 역수로 준게 매우 악영향을 미쳤었다는것을 추측할 수 있었다.

![image](/assets/images/2025-11-01-13-30-41.png)

모델의 최종 예측분포를 보면 EDA했을때 전체 데이터셋의 분포와 비슷함을 알 수 있다. 약한 부정비율은 많이 낮아졌지만, 그래도 다수의 클래스에서 Ground Truth와 같은 정답을 예측하는 숫자가 늘어났기 때문에 acc가 향상된것 같다.

이전에는 이 테스트의 목표는 acc를 높이는건데, 그러면 차라리 소수클래스를 무시하고, recall이 낮아도 상관없다는 말이니깐 loss는 커지더라도 acc만커지면 되니깐, loss는 오히려 어느정도는 커져도 상관없을것 같다고 생각을 했다. 하지만 f1 score와 acc가 동시에 올라가는 결과를 관측할 수 있었다.

이제 남은것은 BERT기반 모델 3개의 앙상블을 통한 모델이 더 높은 성능을 보이는지 여부를 확인하고, 하이퍼파라미터 튜닝을 통해 최적의 하이퍼파라미터로 성능을 뽑아내는것이다.

## BERT기반 모델 3개 앙상블

```python
# === Ensemble Training Block ===
# Replace the original training cell (ID: 176174ee) with this code.

# --- Model Training Settings ---
SAVE_MODEL = True
USE_WANDB = True # Set to True if you want to use Weights & Biases logging for each model
from transformers import EarlyStoppingCallback 
# List of models for ensemble
model_names = [
    # "./tapted_klue_roberta_base",
    "./tapted_klue_bert_base",
    "./tapted_kykim_bert_kor_base",
    "./tapted_beomi_kcbert_base"
    # "./tapted_koelectra_base_v3"
]

# --- Loop through each model for training ---
all_training_results = {}

for model_name in model_names:
    current_seed = random.randint(1, 100000)
    model_short_name = model_name.split('/')[-1].replace('-', '_') # Create a short name for directories/logging
    output_dir = f"./results_{model_short_name}"
    run_name = f"{model_short_name}-movie-review"

    print("\n" + "=" * 50)
    print(f"모델 훈련 시작: {model_name}")
    print(f"결과 저장 경로: {output_dir}")
    print(f"현재 시드값: {current_seed}") # 💡 현재 시드값 출력
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
    
    # SWA 설정 추가
    SWA_K = 2 # 평균낼 마지막 체크포인트 개수
    SAVE_TOTAL_LIMIT_FOR_SWA = SWA_K + 1 # best 모델 + 마지막 K개 저장되도록 (최소 2)

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
        metric_for_best_model= "eval_accuracy" if SAVE_MODEL else None,
        greater_is_better=True,
        save_total_limit=SAVE_TOTAL_LIMIT_FOR_SWA if SAVE_MODEL else 0, # 수정: SWA 위해 늘림
        label_smoothing_factor=0.18547040498752376,
        report_to="wandb" if USE_WANDB else "none",
        run_name=run_name if USE_WANDB else None,
        seed=current_seed,
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
        # tokenizer=current_tokenizer,
        num_classes=NUM_CLASSES,
        callbacks=[early_stopping_callback],
        focal_loss_alpha=[1, 1, 1, 1], # Adjust alpha if needed
        focal_loss_gamma=2.965201490877045,
        data_collator=DataCollatorWithPadding(tokenizer=current_tokenizer),
        compute_metrics=compute_metrics,
    )
```

```python
    # --- Training Execution ---
    print(f"훈련 정보 ({model_name}):")
    print(f"  훈련 샘플: {len(train_dataset_current):,}개")
    print(f"  검증 샘플: {len(val_dataset_current):,}개")
    print(f"  훈련 에포크: {training_args.num_train_epochs}회")
    print(f"  배치 크기: {BATCH_SIZE_TRAIN} (훈련) / {BATCH_SIZE_EVAL} (검증)")
    print(f"  학습률: {LEARNING_RATE}")
    print(f"  시드값: {current_seed}")
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

            
            # === SWA (Stochastic Weight Averaging) 적용 시작 ===
            print(f"\n--- SWA 가중치 평균 시작 (모델: {model_name}) ---")
            print(f"마지막 {SWA_K}개 체크포인트를 평균냅니다.")

            try:
                # 1. SWA에 사용할 체크포인트 경로 찾기
                checkpoint_dir = output_dir
                checkpoints = sorted(
                    glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
                    key=lambda x: int(x.split('-')[-1]) # 스텝 번호 기준 정렬
                )

                if len(checkpoints) >= SWA_K:
                    swa_checkpoints = checkpoints[-SWA_K:]
                    print(f"SWA에 사용할 체크포인트 ({len(swa_checkpoints)}개):")
                    for ckpt in swa_checkpoints:
                        print(f" - {os.path.basename(ckpt)}")

                    # 2. 기본 모델 로드 (구조만 사용)
                    # best_model_path에서 config를 로드하여 동일한 구조 보장
                    print("SWA 기본 모델 구조 로딩 중...")
                    config = AutoConfig.from_pretrained(best_model_path)
                    swa_base_model = AutoModelForSequenceClassification.from_config(config)
                    swa_base_model.to(device)

                    # 3. 가중치 평균내기
                    print("체크포인트 가중치 평균 계산 중...")
                    avg_state_dict = OrderedDict()
                    model_keys = swa_base_model.state_dict().keys() # 모델 키 저장

                    for i, checkpoint_path in enumerate(swa_checkpoints):
                        print(f"  ({i+1}/{len(swa_checkpoints)}) 체크포인트 로딩: {checkpoint_path}")
                        # safetensors 우선 로드 시도
                        safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
                        bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")

                        if os.path.exists(safetensors_path):
                            state_dict = load_safetensors_file(safetensors_path, device="cpu")
                        elif os.path.exists(bin_path):
                            state_dict = torch.load(bin_path, map_location="cpu")
                        else:
                            print(f"  경고: {checkpoint_path} 에서 모델 파일을 찾을 수 없습니다. 건너<0xEB><0x9A><0x8D>니다.")
                            continue

                        # 평균 계산 (첫 번째 체크포인트는 복사, 이후는 더하기)
                        for key in model_keys:
                            if key not in state_dict:
                                print(f"  경고: 키 '{key}'가 체크포인트에 없습니다. 건너<0xEB><0x9A><0x8D>니다.")
                                continue
                            if i == 0:
                                avg_state_dict[key] = state_dict[key].clone().float() # float()로 타입 통일
                            else:
                                if key in avg_state_dict:
                                     avg_state_dict[key] += state_dict[key].float() # float()로 타입 통일
                                else:
                                     print(f"  경고: 이전 체크포인트에 없던 키 '{key}' 발견. 건너<0xEB><0x9A><0x8D>니다.")

                        del state_dict # 메모리 확보

                    # 평균 계산 (K로 나누기)
                    num_averaged = len(swa_checkpoints) # 실제 로드된 체크포인트 수
                    if num_averaged > 0:
                        for key in avg_state_dict:
                            avg_state_dict[key] /= num_averaged

                        # 4. 평균낸 가중치를 모델에 로드
                        missing_keys, unexpected_keys = swa_base_model.load_state_dict(avg_state_dict, strict=False)
                        if missing_keys: print(f"  경고: 로드되지 않은 키: {missing_keys}")
                        if unexpected_keys: print(f"  경고: 예상치 못한 키: {unexpected_keys}")
                        print("평균 가중치 모델에 로드 완료.")

                        # 5. 배치 정규화(Batch Normalization) 통계 업데이트 (매우 중요!)
                        print("배치 정규화 통계 업데이트 중...")
                        # update_bn 함수는 DataLoader가 필요
                        swa_data_collator = DataCollatorWithPadding(tokenizer=current_tokenizer)
                        train_dataloader_for_bn = DataLoader(
                            train_dataset_current,
                            batch_size=BATCH_SIZE_TRAIN, # 훈련 배치 크기 사용 권장
                            collate_fn=swa_data_collator,
                            shuffle=False, # BN 업데이트 시에는 셔플 불필요
                            num_workers=2
                        )
                        # update_bn 함수 호출
                        # swa_utils.update_bn()은 AveragedModel 객체를 받지만,
                        # 여기서는 직접 평균낸 모델을 사용하므로, 수동 루프 또는 아래와 같이 직접 호출
                        swa_base_model.train() # BN 업데이트는 train 모드에서
                        with torch.no_grad():
                            num_batches_tracked = 0
                            for batch in tqdm(train_dataloader_for_bn, desc="BN 업데이트", leave=False):
                                inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
                                if not inputs: continue # 입력이 없는 경우 건너뛰기
                                swa_base_model(**inputs)
                                num_batches_tracked += 1
                                if num_batches_tracked >= 100: # 전체 데이터 대신 일부(예: 100 배치)만 사용해도 충분할 수 있음
                                    break
                        swa_base_model.eval() # 다시 eval 모드로
                        print(f"배치 정규화 통계 업데이트 완료 (약 {num_batches_tracked} 배치 사용).")

                        # 6. SWA 모델 저장
                        swa_model_path = f"{output_dir}/swa_model_{model_short_name}"
                        os.makedirs(swa_model_path, exist_ok=True)
                        print(f"SWA 모델 저장 중: {swa_model_path}")
                        # state_dict() 저장 방식 사용
                        torch.save(swa_base_model.state_dict(), os.path.join(swa_model_path, "pytorch_model.bin"))
                        # save_pretrained는 전체 모델 객체를 저장하려 하므로 여기서는 state_dict 저장 권장
                        # swa_base_model.save_pretrained(swa_model_path) # 필요시 사용 가능
                        current_tokenizer.save_pretrained(swa_model_path) # 토크나이저도 함께 저장
                        # config.json도 저장 (save_pretrained가 없으므로 수동 복사 또는 저장)
                        config.save_pretrained(swa_model_path)

                        print(f"SWA 모델 저장 완료: {swa_model_path}")

                    else:
                        print("평균낼 유효한 체크포인트가 없습니다.")

                else:
                    print(f"SWA를 위한 체크포인트가 충분하지 않습니다 ({len(checkpoints)}개 발견, {SWA_K}개 필요). SWA를 건너<0xEB><0x9A><0x8D>니다.")

            except Exception as e:
                print(f"SWA 처리 중 오류 발생: {e}")
            finally:
                # SWA 모델 객체 메모리 정리 (필요시)
                if 'swa_base_model' in locals():
                    del swa_base_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print("--- SWA 가중치 평균 종료 ---")
        # === SWA 적용 종료 ===
        
        
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

우선 앙상블할 각 모델마다 랜덤시드를 다르게주도록 구현해서 모델들이 보는 데이터셋의 분산을 키우고자 했다. 일반적으로 앙상블의 효과는 각 모델이 상이한 부분에서 하는 실수를 매꿀때 극대화 되므로 서로다른 모델이 학습할때마다 루프안에서 다른 랜덤시드를 가지도록 코드로 구현했다.

### 앙상블한 결과

![image](/assets/images/2025-11-01-13-30-56.png)

모델 3개를 앙상블한 결과 acc가 0.7% 올랐다. 즉 유의미한 효과가 있었다.

![image](/assets/images/2025-11-01-13-31-02.png)

이전 실험과 마찬가지로 최종 예측분포를 보면 전체 데이터셋의 분포와 비슷하게 맞춤을 알 수 있다. acc를 높이려면 test 데이터의 분포와 유사해야함을 시사한다. 그리고 소수를 차지하는 분포인 약한부정은 더 덜맞추는 경향성이 있지만, 0,1,3레이블은 더 정교하게 원래 데이터분포와 비슷해졌음을 알 수 있다. 대다수의 레이블을 잘 못추면 역시 acc가 올라감을 확인할 수 있었다.

# 이후 실험에서 고려해야 할 것들

- 하이퍼파라미터 튜닝
    - Focal Loss에서 알파 가중치를 얼마나 줘야할까?
        - 최적의 알파 가중치는 얼마일까?
    - 이외에 하이퍼파라미터 튜닝할 수 있는값 모두 넣기
- valid 데이터를 포함해서 전체 데이터를 기준으로 학습시켜보기
- 모델 앙상블 추론을 할때, 모델별로 Weight를 줘서 앙상블을 해보기