# 문장 내 개체간 관계 추출

## Summary

이 과제의 주 목적은 각 문장 내에서 주어진 단어들(entities) 사이의 관계를 추측하는 언어모델을 학습시키는 것이다.
과제를 위해 주어진 데이터셋은 한글로 이루어진 언어 데이터이다.
기본적으로 주어진 베이스라인 코드에서도 허깅페이스 트랜스포머를 사용해서 모델을 학습시키고 있기에, 이 과제의 해결을 위해서 허깅페이스 트랜스포머를 사용해서 klue 데이터셋으로 pretrained된 언어모델들을 finetuning하는 방식으로 실험을 진행하였다.

## Experimental results

1) 단일 모델 성능

| Model | Data Augmentation | Feature | CV Score | Public LB Score |
| --- | --- | --- | --- | --- |
| roberta-large (linear) | no | no fgm, lr=2e-5 | 85.0718 | 73.7778 |
| roberta-large (linear) | no | fgm, lr=2e-5 | 85.2033 | 73.7778 |
| roberta-large (bi-lstm) | no | no fgm, lr=2e-5 | 84.8133 | 미제출 |
| roberta-large (bi-lstm) | no | fgm, lr=2e-5 | 84.9685 | 73.2562 |
| roberta-large (bi-lstm) | yes | fgm, lr=2e-5 | 84.8542 | 73.6476 |
| bert-base (linear) | no | no fgm, lr=2e-5 | 83.4184 | 68.8606 |

Publich Leaderboard 점수만 비교해보면 Linear head 모델이 micro f1 score가 더 높고, LSTM head 모델이 auroc 점수가 더 높다.

또한, 헤드모델의 종류와 관계없이 FGM을 적용한 모델이 그렇지 않은 모델에 비해 성능이 더 높게 나온다.

학습 데이터에 대해서 data augmentation을 적용한 뒤 모델을 학습시켰을 경우에는 헤드 모델의 종류에 관계 없이 오히려 micro f1 score가 떨어지는 문제가 발생하였다. 이러한 현상이 일어난 정확한 이유에 대해서는 알아내지 못하였다.

2) Ensemble 성능

아래의 model1 ~ model5는 각각 다음과 같다 (참고로 5개의 모델 모두 "klue/roberta-large"를 사용하였다)

- model1: linear head 모델, fgm 적용, data augmentation 미적용
- model1: linear head 모델, fgm 미적용, data augmentation 미적용
- model1: LSTM head 모델, fgm 적용, data augmentation 미적용
- model1: LSTM head 모델, fgm 미적용, data augmentation 미적용
- model5: linear head 모델, fgm 적용, data augmentation 적용

추가로, model1 ~ model5 column들은 각각 ensemble을 할 때 해당 모델의 output에 곱하는 크기를 나타낸다.

| model 1 | model 2 | model 3 | model 4 | model 5 | Public LB F1 score | Public LB auroc |
| --- | --- | --- | --- | --- | --- | --- |
| 0.35 | 0.2 | 0.2 | 0.15 | 0.1 | 74.2621 | 76.6472 |
| 0.3 | 0.2 | 0.2 | 0.15 | 0.15 | 74.1281 | 76.4925 |
| 0.4 | 0.3 | 0.15 | 0.15 | 0 | 73.9613 | 76.5048 |
| 0.25 | 0.25 | 0.25 | 0.25 | 0 | 73.8841 | 76.5953 |
| 0.5 | 0.3 | 0.1 | 0.1 | 0 | 73.8813 | 76.4344 |
| 0.55 | 0.3 | 0.1 | 0.05 | 0 | 73.8247 | 76.3879 |
| 0.6 | 0.3 | 0.15 | 0.05 | 0 | 73.7544 | 76.2841 |

## Instructions

기본적으로 모든 코드들은 jupyter notebook의 형태로 작성되었다. 각 노트북 내의 설정 값들을 원하는 값으로 설정한 뒤 노트북 내의 모들 셀들을 순차적으로 실행시키면 된다.

### Training

우선, 모델의 학습은 Upstage_train.ipynb 노트북을 활용하였다. 이 노트북에서 6번째 셀을 보면 CFG 클래스에 대한 정의가 나오게 된다. 이 CFG 클래스는 학습/추론 과정에 사용되는 설정 값들을 저장한다. 어떤 모델을 사용해서 finetuning을 진행할지 설정하는 부분은 model이다. 따라서, klue/roberta-large가 아닌 다른 모델을 활용해서 학습을 진행하고 싶다면 해당 부분을 변경하면 된다.

n_fold와 trn_fold는 k-fold와 관련된 설정 값들이다. n_fold는 k-fold의 k 값을 정하기 위한 변수이며, trn_fold는 이번 학습 과정에서 k-fold의 폴드들 중 어떤 폴드들에 대해서 학습을 진행할 지 결정할 수 있다. 이렇게 디자인을 한 이유는, 기본적으로 colab의 환경에는 제한시간이 있기 때문에 큰 fold 값을 설정하게 되면 모든 fold에 대해서 학습을 진행하기도 전에 런타임 환경이 종료되는 문제가 생기기 때문이다. trn_fold를 사용하면 지난 런타임에서 학습을 미처 다 끝내지 못한 폴드부터 다시 학습을 시작하도록 설정할 수 있기 때문이다.

use_rnn은 헤드모델로 LSTM을 사용할지에 대해서 결정하는 변수이다. use_rnn=True이면 LSTM + MeanPooling 헤드 모델을 사용하고, False이면 Linear 헤드 모델을 사용한다. bidirectional은 use_rnn이 True일 때 사용되는 변수로, LSTM 층이 bidirectional 하게 작용할지 여부에 대한 설정 값이다. fc_dropout은 헤드모델 내에 있는 dropout layer의 dropout probability에 대한 설정 값이다.

use_fgm은 adversarial training을 적용할지 여부에 대해서 정하는 변수이다. 이 값을 True로 설정하면 학습 루프 내에 FGM(Fast Gradient Method)방식이 적용되게 된다. add_augmented는 data augmentation을 통해 추가된 데이터를 학습 데이터에 추가를 할지에 대해서 정하는데 사용되는 변수이다. 이때, data augmentation을 통해서 생성된 csv파일 (augmented.csv)파일이 없으면 file not found 에러가 나기 때문에, 이 값을 True로 설정하려면 Data augmentation을 먼저 실행시키면 된다.

위의 CFG 변수들을 원하는 값으로 잘 설정해 놓은 뒤, 노트북을 전체 실행을 시키면 학습이 자동으로 실행된다.

### Data augmentation

Data augmentation을 위한 노트북으로는 GoogleTranslation.ipynb이 있다. 이 노트북을 열게 되면 6번째 셀에 df.sample() 메소드를 사용하는 코드가 있다. 여기서 fraction의 값에 따라 sampling의 비율이 바뀌게 된다.

원하는 sampling fraction을 설정한 뒤, 노트북을 실행시키면 데이터 어그멘테이션 과정이 진행된다.

### Inference

추론을 위한 노트북은 모두 2개가 존재한다.

우선, 첫번째로 Upstage_inference.ipynb 파일이 있는데, 이 파일은 단일 모델만을 사용해서 inference를 진행하는 방식으로 동작한다. 이 노트북 파일에서도 CFG 클래스에 대해서 정의한 부분이 있는데, 대부분 학습 코드에서도 봤던 변수들일 것이다. 이들 중 추론 과정만을 위해 추가된 변수가 하나 있는데, 바로 "path"이다. 이 path 변수는 학습을 통해 생성된 파일들이 저장되어 있는 폴더의 경로값을 넣어주어야 한다. 추가로, 이 path의 값은 반드시 "/"로 끝나야 한다 (i.e. "./upstage-roberta-large-bi-lstm-fgm/"). 만약 정확한 경로값을 설정하지 않으면 state_dict 파일이나 tokenizer 파일 등을 로딩할 때 에러가 발생하게 된다. 실제로 리더보드에 제출한 모델들을 사용하기 위한 parameter 조합은 다음과 같다.

| model | path | use_rnn |
|:---:|:---:|:---:|
| klue/roberta-large | ./upstage-roberta-large-bi-lstm-fgm/ | True |
| klue/roberta-large | ./upstage-roberta-large-bi-lstm-nofgm/ | True |
| klue/roberta-large | ./upstage-roberta-large-lr2e5-fgm/ | False |
| klue/roberta-large | ./upstage-roberta-large-lr2e5-nofgm/ | False |
| klue/roberta-large | ./upstage-roberta-large-lstm-outputs/ | False |

Ensemble을 위한 노트북은 Upstage_ensemble.ipynb 파일 안에 있다. 해당 파일을 열게 되면 2번째 셀에서 w1~w5의 값들이 있다. 해당 변수들은 model 1 ~ model 5까지의 모델에 대해서 ensemble 시에 적용하는 비율을 정의한다. 즉, pn이 n번째 모델의 결과라고 하면 ensemble의 결과는 w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4 + w5 * p5이다. 해당 값들을 원하는 값들로 변환한 뒤 모든 셀들을 순차적으로 실행시키면 된다.

두 노트북 모두 실행이 성공적으로 완료가 되면 submission.csv 파일을 생성하는데, 이 파일이 바로 리더보드에 제출하는 결과 파일이다.

## Approach

과제의 성공적인 수행을 위해서 다음과 같은 실험 계획을 세웠다.

1) 모델 선택 - 어떤 Transformer 모델을 사용해야 더 좋은 결과를 얻을 수 있을까?
2) learning rate - 초기 learning rate 값은 어떤 값을 사용하는 것이 좋을까?
3) K-fold CV - 과연 k-fold에서 k의 값을 얼마로 하는 것이 좋을 것이며, 어떤 식으로 폴드를 나누어야 할까?
4) warm-up step - warm-up step을 추가하는 것이 더 나을까?
5) Adversarial Learning - FGM 같은 적대적 학습 기법을 적용하면 더 성능이 좋아질까?
6) Data Augmentation - 데이터 증강기법을 통해서 학습 데이터를 늘리면 성능이 더 좋아질까?
7) Ensemble - 학습시킨 다양한 모델들을 어떻게 앙상블해야 더 좋은 점수를 얻을 수 있을까?

### 모델

우선 주어진 베이스 코드를 보면 metric 계산 관련 함수의 이름들에 "klue"가 들어 있는 것을 봐서는 학습 데이터와 테스트 데이터가 klue와 관련이 있지 않을까하는 가설을 세웠다. 게다가 베이스 코드에서 사용하는 모델이 "klue/bert-base" 모델이었기에 데이터가 klue 벤치마크와 완전히 무관하지는 않을 것이라고 생각하게 되었다.

klue 벤치마크의 리더보드를 보면 RE 부문에서 SOTA를 차지한 모델은 "klue/roberta-large" 모델이었다. 따라서 이 과제의 기본 모델로 "klue/roberta-large" 모델을 사용하기로 정하였다. 기본적으로 모델의 구성은 "klue/roberta-large" 모델을 Huggingface transformers의 AutoModel을 사용해서 로딩한 뒤, 해당 모델의 끝에 간단한 헤드를 붙이고 fine-tuning을 시키는 전략을 채택하기로 하였다.

이번 과제를 위해 구현된 헤드 모델은 총 2가지 종류가 있다.

    (1) Dropout + Linear (편의상 linear head 모델로 부르도록 하겠다)
    (2) bi-LSTM + MeanPooling + Dropout + Linear (편의상 lstm head 모델로 부르도록 하겠다)

위의 구조들을 선택한 가장 큰 이유로는 최근 캐글에서 참여했었던 "US - Patent Phrase to Phrase Matching" 대회에서 위의 모델들을 활용하여서 메달을 땄었기 때문이다. 위의 대회의 주 task는 주어진 target phrase와 anchor phrase 사이의 유사도를 점수화하는 것이었다. 물론, phrase matching은 relation extraction과는 다른 task이지만, 문장간의 유사도를 계산하는 것이 문장 내의 객체 간의 관계를 유추하는 것과 완전히 무관하지는 않을 것이라고 생각이 되었다. 따라서, 이 과제를 풀기 위해 "US - Patent Phrase to Phrase Matching" 대회에서 사용했었던 모델 구조를 사용하기로 마음을 먹게 되었다.

### 학습

학습을 진행하기 전에 앞서, 주어진 3개의 지표들(micro f1, auroc, accurac) 중에서 어떤 지표를 사용해서 best model을 판단할 지에 대해서 많은 고민을 하였다. 그러나, 결국 리더보드에서 가장 우선시하는 지표인 micro f1을 모델 평가 지표로 사용해서 epoch을 진행하면서 가장 micro f1 점수가 높은 모델을 best model로 고려하기로 결정하였다.

#### Mixed Precision Training

알다시피, 파이토치에서는 mixed precision training을 지원하기 위해 cuda amp(automatic mixed precision)를 제공한다. 이 기능을 사용하면 처리 속도를 높이기 위한 FP16(16bit floating point)연산과 정확도 유지를 위한 FP32 연산을 섞어서 학습을 진행하게 된다. 더 빠른 학습을 가능하게 하면서 성능을 유지하기 위해서 학습 코드에 torch.cuda.amp 모듈의 기능을 사용하였다.

#### Loss function

우선 이 과제에서 성능 테스트를 위해서 주어진 함수를 보게 되면, 총 30개의 클래스가 정의되어 있음을 알 수 있다. 즉, 이 문제는 multiclass classification 문제이므로 분류와 관련된 loss 함수를 사용해서 학습을 진행해야 한다. 가장 기본적인 손실함수로는 Cross Entropy가 있을 수 있다. 게다가, 학습 데이터 내에서 라벨의 분포가 고르게 되어 있는지 간단하게 확인을 해 보니 모든 라벨이 고르게 분포되어 있지는 않은 것을 확인할 수 있었다. 그대로 바로 학습을 진행하면 비교적 적은 라벨에 대해서는 학습이 잘 이루어지지 않을 수 있기 때문에 단순히 Cross Entropy 함수를 쓰기 보다는 Label smoothing을 사용하는 Label Smoothed Cross Entropy Loss를 파이토치를 사용해서 간단하게 구현한 뒤 손실 함수로 사용하였다.

#### Learning Rate

주어진 베이스 코드에서는 initial learning rate 값으로 5e-5를 사용하고 있기에 처음에는 해당 값을 사용해서 학습을 진행해보았는데 weight&bias에서 그래프를 보니 loss가 잘 수렴하지 않는 것처럼 보였다. 따라서 learning rate 값을 조금씩 줄여나갔고, 그나마 2e-5를 쓸 때 조금 더 loss가 학습을 진행하면서 더 잘 줄어들어서 initial learning rate 값으로 5e-5가 아닌 2e-5를 사용하였다.

Optimizer로는 트랜스포머 기반 모델들의 학습에 가장 많이 사용되는 AdamW를 사용하였으며, scheduler로는 huggingface transformers의 get_cosine_schedule_with_warmup 함수를 활용하였다.

#### K-fold

K-fold 방식은 CV(cross validation)에서 가장 자주 사용하는 기법이다. 데이터셋을 k개의 폴드로 나눈 뒤, k번 돌아가면서 각 차례마다 해당하는 폴드를 validation set으로 사용하고 나머지 데이터를 training set으로 사용하는 기법이다. k-fold에서 k의 값이 클 수록 학습에 사용되는 데이터의 양이 많아지기 때문에 더 큰 k 값을 사용하면 더 좋은 성능의 모델을 학습시킬 수 있는 경우가 많다. 물론, 늘 큰 k 값이 최선의 선택은 아니다.

이번 과제를 진행함에 있어서 가장 적당한 크기의 k 값은 몇인지 확인하기 위해서 5-fold와 10-fold에 대해서 각각 학습을 진행하고 성능을 비교해보았다. 그 결과, 10-fold일 때에 비해 오히려 5-fold 모델의 f1-score가 더 높았다. 물론 auroc 점수 역시 5-fold 모델이 10-fold 모델보다 더 좋은 결과를 보였다. 물론 더 좋은 k 값이 있을 수도 있겠지만, k값을 찾는 데에 이 이상의 시간을 사용하는 것은 좋지 않을 것이라고 생각하여서 모든 모델에 대해서 5-fold를 사용하기로 정하였다.

##### CV split

CV를 위해서 k-fold로 나눌 때, 단순히 데이터셋을 k등분하는 방법보다는 데이터를 조금 더 의미있게 나누는 것이 모델 성능 향상에 도움이 되기도 한다. 예를 들어, 앞서 언급하였던 "US - Patent Phrase to Phrase Matching" 대회에서 k-fold 기법을 사용할 때 단순히 데이터를 k개로 등분하기 보다는 target phrase에 대해서 grouping을 한 뒤, scikit-learn의 MultilabelStratifiedKFold을 사용해서 폴드들을 나누는 기법을 사용했을 때 꽤나 큰 점수 향상을 얻을 수 있었다. 따라서, 이번 과제에서도 k개로 나누기보다는 조금 더 의미있게 나누는 것이 좋을 것이라고 생각을 하게 되었다. 이를 위해 MultilabelStratifiedKFold를 사용하였으며, 본 과제에서 가장 중요한 데이터 중 하나인 subject_entity에 대해서 grouping을 한 뒤, 이를 활용해서 k-fold로 나누는 전략을 사용하였다. 그 결과, 단순히 k등분할 때에 비해 auroc와 micro-f1 모두 더 높은 점수를 달성할 수 있었다.

#### warm-up steps

DeBERTa나 RoBERTa 등의 논문들을 보게 되면, 학습을 진행할 때 적당한 warm-up step이 학습에 도움을 준다는 결과를 확인해 볼 수 있다. 이 warm-up step이 이번 데이터셋의 학습에서도 도움이 될지 알아보기 위해 linear head 모델을 각각 warm-up = 0과 warm-up = 50에 대해서 학습을 시켜보았다. 성능 비교를 위해 5-fold 방식으로 학습을 진행하였으며, 성능 비교는 f1-score를 비교하였다. 그 결과, warm-up = 50인 모델은 f1-score가 84.37이 나온 반면, warm-up = 0인 모델은 f1-score가 83.62가 나왔다. 해당 모델들에 대해서 Leaderboard score 비교는 하지 않았지만, CV 점수에서 더 좋은 결과가 나왔으며 과제의 제한 기간이 있는만큼 warm-up 파라미터에 대해 추가적인 실험을 하기 보다는 warm-up 값을 50을 사용하기로 정하였다.

#### Adversarial Learning

이번 과제에서 모델의 성능 향상을 위해서 Adversarial Learning 기법을 도입하였다. 다양한 적대적 학습 기법들 중 이번에 사용한 기법은 FGM(Fast Gradient Method)라는 기법이다. 이 방법은 최근 캐글의 NLP 대회들에서 많이 사용되는 방법으로 메달권 내에서 더 좋은 성적을 내기 위한 상위권 참가자들이 자주 사용하는 기법이다. 사용방법은 아주 간단하다. FGM 클래스를 미리 정의한 뒤, 아래의 코드를 파이토치 학습 코드 내에 추가하면 된다.

```python
fgm = FGM(model)

for batch_input, batch_label in data:
    loss = model(batch_input, batch_label)
    loss.backward()  

    # adversarial training
    fgm.attack() 
    loss_adv = model(batch_input, batch_label)
    loss_adv.backward() 
    fgm.restore()  

    optimizer.step()
    model.zero_grad()
```

실제로, 이번 과제에서도 이 방식으로 학습을 진행해서 CV score와 LB score 모두 향상시킬 수 있었다.

#### Data augmentation

머신러닝 학습을 진행함에 있어서 데이터가 가장 중요한 요소라는 점은 누구나 잘 알고 있는 사실이다. 그러나 이 대회의 경우 외부 데이터셋을 활용하는 것이 금지되어 있기에, 주어진 학습 데이터를 활용해서 최대한의 학습을 이루어내야만 한다. 이를 위해 주어진 학습 데이터를 사용해서 data augmentation을 진행할 수 있는 방법에 대해서 고심해보았고, 최종적으로 생각해낸 방법은 주어진 학습 데이터를 일정 비율로 샘플링한 뒤, 샘플링된 데이터를 google translation API를 사용해서 영어로 번역을 했다가 다시 한국어로 재번역을 하는 방식을 통해서 데이터를 추가하였다. 이때, 재번역된 문장이 기존 문장과 같을 경우에는 augmentation dataset에 추가하지 않았다.

구글 번역 API를 위해 사용한 언어는 영어와 일본어가 있었는데, 이 둘 중 영어로 했을 때 새로 생기는 문장의 수가 더 많았기에 한국어를 영어로 번역하였다가 다시 한국어로 번역하는 기법을 통해서 data augmentation을 진행하였다.

번역을 통해서 데이터를 증강시키다보니 몇가지 문제점이 생기게 되었다. 우선, 영어로 번역했다가 다시 한글로 바꾼 문장이 문법적으로 잘 맞지 않는 경우가 있었다. 번역한 문장을 재번역하는 과정으로 데이터를 늘리다보니 생각보다 결과 문장의 퀄리티가 떨어지는 경우가 꽤 있었다. 생성된 문장이 그렇게 좋지 않은 데이터라면 굳이 모든 학습 데이터에 대해서 data augmentation을 적용하기보다는 학습 데이터에서 일정 비율만큼 샘플링을 진행한 뒤, 뽑은 데이터에 대해서만 data augmentation 기법을 적용하는 것이 더 좋을 것이라고 생각이 되었다. 이것이 일정 비율만큼만 샘플링을 한 뒤 data augmentation 기법을 적용한 이유이다.

이렇게 결정을 한 이후, 테스트 삼아서 10% 샘플링을 한 뒤 data augmentation을 적용하였는데, 재번역한 문장이 기존 문장과 같아서 augmentation 결과에 포함되지 않는 경우가 있어서 그런지 생각보다 생성된 데이터의 크기가 작았다. 따라서, 30% 샘플링을 한 뒤 data augmentation을 적용하였다. 이렇게 하니 약 3000개의 문장이 새로 생성이 되었다.

이렇게 data augmentation을 적용한 데이터를 추가한 뒤 학습을 진행해 본 결과, 오히려 CV score가 더 떨어지는 상황이 발생하였다. 물론, public LB 점수 역시 약간이지만 더 떨어졌다. public LB score만 감소하였다면 모르겠지만 CV score 역시 감소하였기에 data augmentation을 하지 않은 데이터로 학습 시킨 모델들을 ensemble 시에 메인 모델로 활용하기로 결정하였다.

### Ensemble

다양한 모델들을 학습시킨 뒤, 최종 제출을 위해서 ensemble을 진행하였다. Ensemble을 적용하면서 확인했던 점으로는, Linear 헤드 모델의 비중을 늘리면 micro f1 score가 높아지게 되고, LSTM 헤드 모델의 비중을 늘리면 auroc가 높아지게 된다. 리더보드에서 우선적으로 보는 metric은 micro f1 score이기에 Linear 헤드 모델의 비중을 LSTM 헤드 모델보다 높게 설정하였다.
