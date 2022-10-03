# DETR

## 핵심 아이디어

- CNN이 아닌 Vanilla Transformer를 주요 backbone 모델로 활용해 object detection 수행
- 기존 객체 탐지 모델들은 너무 복잡해서 다양한 라이브러리를 사용해야 하며, 모델의 이해가 어려움
    - NMS, Anchor, IoU 등
    - DETR는 Transformer와 bipartite matching을 사용해 기존 모델들의 복잡성을 줄임
- DETR은 CNN을 사용하지 않고, Transformer만을 사용해 object detection 수행

## DETR의 구조

![DETR](https://user-images.githubusercontent.com/48716219/103141201-2b2b4a00-4731-11eb-9b0a-5b0e8c5b8b0a.png)

- DETR은 Transformer의 Encoder와 Decoder를 사용하여 Object Detection을 수행
- Encoder는 이미지를 Feature Map으로 변환
- Decoder는 Feature Map을 입력으로 받아 Object Detection 수행
- Encoder와 Decoder는 동일한 구조를 사용

## DETR의 Encoder

![DETR Encoder](https://user-images.githubusercontent.com/48716219/103141202-2bc3e080-4731-11eb-9b1a-5b2b2b2b2b2b.png)

## DETR의 Decoder

![DETR Decoder](https://user-images.githubusercontent.com/48716219/103141203-2bc3e080-4731-11eb-9b1b-5b2b2b2b2b2b.png)

## DETR의 Loss

![DETR Loss](https://user-images.githubusercontent.com/48716219/103141204-2bc3e080-4731-11eb-9b1c-5b2b2b2b2b2b.png)

- DETR은 Cross Entropy Loss를 사용하여 Loss를 계산
- Cross Entropy Loss는 Classification Loss와 Bounding Box Regression Loss를 포함
- Classification Loss는 Object가 존재하는지 여부를 판단
- Bounding Box Regression Loss는 Object의 Bounding Box를 예측

## DETR의 성능

![DETR Performance](https://user-images.githubusercontent.com/48716219/103141205-2c5c7700-4731-11eb-9b1d-5b2b2b2b2b2b.png)

- DETR은 SOTA를 달성
