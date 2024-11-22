# Exploration 01 Project - tf_flowers : 민혁, 고은비, 임만순

# 요구사항

## 루브릭

![image.png](Exploration%2001%20Project%20-%20tf_flowers%20%E1%84%86%E1%85%B5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%A8,%20%E1%84%80%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%AB%E1%84%87%2014639ab0e159803fb6adeacad73908e0/image.png)

```
평가문항 -> 상세기준
1. base 모델을 활용한 Transfer learning이 성공적으로 진행되었는가? -> VGG16 등이 적절히 활용되었음
2. 학습과정 및 결과에 대한 설명이 시각화를 포함하여 체계적으로 진행되었는가? -> loss, accuracy 그래프 및 임의 사진 추론 결과가 제시됨
3. 분류모델의 test accuracy가 기준 이상 높게 나왔는가? -> test accuracy가 85% 이상 도달하였음
```

## 피어리뷰 템플릿

<aside>
🤔 피어리뷰 템플릿
코더: 고은비, 민 혁, 임만순
리뷰어: 김승기, 김지혜, 이동수


- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요? (완성도)**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 부분의 코드 및 결과물을 캡쳐하여 사진으로 첨부
      ![image](https://github.com/user-attachments/assets/ea44b23b-079c-4a5b-98a3-eefec292fe09)
    

- [x]  **2. 프로젝트에서 핵심적인 부분에 대한 설명이 주석(닥스트링) 및 마크다운 형태로 잘 기록되어있나요? (설명)**
    - [x]  모델 선정 이유
          ![스크린샷 2024-11-22 173313](https://github.com/user-attachments/assets/6166c4cf-0a57-44e1-94d5-cb5a32bb5d2e)

    - [x]  하이퍼 파라미터 선정 이유
          ![image](https://github.com/user-attachments/assets/7602fb5c-951b-4c8a-ba99-681dabfbc991)

    - [x]  데이터 전처리 이유 또는 방법 설명
        

- [x]  **3. 체크리스트에 해당하는 항목들을 수행하였나요? (문제 해결)**
    - [x]  데이터를 분할하여 프로젝트를 진행했나요? (train, validation, test 데이터로 구분)
    - [x]  하이퍼파라미터를 변경해가며 여러 시도를 했나요? (learning rate, dropout rate, unit, batch size, epoch 등)
    - [x]  각 실험을 시각화하여 비교하였나요?
    - [x]  모든 실험 결과가 기록되었나요?

- [x]  **4. 프로젝트에 대한 회고가 상세히 기록 되어 있나요? (회고, 정리)**
    - [x]  배운 점
    - [x]  아쉬운 점
    - [x]  느낀 점
    - [x]  어려웠던 점
        ![image](https://github.com/user-attachments/assets/a2330737-d889-441c-bcb1-ef0486543d9b)


- [x]  **5.  앱으로 구현하였나요?**
    - [x]  구현된 앱이 잘 동작한다.
    - [x]  모델이 잘 동작한다.
          ![image](https://github.com/user-attachments/assets/999dad43-671d-4199-860e-dd4699c5ee97)

# 회고(참고 링크 및 코드 개선)
```
김승기: 오전부터 플러터 앱 연동에 매진했는데 구현에 성공하신 것 보니 부럽네요. 프로젝트 과정을 각 노트북 파일로 나눠 진행 과정을 보기 편하게 확인할 수 있었습니다!
김지혜: 앱 구현하신 것을 보니, 신기했습니다. 손실함수 아니고 실손함수 ㅋㅋㅋ ㅌ
이동수: 각 다른 코드를 보면서 새로운 지식도 습득 할 수 있었습니다. 우리 조는 api 구축과 플러터에 연동하는것에 많은 어려움이 있었는데 구현까지 성공하셔서 시간이 있을때 배울 수 있다면 상당히 이로울 것 같습니다.

```

</aside>

# 개념 정리 (헷갈리는 것들 위주)

- 용어 정리: Dataset, batch size, iteration, epoch

![image.png](Exploration%2001%20Project%20-%20tf_flowers%20%E1%84%86%E1%85%B5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%A8,%20%E1%84%80%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%AB%E1%84%87%2014639ab0e159803fb6adeacad73908e0/image%201.png)

- `VGG16` 모델

![image.png](Exploration%2001%20Project%20-%20tf_flowers%20%E1%84%86%E1%85%B5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%A8,%20%E1%84%80%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%AB%E1%84%87%2014639ab0e159803fb6adeacad73908e0/image%202.png)

- `EfficientNetB0` 모델

![image.png](Exploration%2001%20Project%20-%20tf_flowers%20%E1%84%86%E1%85%B5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%A8,%20%E1%84%80%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%AB%E1%84%87%2014639ab0e159803fb6adeacad73908e0/image%203.png)

- `MobileNetV2` 모델

![image.png](Exploration%2001%20Project%20-%20tf_flowers%20%E1%84%86%E1%85%B5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%A8,%20%E1%84%80%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%AB%E1%84%87%2014639ab0e159803fb6adeacad73908e0/image%204.png)

# Simple CNN 모델 1 : Conv2D - MaxPooling - Flatten - Dense

- 구조
    
    ```python
    model = Sequential([
        Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(160, 160, 3)),
        MaxPooling2D(),
        Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(units=512, activation='relu'),
        Dense(units=2, activation='softmax')
    ])
    ```
    
- 성능
    
    ![image.png](Exploration%2001%20Project%20-%20tf_flowers%20%E1%84%86%E1%85%B5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%A8,%20%E1%84%80%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%AB%E1%84%87%2014639ab0e159803fb6adeacad73908e0/image%205.png)
    
    ```bash
    Epoch 1/20
    92/92 [==============================] - 3s 29ms/step - loss: 0.2653 - accuracy: 0.9264 - val_loss: 1.0021 - val_accuracy: 0.6403
    Epoch 2/20
    92/92 [==============================] - 3s 28ms/step - loss: 0.2032 - accuracy: 0.9554 - val_loss: 1.0769 - val_accuracy: 0.6376
    Epoch 3/20
    92/92 [==============================] - 3s 28ms/step - loss: 0.1679 - accuracy: 0.9595 - val_loss: 1.1234 - val_accuracy: 0.6485
    Epoch 4/20
    92/92 [==============================] - 3s 28ms/step - loss: 0.1425 - accuracy: 0.9676 - val_loss: 1.1217 - val_accuracy: 0.6512
    Epoch 5/20
    92/92 [==============================] - 3s 29ms/step - loss: 0.1000 - accuracy: 0.9796 - val_loss: 1.2000 - val_accuracy: 0.6322
    Epoch 6/20
    92/92 [==============================] - 3s 28ms/step - loss: 0.0878 - accuracy: 0.9789 - val_loss: 1.2242 - val_accuracy: 0.6431
    Epoch 7/20
    92/92 [==============================] - 3s 28ms/step - loss: 0.0671 - accuracy: 0.9888 - val_loss: 1.3881 - val_accuracy: 0.6185
    Epoch 8/20
    92/92 [==============================] - 3s 28ms/step - loss: 0.0652 - accuracy: 0.9840 - val_loss: 1.3388 - val_accuracy: 0.6403
    Epoch 9/20
    92/92 [==============================] - 3s 29ms/step - loss: 0.0497 - accuracy: 0.9881 - val_loss: 1.3922 - val_accuracy: 0.6349
    Epoch 10/20
    92/92 [==============================] - 3s 29ms/step - loss: 0.0348 - accuracy: 0.9942 - val_loss: 1.5233 - val_accuracy: 0.6267
    Epoch 11/20
    92/92 [==============================] - 3s 28ms/step - loss: 0.0360 - accuracy: 0.9918 - val_loss: 1.6395 - val_accuracy: 0.5913
    Epoch 12/20
    92/92 [==============================] - 3s 31ms/step - loss: 0.0437 - accuracy: 0.9881 - val_loss: 1.4766 - val_accuracy: 0.6431
    Epoch 13/20
    92/92 [==============================] - 3s 28ms/step - loss: 0.0420 - accuracy: 0.9911 - val_loss: 1.4814 - val_accuracy: 0.6676
    Epoch 14/20
    92/92 [==============================] - 3s 28ms/step - loss: 0.0257 - accuracy: 0.9935 - val_loss: 1.6044 - val_accuracy: 0.6349
    Epoch 15/20
    92/92 [==============================] - 3s 28ms/step - loss: 0.0268 - accuracy: 0.9928 - val_loss: 1.6142 - val_accuracy: 0.6213
    Epoch 16/20
    92/92 [==============================] - 3s 28ms/step - loss: 0.0255 - accuracy: 0.9915 - val_loss: 1.7299 - val_accuracy: 0.6131
    Epoch 17/20
    92/92 [==============================] - 3s 28ms/step - loss: 0.0261 - accuracy: 0.9935 - val_loss: 1.6693 - val_accuracy: 0.6431
    Epoch 18/20
    92/92 [==============================] - 3s 29ms/step - loss: 0.0196 - accuracy: 0.9942 - val_loss: 1.7468 - val_accuracy: 0.6403
    Epoch 19/20
    92/92 [==============================] - 3s 29ms/step - loss: 0.0215 - accuracy: 0.9942 - val_loss: 1.7597 - val_accuracy: 0.6431
    Epoch 20/20
    92/92 [==============================] - 3s 30ms/step - loss: 0.0147 - accuracy: 0.9959 - val_loss: 1.8493 - val_accuracy: 0.6403
    ```
    

# Simple CNN 모델 2 : dropout 추가

- 구조
    
    ```python
    from tensorflow.keras.layers import Dropout
    
    model = Sequential([
        Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(160, 160, 3)),
        MaxPooling2D(),
        Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.25),  # 드롭아웃 추가
        Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.25),  # 드롭아웃 추가
        Flatten(),
        Dense(units=512, activation='relu'),
        Dropout(0.5),  # 완전 연결층 드롭아웃
        Dense(units=5, activation='softmax')  # 클래스 수 5개
    ])
    
    ```
    
- 성능
    
    ![image.png](Exploration%2001%20Project%20-%20tf_flowers%20%E1%84%86%E1%85%B5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%A8,%20%E1%84%80%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%AB%E1%84%87%2014639ab0e159803fb6adeacad73908e0/image%206.png)
    
    ```bash
    Epoch 1/20
    92/92 [==============================] - 4s 31ms/step - loss: 1.5061 - accuracy: 0.3464 - val_loss: 1.2379 - val_accuracy: 0.5341
    Epoch 2/20
    92/92 [==============================] - 3s 29ms/step - loss: 1.1915 - accuracy: 0.5041 - val_loss: 1.1583 - val_accuracy: 0.5477
    Epoch 3/20
    92/92 [==============================] - 3s 29ms/step - loss: 1.1079 - accuracy: 0.5456 - val_loss: 1.1435 - val_accuracy: 0.5204
    Epoch 4/20
    92/92 [==============================] - 3s 29ms/step - loss: 1.0535 - accuracy: 0.5838 - val_loss: 1.0648 - val_accuracy: 0.6185
    Epoch 5/20
    92/92 [==============================] - 3s 29ms/step - loss: 1.0066 - accuracy: 0.5960 - val_loss: 1.0572 - val_accuracy: 0.6240
    Epoch 6/20
    92/92 [==============================] - 3s 29ms/step - loss: 0.9466 - accuracy: 0.6213 - val_loss: 1.0189 - val_accuracy: 0.6322
    Epoch 7/20
    92/92 [==============================] - 3s 30ms/step - loss: 0.9090 - accuracy: 0.6434 - val_loss: 1.0888 - val_accuracy: 0.5477
    Epoch 8/20
    92/92 [==============================] - 3s 30ms/step - loss: 0.8678 - accuracy: 0.6809 - val_loss: 0.9605 - val_accuracy: 0.6621
    Epoch 9/20
    92/92 [==============================] - 3s 30ms/step - loss: 0.8320 - accuracy: 0.6805 - val_loss: 0.9956 - val_accuracy: 0.6240
    Epoch 10/20
    92/92 [==============================] - 3s 30ms/step - loss: 0.7896 - accuracy: 0.6986 - val_loss: 0.9076 - val_accuracy: 0.6703
    Epoch 11/20
    92/92 [==============================] - 3s 30ms/step - loss: 0.7614 - accuracy: 0.7098 - val_loss: 0.9124 - val_accuracy: 0.6621
    Epoch 12/20
    92/92 [==============================] - 3s 29ms/step - loss: 0.7300 - accuracy: 0.7296 - val_loss: 1.0138 - val_accuracy: 0.5967
    Epoch 13/20
    92/92 [==============================] - 3s 28ms/step - loss: 0.6879 - accuracy: 0.7384 - val_loss: 0.9119 - val_accuracy: 0.6403
    Epoch 14/20
    92/92 [==============================] - 3s 29ms/step - loss: 0.6545 - accuracy: 0.7595 - val_loss: 0.8901 - val_accuracy: 0.6649
    Epoch 15/20
    92/92 [==============================] - 3s 29ms/step - loss: 0.6284 - accuracy: 0.7708 - val_loss: 0.9354 - val_accuracy: 0.6512
    Epoch 16/20
    92/92 [==============================] - 3s 29ms/step - loss: 0.5777 - accuracy: 0.7851 - val_loss: 0.9183 - val_accuracy: 0.6567
    Epoch 17/20
    92/92 [==============================] - 3s 29ms/step - loss: 0.5546 - accuracy: 0.7936 - val_loss: 0.9153 - val_accuracy: 0.6621
    Epoch 18/20
    92/92 [==============================] - 3s 29ms/step - loss: 0.5283 - accuracy: 0.8106 - val_loss: 0.9167 - val_accuracy: 0.6349
    Epoch 19/20
    92/92 [==============================] - 3s 29ms/step - loss: 0.5013 - accuracy: 0.8280 - val_loss: 0.9631 - val_accuracy: 0.6349
    Epoch 20/20
    92/92 [==============================] - 3s 28ms/step - loss: 0.4586 - accuracy: 0.8423 - val_loss: 1.0076 - val_accuracy: 0.6322
    ```
    

# Simple CNN은 너무 구리다!! 빨리 전이학습으로 넘어가자.

- (GPT-4o) 꽃 분류의 베이스 모델을 VGG16 로 가져와서, 전이학습 시킬까 하는데, VGG16 보다 좋은 모델이 있을까?
    
    꽃 분류와 같은 이미지 분류 작업에서 **VGG16**은 안정적이고 간단히 전이 학습을 시도하기에 적합한 모델이야. 하지만 최신 연구에서는 VGG16보다 성능이 더 뛰어나고 경량화된 모델이 많이 등장했어. 아래에서 VGG16과 비교해 더 나은 모델을 추천할게. 😊
    
    ---
    
    ### **VGG16의 장점과 단점**
    
    ### 장점:
    
    1. **안정성**: 잘 설계된 구조로 작은 데이터셋에서도 좋은 성능.
    2. **사용 용이성**: Keras와 같은 라이브러리에서 손쉽게 사용 가능.
    3. **풍부한 전이 학습 사례**: 많은 작업에서 사용된 역사가 있어 적용이 간단.
    
    ### 단점:
    
    1. **모델 크기**: 가중치 파일 크기가 크고, 메모리 사용량이 많음.
    2. **연산량**: Convolution 연산이 많아 학습과 추론 속도가 느림.
    3. **최신 모델 대비 성능 부족**: 최신 경량화 모델보다 성능이 떨어질 수 있음.
    
    ---
    
    ### **추천 모델**
    
    아래는 꽃 분류와 같은 **작은 데이터셋**에서 VGG16을 대체할 수 있는 모델이야.
    
    ### 1. **MobileNetV2**
    
    - **장점**:
        - 경량화된 모델로 **빠른 추론** 가능.
        - 적은 연산량과 메모리 사용량으로 모바일 디바이스에서도 사용 가능.
        - 작은 데이터셋에서도 성능이 좋음.
    - **단점**:
        - VGG16에 비해 학습 초기 성능이 낮을 수 있음.
    
    ```python
    from tensorflow.keras.applications import MobileNetV2
    
    base_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    
    ```
    
    ---
    
    ### 2. **EfficientNet**
    
    - **장점**:
        - **SOTA(State-of-the-Art)** 모델로, 정확도와 효율성을 모두 고려.
        - 다양한 크기 버전(`EfficientNetB0`, `B1`, ...) 제공.
        - 높은 분류 성능을 제공하면서도 연산량이 적음.
    - **단점**:
        - 학습 초기 설정이 까다로울 수 있음.
    
    ```python
    from tensorflow.keras.applications import EfficientNetB0
    
    base_model = EfficientNetB0(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    
    ```
    
    ---
    
    ### 3. **ResNet50**
    
    - **장점**:
        - 깊은 네트워크로 복잡한 데이터에서도 뛰어난 성능.
        - *스킵 연결(skip connection)**을 통해 기울기 소실 문제 해결.
    - **단점**:
        - MobileNet이나 EfficientNet보다 연산량이 많아 속도가 느림.
    
    ```python
    from tensorflow.keras.applications import ResNet50
    
    base_model = ResNet50(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    
    ```
    
    ---
    
    ### 4. **InceptionV3**
    
    - **장점**:
        - 효율적인 모델 구조로 정확도가 높음.
        - 다양한 크기의 필터를 동시에 학습하여 다양한 패턴을 학습 가능.
    - **단점**:
        - MobileNetV2보다 연산량이 많고, 설정이 다소 복잡할 수 있음.
    
    ```python
    from tensorflow.keras.applications import InceptionV3
    
    base_model = InceptionV3(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    
    ```
    
    ---
    
    ### **비교 및 추천**
    
    | 모델 | 정확도 | 연산량 | 모델 크기 | 모바일 친화성 | 학습 난이도 |
    | --- | --- | --- | --- | --- | --- |
    | **VGG16** | 중상 | 높음 | 큼 | 낮음 | 쉬움 |
    | **MobileNetV2** | 중 | 낮음 | 작음 | 높음 | 쉬움 |
    | **EfficientNet** | 상 | 낮음 | 중 | 높음 | 중간 |
    | **ResNet50** | 상 | 중간 | 중 | 중간 | 중간 |
    | **InceptionV3** | 상 | 중간 | 중 | 낮음 | 중간 |
    
    ---
    
    ### **최종 추천**
    
    1. **효율성과 정확도**를 모두 원한다면:
        - **EfficientNetB0**: 경량화와 높은 성능.
    2. **빠르고 경량화된 모델**을 원한다면:
        - **MobileNetV2**.
    3. **더 깊고 강력한 성능**이 필요하다면:
        - **ResNet50** 또는 **InceptionV3**.
    
    ---
    
    ### **전이 학습 예제 코드**
    
    ```python
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    
    # EfficientNetB0 불러오기
    base_model = EfficientNetB0(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Pre-trained weights freezing
    
    # 모델 구성
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')  # 꽃 분류 클래스 수에 맞춤
    ])
    
    # 모델 컴파일
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    ```
    
    ---
    
    ### **결론**
    
    `VGG16`도 나쁘지 않지만, 더 경량화되고 최신 구조인 **`EfficientNetB0`** 또는 **`MobileNetV2`**를 추천할게!
    
    이 모델들은 속도, 정확도, 메모리 사용 면에서 더 나은 선택이 될 가능성이 커. 😊
    

**결론**

`VGG16`도 나쁘지 않지만, 더 경량화되고 최신 구조인 **`EfficientNetB0`** 또는 **`MobileNetV2`**를 추천할게!

이 모델들은 속도, 정확도, 메모리 사용 면에서 더 나은 선택이 될 가능성이 커. 😊

# VGG 16 기반 모델

- 구조
    
    ```bash
    
    ```
    
- 시각화
    
    ![image.png](Exploration%2001%20Project%20-%20tf_flowers%20%E1%84%86%E1%85%B5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%A8,%20%E1%84%80%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%AB%E1%84%87%2014639ab0e159803fb6adeacad73908e0/image%207.png)
    
- 성능
    
    ```bash
    
    ```
    
    ![image.png](Exploration%2001%20Project%20-%20tf_flowers%20%E1%84%86%E1%85%B5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%A8,%20%E1%84%80%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%AB%E1%84%87%2014639ab0e159803fb6adeacad73908e0/image%208.png)
    

# EfficientNetB0 기반 모델

[efficientnetb0_model.h5](Exploration%2001%20Project%20-%20tf_flowers%20%E1%84%86%E1%85%B5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%A8,%20%E1%84%80%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%AB%E1%84%87%2014639ab0e159803fb6adeacad73908e0/efficientnetb0_model.h5)

- 구조
    
    ```python
    # @ EfficientNetB0 
    from tensorflow.keras.applications import EfficientNetB0
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    
    # @ Create the base model from the pre-trained model EfficientNetB0
    base_model_effi = tf.keras.applications.EfficientNetB0(input_shape=IMG_SHAPE,
                                             include_top=False,
                                             weights='imagenet')
                                             
    image_batch.shape
    
    feature_batch = base_model_effi(image_batch) # @ EfficientNetB0
    feature_batch.shape 
    
    base_model_effi.summary() # @ EfficientNetB0
    
    feature_batch.shape
    
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    print("슝~")
    
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)
    
    # dense layer에는 unit과 activation 2개의 매개변수만 사용해주세요.
    # unit의 값은 위에서 global_average_layer를 통과했을 때의 값을 생각해보세요.
    # 활성화 함수는 ReLU를 사용합니다.
    # 
    # units	Positive integer, dimensionality of the output space.
    # activation	Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
    # 
    # [[YOUR CODE]]
    dense_layer = tf.keras.layers.Dense(
        units=feature_batch_average.shape[-1],
        activation='relu'
        )
    
    # unit은 우리가 분류하고 싶은 class를 생각해보세요.
    # 활성화 함수는 Softmax를 사용합니다.
    # 
    # [[YOUR CODE]]
    prediction_layer = tf.keras.layers.Dense(
        units=5,   # @ 2->5로 수정
        activation='softmax'
        )
    
    # feature_batch_averag가 dense_layer를 거친 결과가 다시 prediction_layer를 거치게 되면
    prediction_batch = prediction_layer(dense_layer(feature_batch_average))  
    print(prediction_batch.shape)
    
    base_model_effi.trainable = False  # @ EfficientNetB0
    print("슝~")
    
    model = tf.keras.Sequential([
      base_model_effi,   # @ EfficientNetB0
      global_average_layer,
      dense_layer,
      prediction_layer
    ])
    print("슝~")
    
    model.summary()
    
    base_learning_rate = 0.0001
    # [[YOUR CODE]]
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    
    validation_steps=10   # @ 20 -> 10 으로 수정
    loss0, accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
    
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))
    
    # Q. 직접 모델을 학습하는 코드를 작성하세요.
    
    EPOCHS = 5   # 이번에는 이전보다 훨씬 빠르게 수렴되므로 5Epoch이면 충분합니다.
    # [[YOUR CODE]]
    history = model.fit(train_batches,
                        epochs=EPOCHS,
                        validation_data=validation_batches)
                        
    # @ Epoch 늘려서 (5 + 10)
    EPOCHS = 10
    history = model.fit(train_batches,
                        epochs=EPOCHS,
                        validation_data=validation_batches)
    
    # @ Epoch 늘려서 (5 + 10 +5)
    EPOCHS = 5
    history = model.fit(train_batches,
                        epochs=EPOCHS,
                        validation_data=validation_batches)
    
    ```
    
- 성능
    
    ![image.png](Exploration%2001%20Project%20-%20tf_flowers%20%E1%84%86%E1%85%B5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%A8,%20%E1%84%80%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%AB%E1%84%87%2014639ab0e159803fb6adeacad73908e0/image%209.png)
    

0.00001

```
Epoch 1/5
92/92 [==============================] - 9s 46ms/step - loss: 1.5425 - accuracy: 0.3089 - val_loss: 1.5222 - val_accuracy: 0.3270
Epoch 2/5
92/92 [==============================] - 5s 45ms/step - loss: 1.5463 - accuracy: 0.3055 - val_loss: 1.5231 - val_accuracy: 0.3215
Epoch 3/5
92/92 [==============================] - 5s 46ms/step - loss: 1.5437 - accuracy: 0.3147 - val_loss: 1.5221 - val_accuracy: 0.3188
Epoch 4/5
92/92 [==============================] - 5s 46ms/step - loss: 1.5380 - accuracy: 0.3191 - val_loss: 1.5282 - val_accuracy: 0.3052
Epoch 5/5
92/92 [==============================] - 5s 46ms/step - loss: 1.5417 - accuracy: 0.3174 - val_loss: 1.5172 - val_accuracy: 0.3569

```

0.0001

```
Epoch 1/10
92/92 [==============================] - 5s 46ms/step - loss: 1.5800 - accuracy: 0.2854 - val_loss: 1.6316 - val_accuracy: 0.2643
Epoch 2/10
92/92 [==============================] - 5s 47ms/step - loss: 1.5797 - accuracy: 0.2854 - val_loss: 1.5494 - val_accuracy: 0.3324
Epoch 3/10
92/92 [==============================] - 5s 46ms/step - loss: 1.5768 - accuracy: 0.2864 - val_loss: 1.5819 - val_accuracy: 0.2643
Epoch 4/10
92/92 [==============================] - 5s 47ms/step - loss: 1.5709 - accuracy: 0.2973 - val_loss: 1.6037 - val_accuracy: 0.3215
Epoch 5/10
92/92 [==============================] - 5s 46ms/step - loss: 1.5694 - accuracy: 0.2916 - val_loss: 1.6040 - val_accuracy: 0.2643
Epoch 6/10
92/92 [==============================] - 5s 48ms/step - loss: 1.5714 - accuracy: 0.2926 - val_loss: 1.5247 - val_accuracy: 0.3815
Epoch 7/10
92/92 [==============================] - 5s 47ms/step - loss: 1.5648 - accuracy: 0.2967 - val_loss: 1.5499 - val_accuracy: 0.2888
Epoch 8/10
92/92 [==============================] - 5s 46ms/step - loss: 1.5619 - accuracy: 0.3038 - val_loss: 1.5450 - val_accuracy: 0.2997
Epoch 9/10
92/92 [==============================] - 5s 46ms/step - loss: 1.5603 - accuracy: 0.2909 - val_loss: 1.5603 - val_accuracy: 0.3052
Epoch 10/10
92/92 [==============================] - 5s 46ms/step - loss: 1.5601 - accuracy: 0.2973 - val_loss: 1.5567 - val_accuracy: 0.2807
```

```python
# 레이블 데이터 타입 확인
for images, labels in train_batches.take(1):
    print(labels.dtype)
    print(labels.numpy())
    
# 전처리된 이미지 확인
for images, labels in train_batches.take(1):
    print(images.numpy().min(), images.numpy().max())
    
    
# 데이터셋 구조 확인
print(train_batches.element_spec)

# 클래스별 데이터 수 확인
import numpy as np

label_list = []
for images, labels in train_batches:
    label_list.extend(labels.numpy())

unique, counts = np.unique(label_list, return_counts=True)
print(dict(zip(unique, counts)))
```

![image.png](Exploration%2001%20Project%20-%20tf_flowers%20%E1%84%86%E1%85%B5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%A8,%20%E1%84%80%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%AB%E1%84%87%2014639ab0e159803fb6adeacad73908e0/image%2010.png)

# MobileNetV2 기반 모델

- 구조
    
    ```bash
    
    ```
    
- 성능
    
    ```bash
    
    ```
    

# 최종 모델 선택 : ?

# 최종 모델 개선 시도

## 모델 변경: dropout 추가

## 모델 변경: L2 정규화

## 하이퍼파라미터 변경: Learning rate, …

# 플러터 앱 탑재 시도

# 앱 결과물

# 회고

## 배운 점

- 민혁
    - 데이터셋을 받는 방법 중, `tfds.load( ... , download=True)` 를 사용함으로써 로컬에 받는 게 제일 편하다는 것을 깨달음.
        - 직접 다운받아서 활용하려고 하니, 다음과 같은 에러가 떴다.
            
            `AssertionError: Dataset tf_flowers: could not find data in ../datasets_origin/. Please make sure to call dataset_builder.download_and_prepare(), or pass download=True to tfds.load() before trying to access the tf.data.Dataset object.` 
            
    - 우찬 is GOAT
        
        ![image.png](Exploration%2001%20Project%20-%20tf_flowers%20%E1%84%86%E1%85%B5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%A8,%20%E1%84%80%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%AB%E1%84%87%2014639ab0e159803fb6adeacad73908e0/image%2011.png)
        
- 고은비
    - 모델 학습이 생각보다 어려운 것 같습니다.ㅠㅠ
- 임만순
    - Conv2D와 Dropout을 활용해 기본 CNN 모델을 구성하면서 **과적합 방지**와 **성능 향상**에 대한 중요성을 배움

## 아쉬운 점

- 민혁
    - Colab에서 진행하면 google drive를 활용해야 할텐데, Jupyter Notebook 파일 켤 때마다 drive가 리셋되는 이슈가 있었던 것으로 기억하여, LMS의 Cloud jupyter/shell을 적극 활용하기로 하였다.
        - 그런데, 생각해보니 팀원분들께서 `Git`을 활용하는데에 익숙하지 않고 터미널에도 익숙하지 않았다. 그런데, 여기서 내가 해당 파트에 대한 가이드를 해주는데에 완전 빠져버려서 오전 파트를 너무 낭비한 것 같다.
    - 처음에는 Exploration 01 - 1번 노드 학습 내용을 그대로 가져와서 하나씩 변수를 바꾸는 식으로 무작정 진행하고 있었다.
        - 그런데, 그렇게 진행하다보니
            - model.evalute 시도했을 때 데이터셋 개수 차이 (cat vs dog: 약 23000개, flowers: 약 3000개)에 의한 학습 데이터 부족 이슈가 떠서 해결하느라 잠시 고생했다.
                - `validation_batch = validation_batch.repeat()`로 얼렁뚱땅 해결하고 넘어가려 하자, 바로 다음 `model.fit()`에서 무한 루프 이슈 터짐.
                - 아쉽게도, 개념 학습이 제대로 되어있지 않아서 ‘갯수’ 이슈를 해결하는데에 GPT 도움을 받을 수밖에 없었다.
            - `val_loss: nan` 이 떴다.
                - 알고 보니, `Dense(units=2, activation='softmax')` 를 그대로 쓰고 앉아 있었다. `units=5` 로 해결
            - `VGG16` 또는 기타 발전된 모델이 아닌 그냥 쌩 `Simple CNN`가지고 하이퍼파라미터 수정하면서 성능 개선을 하고 앉아 있었다.
    - 플러터 완성!!!
        
        ![image.png](Exploration%2001%20Project%20-%20tf_flowers%20%E1%84%86%E1%85%B5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%A8,%20%E1%84%80%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%AB%E1%84%87%2014639ab0e159803fb6adeacad73908e0/image%2012.png)
        
    - 근데, 파이썬 서버 모델 로딩 실패로 멸망
        
        ![Screenshot 2024-11-22 at 17.01.37.png](Exploration%2001%20Project%20-%20tf_flowers%20%E1%84%86%E1%85%B5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%A8,%20%E1%84%80%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%AB%E1%84%87%2014639ab0e159803fb6adeacad73908e0/Screenshot_2024-11-22_at_17.01.37.png)
        
- 고은비
    - 이유를 하나씩 찾아가면서 accuracy를 높이려고 하니,, 시간이 부족합니다ㅠㅠ 할 수 있는한 해보았는데…
- 임만순
    - tf_flowers 데이터셋의 각 클래스 분포를 더 면밀히 분석하지 못한 점이 아쉬웠음. 클래스 불균형 여부를 확인하지 않고 학습했기 때문에 데이터 준비 과정에서 추가적인 전처리를 할 기회를 놓침

## 느낀 점

- 민혁
    - 미리 작성된 코드를 따라서 무작정 암기하다 보면, 반드시 놓치는 부분이 생긴다. 직접 코드를 씹뜯맛즐 해보는 타임이 반드시 필요하다.
- 고은비
    - 재미있지만, 어렵다? ㅎㅎ
- 임만순
    - 모델 설계와 튜닝 과정에서 실험의 중요성을 깨달았음. 학습했던 이론을 실제 코드로 구현하면서 많은 세부 사항을 놓쳤으나 이를 실습으로 해결하며 이론과 실전의 균형이 중요하다는 점을 느꼈음

## 어려웠던 점

- 민혁
    - 시간이 많이 부족해서, 모델의 성능을 충분히 끌어올리기엔 역부족이었다.
    - 플러터 앱이 좀 더 짜임새 있게 만들어졌으면 좋았을 것 같다.
- 고은비
    - 플러터앱까지는 시간이 없을 것 같은데,, 노드학습 하는 날 충분히 더 빠르게 진행해할 것 같다.
- 임만순
    - 기본 CNN 모델에서 validation loss가 급격히 증가하는 현상(과적합)을 해결하기 위해 Dropout, L2 정규화 등을 시도했지만, 완벽히 해결하지 못한 점이 힘들었음

## 추가

- 민혁
    - 점심 대충 먹어서 배고프다.
- 고은비
    - ㅎㅎ 마지막 프로젝트가 기대된다… 그때는 다 할 수 있도록..
- 임만순
    - 프로젝트에 맞는 모델을 선택하는 기준(속도, 크기, 성능 등)을 세우는 습관이 필요함
