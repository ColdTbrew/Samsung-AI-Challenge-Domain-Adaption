# Samsung_AI_Challenge (이미지 분할을 위한 도메인 적응)

## Members  
[3!4! Team]  
  [ColdTbrew](https://github.com/ColdTbrew)   
  [BONG-KI](https://github.com/JB0527)

## Index  
 ### 1. Used model (ViT-Adapter)  
  - [Install ViT-Adapter](#install-ViT-Adapter)  
  - [Pretrained Checkpoints](#pretrained-checkpoints)
  - [How to start training](#how-to-start-training)
  - [How to start inference](#how-to-start-inference)
  ### 2. Competition Info
  - [info](#배경)

  
## 배경
    자율주행은 다양한 센서를 활용하여 주변 환경을 감지하고 차량을 제어하는 중요한 기술입니다. 
    카메라 센서는 차량에 다양한 위치와 종류로 장착되며, 주행 환경에 따라 영상 데이터의 차이가 발생합니다.
    과거 연구에서는 광도와 질감의 차이로 인한 문제를 극복하기 위해 자율주행을 위한 기술을 개발해왔습니다.
    그러나 대부분의 연구는 카메라의 광학 특성, 특히 왜곡 문제를 고려하지 않고 있습니다.
    이 대회에서는 왜곡되지 않은 이미지(원본 이미지)와 해당 이미지의 레이블 정보를 사용하여
    왜곡된 이미지(어안 렌즈로 찍은 이미지)에 대한 뛰어난 이미지 분할 알고리즘을 개발하는 것이 목표입니다.

## 주제
    - 카메라 특성 변화에 강인한 도메인 적응 이미지 분할 알고리즘 개발

## 설명
    도메인 적응 알고리즘은 왜곡되지 않은 이미지와 해당 이미지의 레이블 정보를 활용하여,
    왜곡된 이미지에서도 높은 정확도로 이미지 분할을 수행하는 기술입니다.
    특히 어안 렌즈로 찍은 이미지(시야각 200도)에 대한 이미지 분할을 개발하는 것이 주요 과제입니다.

## 평가 산식
    평가 지표는 mIoU (평균 교차 영역 / 평균 합집합)를 사용합니다.
    IoU (교차 영역 / 합집합)는 다음과 같이 계산됩니다.
    각 클래스마다 예측 결과와 실제 결과의 교집합을 계산한 뒤, 전체 합집합으로 나누어 평균을 구합니다.
    만약 예측 결과와 실제 결과 모두 해당 클래스가 존재하지 않는 경우, 이 클래스는 평가에서 제외됩니다.
    
    - Public score: 전체 테스트 데이터 중 약 50%
    - Private score: 전체 테스트 데이터 중 약 50%

## 데이터셋 정보
[데이콘 홈페이지로 대체](https://dacon.io/competitions/official/236132/data)

----------------------------------------------------------
# ViT-Adapter  
## install-ViT-Adapter  
  [Install from ViT repo](ViT-Adapter/segmentation/README.md)  
# pretrained-checkpoints  
  1. best_mIoU_iter_40000_vit_896_13class_a100.pth  
  [Download Link](https://o365inha-my.sharepoint.com/:u:/g/personal/shchoi8687_office_inha_ac_kr/EXDIk_hKSKpGgB_0a8Frtd0BKxBa8o15xWgW2nMLqNmFWw?e=eOaM1H)  
# how-to-start-training  
   ```
   cd ViT-Adapter/segmentation/
   python train.py work_dirs/work_dirs/vit13class.py
   ```
# how-to-start-inference
   ```
   cd ViT-Adapter/segmentation/
   python infer.py
   ```
