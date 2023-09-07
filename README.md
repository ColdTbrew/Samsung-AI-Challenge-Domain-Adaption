
# Samsung_AI_Challenge (이미지 분할을 위한 도메인 적응)

## 배경
자율주행은 다양한 센서를 활용하여 주변 환경을 감지하고 차량을 제어하는 중요한 기술입니다. 

카메라 센서는 차량에 다양한 위치와 종류로 장착되며, 주행 환경에 따라 영상 데이터의 차이가 발생합니다.

과거 연구에서는 광도와 질감의 차이로 인한 문제를 극복하기 위해 자율주행을 위한 기술을 개발해왔습니다.

그러나 대부분의 연구는 카메라의 광학 특성, 특히 왜곡 문제를 고려하지 않고 있습니다.

이 대회에서는 왜곡되지 않은 이미지(원본 이미지)와 해당 이미지의 레이블 정보를 사용하여

왜곡된 이미지(어안 렌즈로 찍은 이미지)에 대한 뛰어난 이미지 분할 알고리즘을 개발하는 것이 목표입니다.

## 주제
카메라 특성 변화에 강인한 도메인 적응 이미지 분할 알고리즘 개발

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
### train_source_image [폴더]
- TRAIN_SOURCE_0000.png ~ TRAIN_SOURCE_2193.png
- 크기: 2048 x 1024

### train_source_gt [폴더]
- TRAIN_SOURCE_0000.png ~ TRAIN_SOURCE_2193.png
- 크기: 2048 x 1024
- 픽셀 값: 0~11 (각각 class 0부터 class 11), 255(배경)

### train_source.csv [파일]
- id: 학습 데이터 샘플 ID
- img_path: 이미지 경로 (상대 경로)
- gt_path: Ground Truth 경로 (상대 경로)

### train_target_image [폴더]
- TRAIN_TARGET_0000.png ~ TRAIN_TARGET__2922.png
- 크기: 1920 x 1080
- 어안 렌즈 형태의 Target 이미지

### train_target.csv [파일]
- id: 학습 데이터 샘플 ID
- img_path: 이미지 경로 (상대 경로)

### val_source_image [폴더]
- VALID_SOURCE_000.png ~ VALID_SOURCE_465.png
- 크기: 2048 x 1024
- 모델 검증 데이터로 사용 가능

### val_source_gt [폴더]
- VALID_SOURCE_000.png ~ VALID_SOURCE_465.png
- 크기: 2048 x 1024
- 픽셀 값: 0~11 (각각 class 0부터 class 11), 255(배경)

### val_source.csv [파일]
- id: 검증 데이터 샘플 ID
- img_path: 이미지 경로 (상대 경로)
- gt_path: Ground Truth 경로 (상대 경로)
- 모델 검증 데이터로 사용 가능

### test_image [폴더]
- TEST_0000.png ~ TEST_1897.png
- 크기: 1920 x 1080

### test.csv [파일]
- id: 추론 데이터 샘플 ID
- img_path: 이미지 경로 (상대 경로)

### sample_submission.csv [파일] - 제출 양식
- id: 추론된 데이터의 각 클래스 샘플 ID
- mask_rle: RLE로 표현된 이진 마스크 (해당 클래스 부분에만 마스크가 존재)
- 예측 결과가 없는 경우 반드시 -1 처리
- 크기가 960 x 540 이미지로 제출

**참고:** test_image에 대한 Ground Truth는 크기가 940 x 540으로 조정된 이미지를 사용하므로, 제출물은 960 x 540 이미지로 구성되어야 합니다.

클래스와 관련된 정보 및 유의사항은 대회 공지사항을 확인하시기 바랍니다.


----------------------------------------------------------
# 현재까지 작업과정

swinV1 - 최고점 crop 1024 기준으로 train 
aug는  crop flip, photometricdistortion, 
test pipeline (멀티샘플링 , 즉 여러가지 스케일로 리사이즈(keep ratio로 비율은 최대한 맞춰줌) 또, 랜덤 flip도 이용하여 tta) 
 
mmseg에서 대부분 가져옴
swin large patch4 (ade20 - 22k) - pretrained model 이용
*score valid 65 csv 47

infer 0.x
infer 1.x 
두가지로 infer함


실험 순서 (pretrain은 city escape)
크롭을 변인변수로 통제해서 씀

1, swin (pre train Ade22k)

2, mask2former (backbone swin) 

3, ocrnet (hrnet)

4, internimage ( dcn ) 

5, vit (transformer) (mapilary -벤츠사진 프리트레인)

- 예정
- 6, mmseg 와 같은 격의 라이브러리 mic -> 도메인어뎁션용 라이브러리 (gta to cityescape 등) -> 꽤나 성능이 좋아보임 paperswithcode 기준// 타겟어노테이션을 사용하지않고 예측 고로 가장 대회 목적에 가까운 것 같음 //

infer 과정 시각화를 보고 든 생각, 마스킹은 잘됨 문제는 오버해서 마스킹이 되거나, 찌그러진 폴대를 못잡거나(그냥 나무나 구조물에 같이 마스킹 된 경우도 있어)서 train의 iou는 60~70점대로 높은데 , 제출한 점수는 낮은 것 같음.

# 작업예정 1
뒷 범퍼가 여러 다른 클래스로 인지하게 끔 잡혀서 (ex construction 등등) iou가 되게 낮게 잡히는 경우가 있음

*** -> 해결방안
1, 테스트이미지를 몇가지 표본으로 만들어서 테스트이미지를 후처리 후 infer과정에서 적용시켜 동그란 fish데이터에 상응하는 뒷범퍼 이미지를 마스크로 적용
-> data leakage

2, 13번째 class label 을 만들어서 아예 255 로 해버린 뒤 gt처럼 학습이 안되게끔 만들어놓기 
-> 그러나 train은 안되지만, 예측추론과정에서 13번째 class를 골라낼 수 있으므로 문제가 생길 수 있음

# 작업예정 2
만약 target 이미지를 학습할 수 있다면?

3, 학습이미지를 쓴 모델에 타겟이미지 학습을 위해 loadfrom으로 sudo 하고, target image를 쓸 수 있다면 target 이미지의 뒷 범퍼 마스킹한걸 새로운 클래스(뒷범퍼 클래스)로 추가해서 사용

# 작업예정 2
pole 과 같은 되게 얇고 긴 막대기 처럼 생긴 것들은 오히려 왜곡이 많이 일어나 인식이 안됨 (특히 나무나 건물에 같이 휩쓸려서 인식되기도 함)

-> crop을 많이한 후, resize를 하여 좀 더 



*** 질문 1
infer 할 때는 resize 손실로 인해 valid 점수가 낮은걸까요?(nearest 손실이 있을까?)

*** 질문 2
train 아웃풋에다 crf 적용?  어떨까요? dense crf ? (코드구현 잘못해서인지 점수가 너무 낮음)

*** 질문 3
infer할 때, 이미지 사이즈가 클수록 좋을까요? 이게영향이 있을까요?

*** 질문 4 
도메인 어답션, 이것이 정답일까요...? 그냥 작업 예정 2처럼 타겟이미지를 라벨링(anno파일 만들어서 ex-(어떠한 라이브러리를 쓰거나 , 따로 우리가 inference를 해서 마스크만 가져와서, 마스크 가장자리를 다른 클래스로 추가하고 사용하는,, 느낌,,,)) 쓰면 안될까요 (ex. 아니면 semi-supervised를 이용하면 수도라벨링.... 활용)

*** 질문 5
vit adapter 에 맵플러리 pretrained (transformer 백본) 기준에서 4k 랑 16k가 스코어를 보면 4k가 더높다 과적ㅎ합일까? loss를 바꿔야할까?
- 학습이 기본이미지에 충실해지다보니 TEST인 왜곡이미지에는 오히려 점수가 낮아짐이라고 생각함
