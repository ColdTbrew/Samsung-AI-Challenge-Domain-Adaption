# Samsung_AI_Challenge (segmentation)

[배경]
자율주행은 다양한 센서들을 사용해 주변 상황을 인식하고 이를 바탕으로 차량을 제어하게 됩니다. 

카메라 센서의 경우, 장착 위치, 센서의 종류, 주행 환경 등에 따라 영상간의 격차(Domain Gap)가 발생합니다. 

그간 여러 선행 연구에서는 이미지의 광도와 질감(Photometry and Texture) 격차에 의한 인식 성능 저하를 극복하기 위해, 

Unsupervised Domain Adaptation 기술을 광범위하게 적용해왔습니다. 

하지만 대부분의 기존 연구들은 카메라의 광학적 특성, 

특히 이미지의 왜곡 특성(Geometric Distortion)에 따른 영상간의 격차는 고려하지 않고 있습니다. 

따라서 본 대회에서는 왜곡이 존재하지 않는 이미지(Source Domain)와 레이블을 활용하여, 

왜곡된 이미지(Target Domain)에 대해서도 고성능의 이미지 분할(Semantic Segmentation)을 수행하는 AI 알고리즘 개발을 제안합니다.





[주제]
카메라 특성 변화에 강인한 Domain Adaptive Semantic Segmentation 알고리즘 개발





[설명]
왜곡이 없는(Rectilinear Source Domain) 이미지와 대응되는 레이블 정보를 활용하여, 

레이블이 존재하지 않는 왜곡된 영상(Fisheye* Target Domain)에서도 

강인한 이미지 장면 분할(Semantic Segmentation) 인식을 수행하는 알고리즘 개발

* Fisheye: 200도의 시야각(200° F.O.V)을 가지는 어안렌즈 카메라로 촬영된 이미지



평가 산식 : mIoU (mean Intersection over Union)
IoU = Area of Overlap / Area of Union

각 class마다 Ground Truth와 Prediction의 교집합(Intersection = Area of Overlap)과 합집합(Area of Union)의 평균
Ground Truth와 Prediction에 모두 해당하는 class가 존재하지 않을 경우, mIoU 계산에 해당 경우를 포함하지 않음
Public score : 전체 테스트 데이터 중 약 50%
Private score : 전체 테스트 데이터 중 약 50%

Dataset Info.

train_source_image [폴더]
TRAIN_SOURCE_0000.png ~ TRAIN_SOURCE_2193.png
2048 x 1024


train_source_gt [폴더]
TRAIN_SOURCE_0000.png ~ TRAIN_SOURCE_2193.png
2048 x 1024
픽셀값 0~11(각각 class 0부터 class 11), 255(배경)으로 구성된 Ground Truth 이미지


train_source.csv [파일]
id : 학습데이터 샘플 ID
img_path : 이미지 경로 (상대 경로)
gt_path : Ground Truth 경로 (상대 경로)


train_target_image [폴더]
TRAIN_TARGET_0000.png ~ TRAIN_TARGET__2922.png
1920 x 1080
Fisheye 형태의 Target 이미지


train_target.csv [파일]
id : 학습데이터 샘플 ID
img_path : 이미지 경로 (상대 경로)


val_source_image [폴더]
VALID_SOURCE_000.png ~ VALID_SOURCE_465.png
2048 x 1024
기본적으로는 모델의 source 데이터에 대한 성능 검증데이터
다만, 학습 데이터로 자유롭게 활용 가능


val_source_gt [폴더]
VALID_SOURCE_000.png ~ VALID_SOURCE_465.png
2048 x 1024
픽셀값 0~11(각각 class 0부터 class 11), 255(배경)으로 구성된 Ground Truth 이미지
기본적으로는 모델의 source 데이터에 대한 성능 검증데이터
다만, 학습 데이터로 자유롭게 활용 가능


val_source.csv [파일]
id : 검증데이터 샘플 ID
img_path : 이미지 경로 (상대 경로)
gt_path : Ground Truth 경로 (상대 경로)
기본적으로는 모델의 source 데이터에 대한 성능 검증데이터
다만, 학습 데이터로 자유롭게 활용 가능


test_image [폴더]
TEST_0000.png ~ TEST_1897.png
1920 x 1080


test.csv [파일]
id : 추론데이터 샘플 ID
img_path : 이미지 경로 (상대 경로)


sample_submission.csv [파일] - 제출 양식
id : 추론된 데이터의 각 class의 샘플 ID
mask_rle : RLE로 표현된 이진마스크(class에 해당하는 부분에만 mask존재) 정보
단, 예측 결과가 없는 경우 반드시 -1 처리
또한, 본 대회에서 test_image에 대한 Ground Truth는 크기가 940 x 540으로 조정된 이미지를 사용
따라서, 참가자는 크기가 960 x 540 이미지로 제출물을 구성해야 함
class와 관련된 정보는 공지 참고
관련 유의사항 역시 관련 공지 참고



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
