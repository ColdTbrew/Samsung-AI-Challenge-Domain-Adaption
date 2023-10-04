# Samsung_AI_Challenge (이미지 분할을 위한 도메인 적응)
![대회정보](./pngs/dacon.png)  

## Members  
### [3!4! Team]
  [ColdTbrew](https://github.com/ColdTbrew)   
  [BONG-KI](https://github.com/JB0527)

## Index  
 ### 1. Used model (ViT-Adapter)  
  - [Install ViT-Adapter](#install-ViT-Adapter)  
  - [Pretrained Checkpoints](#pretrained-checkpoints)
  - [How to start training](#how-to-start-training)
  - [How to start inference](#how-to-start-inference)  
  - [Download logs](#Download-logs)
  ### 2. Competition Info
  - [info](#배경)
  ### 3. 사용환경  
  - [Environment](#Environment)
    


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
  ### reqirement
  ```
    # recommended environment: torch1.9 + cuda11.1
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
    pip install timm==0.4.12
    pip install mmdet==2.22.0 # for Mask2Former
    pip install mmsegmentation==0.20.2
    ln -s ../detection/ops ./
    cd ops & sh make.sh # compile deformable attention  
  ```
  ### extra install  
  ```
   cd ViT-Adapter/segmentation/
   pip install scipy
   pip install -r requirement.txt
  ```
# pretrained_pths  
download to 'ViT-Adapter/segmentation/pretrained'  
  [beit_large_patch16_224_pt22k_ft22k.pth](https://o365inha-my.sharepoint.com/:u:/g/personal/shchoi8687_office_inha_ac_kr/EQHoIHanx1xApKUIvWvo1L0Byg6Cjx7NkG4W4iPb7pu2LQ?e=xgeQn5)  
  
# best-miou-checkpoints  
1. best_mIoU_iter_40000_vit_896_13class_a100.pth
   download to 'ViT-Adapter/segmentation/work_dirs'  
  [best_mIoU_iter_40000_vit_896_13class_a100.pth](https://o365inha-my.sharepoint.com/:u:/g/personal/shchoi8687_office_inha_ac_kr/EXDIk_hKSKpGgB_0a8Frtd0BKxBa8o15xWgW2nMLqNmFWw?e=eOaM1H)  
# how-to-start-training  
   ```
   cd ViT-Adapter/segmentation/
   python train.py work_dirs/vit_13class.py
   ```
# how-to-start-inference
   ```
   cd ViT-Adapter/segmentation/
   python 0.x_infer+img+filter.py
   ```
# Download-logs  
 [20230928_084831.log Link](https://o365inha-my.sharepoint.com/:u:/g/personal/shchoi8687_office_inha_ac_kr/Ebr5V8U8ejRGhyB0Svu0Ck8B39TiTHr6ZIbYmo6A_reiGA?e=MoEYVj)

# Dataset 
 [prepare dataset](data_preprocessing/data/open)  
    download_open.zip and unzip to this path : data_preprocessing/data/open
 
    ├── data
    │   ├── open
    │   │   ├── train_img
    │   │   │    ├── TRAIN_SOURCE_0000.png
    │   │   ├── valid_img
    │   │   │    ├── VALID_SOURCE_0000.png
    │   │   ├── train_img_anno
    │   │   │    ├── TRAIN_SOURCE_0000.png
    │   │   ├── valid_img_anno
    │   │   │    ├── VALID_SOURCE_0000.png
    │   ├── 13class_dataset
    │   │   ├── train_img
    │   │   │    ├── TRAIN_SOURCE_0000.png
    │   │   ├── valid_img
    │   │   │    ├── VALID_SOURCE_0000.png
    │   │   ├── train_img_anno
    │   │   │    ├── TRAIN_SOURCE_0000.png
    │   │   ├── valid_img_anno
    │   │   │    ├── VALID_SOURCE_0000.png


# Environment  

```

sys.platform: linux
Python: 3.8.0 (default, Nov  6 2019, 21:49:08) [GCC 7.3.0]
CUDA available: True
GPU 0,1: A100-SXM4-40GB
CUDA_HOME: /usr/local/cuda
NVCC: Build cuda_11.0_bu.TC445_37.28845127_0
GCC: gcc (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
PyTorch: 1.9.0+cu111
PyTorch compiling details: PyTorch built with:
- GCC 7.3
- C++ Version: 201402
- Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for Intel(R) 64 architecture applications
- Intel(R) MKL-DNN v2.1.2 (Git Hash 98be7e8afa711dc9b66c8ff3504129cb82013cdb)
- OpenMP 201511 (a.k.a. OpenMP 4.5)
- NNPACK is enabled
- CPU capability usage: AVX2
- CUDA Runtime 11.1
- NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86
- CuDNN 8.0.5
- Magma 2.5.2
- Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.1, CUDNN_VERSION=8.0.5, CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.9.0, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, 
  
TorchVision: 0.10.0+cu111
OpenCV: 4.8.0
MMCV: 1.4.2
MMCV Compiler: GCC 7.3
MMCV CUDA Compiler: 11.1
MMSegmentation: 0.20.2+e1afc82
```


