import os
import mmcv
import torch
from tqdm import tqdm
from mmseg.apis import init_model, inference_model,show_result_pyplot
from mmseg.models import build_segmentor
import pandas as pd
import numpy as np
import json
import numpy as np
from PIL import Image, ImageDraw

import cv2
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import mmcv
import matplotlib.pyplot as plt

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
    
def apply_ellipse_filter(input_image):
    # 이미지 크기 및 배경색 설정
    width, height = input_image.size
    background_color = (255, 255, 255)  # 배경색을 흰색으로 설정

    # 새로운 이미지 생성 (알파 채널 포함)
    new_image = Image.new("RGBA", (width, height), background_color)
    draw = ImageDraw.Draw(new_image)

    # 타원 크기 및 위치 설정
    ellipse_width = 1850//2
    ellipse_height = 1500//2
    ellipse_left = (width - ellipse_width) // 2
    ellipse_top = (height - ellipse_height) // 2
    ellipse_right = ellipse_left + ellipse_width
    ellipse_bottom = ellipse_top + ellipse_height - 100

    # 타원 그리기 (알파 채널 사용)
    ellipse_color = (0, 0, 0, 0)  # 흰색 (RGB) 및 완전 불투명 (알파 채널)
    draw.ellipse((ellipse_left, ellipse_top, ellipse_right, ellipse_bottom), fill=ellipse_color)

    # 입력 이미지와 새로운 이미지 합치기 (타원 외부는 투명, 내부는 그대로)
    result_image = Image.alpha_composite(new_image, input_image.convert("RGBA"))

    # 이미지를 OpenCV 형식으로 변환
    result_cv2 = np.array(result_image)

    # 타원 외부 영역의 픽셀 값을 흰색으로 설정
    mask = np.zeros_like(result_cv2)  # 같은 크기의 빈 이미지 생성
    cv2.ellipse(mask, ((ellipse_left + ellipse_right) // 2, (ellipse_top + ellipse_bottom) // 2),
                (ellipse_width // 2, ellipse_height // 2), 0, 0, 360, (255, 255, 255), -1)  # 타원 내부 채우기
    result_cv2[mask == 0] = 255  # 타원 외부 픽셀 값을 흰색(255)으로 설정

    # 그레이 스케일로 변환
    result_cv2 = cv2.cvtColor(result_cv2, cv2.COLOR_RGBA2GRAY)

    # OpenCV 형식 이미지를 PIL 형식으로 변환
    result_image = Image.fromarray(result_cv2)

    return result_image
def main():
    #----------------------------------------------------------------------
    save_dir = "work_dirs/infer"
    file_name = 'swin_v2_1024_best'
    config_file = 'workdir/swin_v2/swin_v2_1024.py'
    checkpoint_file = 'workdir/swin_v2/savepth/best_mIoU_iter_32000.pth'
    sample_path = "C:/Users/ADMIN/Projects/JBK/mmseg_Data/open/test.csv"
    test_image_path = "C:/Users/ADMIN/Projects/JBK/mmseg_Data/open/test_image"
    submit_path = "C:/Users/ADMIN/Projects/JBK/mmseg_Data/open/sample_submission.csv"
    output_folder = 'work_dirs/swin_v2/infered_img'
    #----------------------------------------------------------------------
    device = "cuda"
    
    model = init_model(config_file, checkpoint_file, device)
    data = pd.read_csv(sample_path)['id'].values.tolist()

    with torch.no_grad():
        model.eval()
        result = []
        print(output_folder)
        for img_id in tqdm(data):
            img_path = os.path.join(test_image_path, img_id + ".png")
            mask = inference_model(model, img_path)
            
            output_path = f'{output_folder}/infer_{img_id}_mask.png' 
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            img_result = show_result_pyplot(model, img_path, mask, opacity=0.7, show = False)         
            mmcv.imwrite(img_result, output_path)
                
            mask = mask.pred_sem_seg.data
            mask = torch.squeeze(mask).cpu().numpy()
            mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask)
            mask = mask.resize((960, 540), Image.NEAREST)
            mask = apply_ellipse_filter(mask)
            mask = np.array(mask)
            
            for class_id in range(12):
                class_mask = (mask == class_id).astype(np.uint8)
                if np.sum(class_mask) > 0:  # 마스크가 존재하는 경우 encode
                    mask_rle = rle_encode(class_mask)
                    result.append(mask_rle)
                else:  # 마스크가 존재하지 않는 경우 -1
                    result.append(-1)
         
            

    submit = pd.read_csv(submit_path)
    submit['mask_rle'] = result
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    submit.to_csv(os.path.join(save_dir, file_name + '.csv'), index=False)

main()