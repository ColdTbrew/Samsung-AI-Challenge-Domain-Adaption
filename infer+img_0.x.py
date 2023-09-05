import os
import mmcv
import torch
from tqdm import tqdm
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.models import build_segmentor
import pandas as pd
import numpy as np
import json
import numpy as np
from PIL import Image, ImageFilter
import mmcv_custom   # noqa: F401,F403
import mmseg_custom 
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

def main():
    #----------------------------------------------------------------------
    csv_save_dir = "work_dirs/infer/inter_xl_18k_i6590"
    file_name = 'inter_xl_18k_i6590'
    img_output_folder = 'work_dirs/infer/inter_xl_18k_i6590'
    config_file = 'work_dirs/intern_XL/intern_XL_for_infer.py'
    checkpoint_file = 'work_dirs/intern_XL/best_mIoU_iter_18000.pth'
    #-----
    sample_path = "C:/Users/ADMIN/Projects/JBK/mmseg_Data/open/test.csv"
    test_image_path = "C:/Users/ADMIN/Projects/JBK/mmseg_Data/open/test_image"
    submit_path = "C:/Users/ADMIN/Projects/JBK/mmseg_Data/open/sample_submission.csv"
    #----------------------------------------------------------------------
    device = "cuda"
    
    model = init_segmentor(config_file, checkpoint_file, device)
    data = pd.read_csv(sample_path)['id'].values.tolist()

    with torch.no_grad():
        model.eval()
        result = []

        for img_id in tqdm(data):
            img_path = os.path.join(test_image_path, img_id + ".png")
            masks_list = inference_segmentor(model, img_path) #리스트로 된 마스크
            #이미지 저장 부분
            output_path = f'{img_output_folder}/infer_{img_id}_mask.png' 
            os.makedirs(img_output_folder, exist_ok=True)
            img_result = model.show_result(img_path, masks_list, palette=None, show=False, opacity=0.7)
            
            #if(int(img_id[5:])%2==1): #홀수만 저장
            mmcv.imwrite(img_result, output_path) 
                
            for mask in masks_list:
                #resize
                mask = mask.astype(np.uint8)
                mask_img = Image.fromarray(mask)
                mask_img = mask_img.resize((960, 540), resample=Image.LANCZOS)

                mask = np.array(mask_img)
                for class_id in range(12):
                    class_mask = (mask == class_id).astype(np.uint8)
                    if np.sum(class_mask) > 0:  # If mask exists, encode
                        mask_rle = rle_encode(class_mask)
                        result.append(mask_rle)

                    else:  # If mask doesn't exist, append -1
                        result.append(-1)
    
    submit = pd.read_csv(submit_path)
    submit['mask_rle'] = result
    if not os.path.exists(csv_save_dir):
        os.makedirs(csv_save_dir)
    submit.to_csv(os.path.join(csv_save_dir, file_name + '.csv'), index=False)

main()
