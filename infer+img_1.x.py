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
from PIL import Image

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
            
            img_result = show_result_pyplot(
                model, img_path, mask, opacity=0.7, show = False)
            
            #if(int(img_id[5:]) % 2 == 1):  # 홀수만 저장
                
            mmcv.imwrite(img_result, output_path)
                
            mask = mask.pred_sem_seg.data
            mask = torch.squeeze(mask).cpu().numpy()
            mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask)
            mask = mask.resize((960, 540), Image.LANCZOS)
            mask = np.array(mask)
            
            #이미지 저장 부분

            
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
