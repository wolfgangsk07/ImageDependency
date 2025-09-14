import numpy as np
import pandas as pd
from openslide import OpenSlide
from multiprocessing import Pool, Value, Lock
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.io import imsave, imread
from skimage.exposure.exposure import is_low_contrast
from skimage.transform import resize
from scipy.ndimage import binary_dilation, binary_erosion
import logging
import h5py
from tqdm import tqdm
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import json
import random
import string
import time
import timm

from torchvision import transforms
import torchvision.models as models
import numpy as np
import torch
from torchvision.models import ResNet50_Weights
from PIL import Image

from sklearn.cluster import KMeans

from src.agentAttention import AgentAttention


min_Patch=100
    
def get_mask_image(img_RGB, RGB_min=50):
    img_HSV = rgb2hsv(img_RGB)
    # 向量化处理RGB三通道
    backgrounds = [
        img_RGB[..., i] > threshold_otsu(img_RGB[..., i])
        for i in range(3)
    ]
    background_R, background_G, background_B = backgrounds  # 解构赋值
    # 组织区域逻辑优化版本
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    # HSV饱和度通道处理 (保持独立)
    tissue_S = img_HSV[..., 1] > threshold_otsu(img_HSV[..., 1])  # S通道索引为1
    # 最小值约束的三通道统一处理
    min_R, min_G, min_B = [
        img_RGB[..., i] > RGB_min
        for i in range(3)
    ]
    mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return mask

def get_mask(slide, level='max', RGB_min=50):
    #read svs image at a certain level  and compute the otsu mask
    if level == 'max':
        level = len(slide.level_dimensions) - 1
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_RGB = np.transpose(np.array(slide.read_region((0, 0),level,slide.level_dimensions[level]).convert('RGB')),
                           axes=[1, 0, 2])
    tissue_mask = get_mask_image(img_RGB, RGB_min)
    return tissue_mask, level

def extract_patches(slide_path, patch_size,
                       max_patches_per_slide = 50000):

    patches = {}
    #hdf = h5py.File("memfile.h5", 'w',driver="core", backing_store=False)

    try:
        slide = OpenSlide(slide_path)
    except Exception as e:
        raise ValueError(f"Can't open SVS file. Is that a valid SVS?")
    if len(slide.level_dimensions) == 0:
        raise ValueError(f"SVS文件无效，没有图像层级: {slide_path}")
    mask, mask_level = get_mask(slide)
    mask = binary_dilation(mask, iterations=3)
    mask = binary_erosion(mask, iterations=3)

    mask_level = len(slide.level_dimensions) - 1

    PATCH_LEVEL = 0
    BACKGROUND_THRESHOLD = .2


    ratio_x = slide.level_dimensions[PATCH_LEVEL][0] / slide.level_dimensions[mask_level][0]
    ratio_y = slide.level_dimensions[PATCH_LEVEL][1] / slide.level_dimensions[mask_level][1]

    xmax, ymax = slide.level_dimensions[PATCH_LEVEL]

    # handle slides with 40 magnification at base level
    resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
    if not slide.properties.get('.AppMag', 20): print(f"magnifications for {slide_path} is not found, using default magnification 20X")

    patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))
    print(f"patch size for {slide_path}: {patch_size_resized}")

    i = 0
    indices = [(x, y) for x in range(0, xmax, patch_size_resized[0]) for y in
                range(0, ymax, patch_size_resized[0])]

    # here, we generate all the pathes with valid mask
    if max_patches_per_slide is None:
        max_patches_per_slide = len(indices)

    np.random.seed(5)
    np.random.shuffle(indices)

    for x, y in indices:
        # check if in background mask
        x_mask = int(x / ratio_x)
        y_mask = int(y / ratio_y)
        if mask[x_mask, y_mask] == 1:
            patch = slide.read_region((x, y), PATCH_LEVEL, patch_size_resized).convert('RGB')       
            mask_patch = get_mask_image(np.array(patch))
            mask_patch = binary_dilation(mask_patch, iterations=3)
            
            if (mask_patch.sum() > BACKGROUND_THRESHOLD * mask_patch.size) and not (is_low_contrast(patch)):
                if resize_factor != 1.0:
                    patch = patch.resize(patch_size)
                patch = np.array(patch)
                patches[f"{x}_{y}"] = np.array(patch)
                #tile_name = f"{x}_{y}"
                #hdf.create_dataset(tile_name, data=patch)
                i = i + 1
        if i >= max_patches_per_slide:
            break
    if len(patches) <=min_Patch:
        raise ValueError(f"Too little valid tissue was extracted from the SVS file, with fewer than {min_Patch} patches.")
    return(patches)


def normalize_single_file(patches_dict, target_stain_matrix, maxC_ref):
    """标准化内存中的图像字典
    
    参数:
        patches_dict (dict): {坐标: 图像数组}
        target_stain_matrix: 目标染色矩阵
        maxC_ref: 参考最大浓度值
    
    返回:
        dict: 标准化后的图像字典
    """
    normalizer = MacenkoNormalizer()
    normalizer.stain_matrix = target_stain_matrix
    normalizer.maxC_ref = maxC_ref
    
    normalized_dict = {}
    count = 0
    skipped = 0
    
    start_time = time.time()
    
    print(f"开始标准化 {len(patches_dict)} 个图像区块...")
    
    for key, patch in tqdm(patches_dict.items(), desc="标准化图像数据集"):
        try:
            # 跳过空白区域
            mean_val = np.mean(patch)
            if mean_val < 30 or mean_val > 220 or is_low_contrast(patch):
                normalized_dict[key] = patch
                skipped += 1
                continue
            
            # 应用标准化
            normalized = normalizer.transform(patch, target_stain_matrix)
            
            normalized_dict[key] = normalized
            count += 1
        except Exception as e:
            print(f"标准化区块 '{key}' 失败: {str(e)}")
            normalized_dict[key] = patch  # 保留原始作为备用
    
    elapsed = time.time() - start_time
    print(f"标准化完成: 处理 {count} 个区块, 跳过 {skipped} 个区块")
    print(f"总耗时: {elapsed:.2f}秒 (平均 {elapsed/max(1, len(patches_dict)):.4f}秒/图像)")
    
    return normalized_dict

class MacenkoNormalizer:
    """CPU优化的染色标准化器（支持预存向量）"""
    def __init__(self, stain_matrix=None, maxC_ref=None):
        self.stain_matrix = stain_matrix
        self.maxC_ref = maxC_ref
    
    def fit(self, I, beta=0.15, alpha=1):
        """从图像中学习染色向量"""
        try:
            # 确保图像是uint8格式
            if I.dtype != np.uint8:
                I = I.astype(np.uint8)
            
            # 转换图像为OD空间
            OD = self._convert_to_OD(I)
            
            # 去除背景像素
            OD_flat = OD.reshape((-1, 3))
            OD_fg = OD_flat[np.max(OD_flat, axis=1) > beta]
            
            if len(OD_fg) == 0:
                return False
                
            # 计算像素的奇异值分解
            _, eigvecs = np.linalg.eigh(np.cov(OD_fg, rowvar=False))
            
            # 投影到前两个主方向
            proj = np.dot(OD_fg, eigvecs[:, -2:])
            
            # 找到alpha分位数作为染色向量方向
            phi = np.arctan2(proj[:, 1], proj[:, 0])
            min_phi = np.percentile(phi, alpha)
            max_phi = np.percentile(phi, 100 - alpha)
            
            v1 = np.dot(eigvecs[:, -2:], np.array([np.cos(min_phi), np.sin(min_phi)]))
            v2 = np.dot(eigvecs[:, -2:], np.array([np.cos(max_phi), np.sin(max_phi)]))
            
            # 确保向量朝向正确
            if v1[0] < 0: v1 = -v1
            if v2[0] < 0: v2 = -v2
            
            # 构建染色矩阵
            stain_matrix = np.array([v1, v2]).T
            stain_matrix = self._normalize_columns(stain_matrix)
            
            # 计算浓度
            concentrations = self._get_concentrations(OD_fg, stain_matrix)
            
            # 保存结果
            self.stain_matrix = stain_matrix
            self.maxC_ref = np.percentile(concentrations, 99, axis=0)
            return True
        except Exception as e:
            print(f"拟合染色向量失败: {str(e)}")
            return False
    
    def transform(self, I, target_stain_matrix, beta=0.15):
        """应用染色标准化"""
        try:
            # 确保图像是uint8格式
            if I.dtype != np.uint8:
                I = I.astype(np.uint8)
            
            # 转换到OD空间
            OD = self._convert_to_OD(I)
            
            # 创建前景掩码
            mask = np.max(OD, axis=2) > beta
            OD_fg = OD[mask]
            
            if len(OD_fg) == 0:
                return I  # 没有前景，返回原图
            
            # 计算当前浓度
            concentrations = self._get_concentrations(OD_fg, self.stain_matrix)
            
            # 标准化浓度
            maxC = np.percentile(concentrations, 99, axis=0)
            concentrations_norm = concentrations * (self.maxC_ref / maxC)
            
            # 重建图像
            OD_norm = np.zeros_like(OD)
            OD_norm[mask] = np.dot(concentrations_norm, target_stain_matrix.T)
            
            # 转换为RGB
            I_norm = np.exp(-OD_norm) * 255
            I_norm = np.clip(I_norm, 0, 255).astype(I.dtype)
            return I_norm
        except Exception as e:
            print(f"转换失败: {str(e)}")
            return I
    
    def _convert_to_OD(self, I):
        """将RGB转换为光密度(OD)"""
        mask = I == 0
        I = I.copy()  # 避免修改原始数据
        I[mask] = 1  # 避免除以零
        return np.maximum(-np.log(I / 255.0), 1e-6)
    
    def _normalize_columns(self, M):
        """标准化矩阵的列"""
        return M / np.linalg.norm(M, axis=0)
    
    def _get_concentrations(self, OD, stain_matrix):
        """计算染色浓度"""
        # 使用伪逆提高稳定性
        pinv = np.linalg.pinv(stain_matrix)
        return np.dot(OD, pinv.T)


def extract_vit_features(patches_dict, seed=99, device='cuda'):
    """
    从内存中的图像字典提取ViT特征
    
    参数:
        patches_dict (dict): 包含图像数据的字典 {坐标: 图像数组}
        seed (int): 随机种子 (默认99)
        device (str): 计算设备 (默认'cuda')
    返回:
        features_array (np.ndarray): 提取的特征数组
    """
    if len(patches_dict) < 100:
        raise ValueError(f"图像块数量不足，无法进行特征提取: {len(patches_dict)} < 100")
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 初始化ViT模型
    vit_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225)),
    ])
    
    print("加载ViT模型...")
    #vit_model = timm.create_model(
    #    "vit_large_patch16_224",
    #    img_size=224,
    #    patch_size=16,
    #    init_values=1e-5,
    #    num_classes=0,
    #    dynamic_img_size=True
    #)
    #vit_model.load_state_dict(torch.load("/var/www/html/ImageDependency/pyserver/pytorch_model.bin", map_location=device))
    #vit_model = vit_model.to(device).eval()
    
    # 获取所有图像键
    keys = list(patches_dict.keys())
    print(f"共 {len(keys)} 张图像待处理")
    
    # 如果图像过多则采样
    if len(keys) > 50000:
        print(f"图像数量过多({len(keys)})，随机采样50,000张")
        keys = random.sample(keys, 50000)
    
    # 批量处理数据
    features = []
    batch_size = 128
    
    for i in tqdm(range(0, len(keys), batch_size), desc="提取ViT特征"):
        batch_keys = keys[i:i+batch_size]
        batch_images = []
        
        # 加载并预处理批次图像
        for key in batch_keys:
            img = Image.fromarray(patches_dict[key]).convert("RGB")
            batch_images.append(vit_transform(img))
        
        # 转换为tensor并传输到设备
        batch_tensor = torch.stack(batch_images).to(device)
        
        # 提取特征
        with torch.no_grad():
            batch_features = vit_model(batch_tensor).cpu().numpy()
            features.append(batch_features)
    
    # 合并所有特征
    features_array = np.vstack(features)
    print(f"特征提取完成，形状: {features_array.shape}")
    return features_array

def cluster_vit_features(features, cluster_number=900):

    # 读取特征文件
    # 获取特征数据集（假设数据集名为'vit'）
    #features = hd5f['vit'][:]
    n_patches = features.shape[0]
    if n_patches < min_Patch:
        raise ValueError(f"特征数量不足，无法进行聚类: {n_patches} < {min_Patch}")
    # KMeans聚类
    kmeans = KMeans(n_clusters=cluster_number, n_init="auto", random_state=0)
    labels = kmeans.fit_predict(features)
    
    # 计算簇中心特征
    cluster_centers = np.stack([
        features[labels == c].mean(axis=0) 
        for c in range(cluster_number)
    ])
    
    # 保存聚类结果
    #out=h5py.File("kmeansout", 'w',driver="core", backing_store=False)
    #out.create_dataset("KMeans", data=cluster_centers)

    return cluster_centers

def predict_gene_expression(cluster_centers, cancer_type='BRCA'):
    """
    简化版基因表达预测函数
    输入: 
        h5_file - 打开的h5py文件对象 (包含'KMeans'特征组)
        cancer_type - 癌种类型 (默认为'BRCA')
    输出:
        包含5个fold预测结果和集成预测的字典
    """
    # 固定参数设置
    prediction_type = 'Agent'
    extraction_model = 'vit'
    #num_clusters = 900
    num_heads = 16
    depth = 8
    device = torch.device("cuda")
    
    # 从H5文件中读取特征
    features = np.array(cluster_centers)
    num_patches, feature_dim = features.shape
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 模型文件列表 (5个fold)
    model_dir = f"pyserver/{prediction_type}/{cancer_type}"
    model_files = [
        "model_best.pt",
        "model_best_1.pt",
        "model_best_2.pt",
        "model_best_3.pt",
        "model_best_4.pt"
    ]
    first_model_path = f"{model_dir}/{model_files[0]}"
    state_dict = torch.load(first_model_path, map_location=device)
    print(state_dict.keys())
    num_outputs = state_dict['linear_head.3.weight'].shape[0]
    # 创建模型并加载权重
    models = []
    for model_file in model_files:
        model = AgentAttention(
            dim=feature_dim,
            num_patches=num_patches,
            num_outputs=num_outputs,  # 需要从模型获取
            num_heads=num_heads
        ).to(device)
        
        model_path = f"{model_dir}/{model_file}"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
    
    # 预测结果存储
    predictions = {}
    
    # 进行预测
    with torch.no_grad():
        for i, model in enumerate(models):
            pred = model(features_tensor)
            predictions[f'fold{i}'] = pred.squeeze(0).cpu().numpy()
    
    # 计算集成结果 (5个fold的平均)
    all_preds = np.array([pred for pred in predictions.values()])
    ensemble_pred = np.mean(all_preds, axis=0)
    predictions['ensemble'] = ensemble_pred
    
    return predictions








#svsfile=

def process_svs_to_expression(task_dir,f,  cancer_type):
    """
    处理SVS文件并返回基因表达数据
    
    参数:
        svsfile: SVS文件路径
        vectorfile: 向量文件路径(JSON格式)
        cancer_type: 癌症类型，默认为'BRCA'
        patch_size: 图像块大小，默认为(256, 256)
        max_patches: 最大图像块数量，默认为50000
    
    返回:
        exprs: 基因表达预测结果
    """
    svsfile=os.path.join(task_dir,f)
    patch_size=(256, 256)
    max_patches=50000
    vectorfile=f"/var/www/html/ImageDependency/pyserver/jsons/{cancer_type}.json"
    # 读取向量文件
    with open(vectorfile, 'r') as f:
        data = json.load(f)
        stain_matrix = np.array(data['stain_matrix'])
        maxC_ref = np.array(data['maxC_ref'])
    
    # 提取图像块
    patches = extract_patches(svsfile, patch_size, max_patches)
    
    # 检查是否有足够的图像块
    if len(patches) < min_Patch:
        raise ValueError(f"提取的图像块数量不足: {len(patches)} < {min_Patch}")
    
    # 标准化图像块
    normalized_patches = normalize_single_file(patches, stain_matrix, maxC_ref)
    
    # 提取ViT特征
    features = extract_vit_features(normalized_patches)
    
    # 聚类特征
    cluster_centers = cluster_vit_features(features)
    
    # 预测基因表达
    results = predict_gene_expression(cluster_centers, cancer_type)
    
    values=results['ensemble'].astype(float).tolist()
    
    #result_dict = {genename: num for genename, num in zip(genenames, values)}
    return(values)

vit_model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True
    )
vit_model.load_state_dict(torch.load("/var/www/html/ImageDependency/pyserver/pytorch_model.bin", map_location="cuda"))
vit_model = vit_model.to("cuda").eval()



#result=process_svs_to_expression("/var/www/html/ImageDependency/pyserver/test.svs","BRCA")
