import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from scipy.stats import t, norm
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
def compute_correlations(y_true, y_pred):
    """计算预测值与真实值之间的相关系数"""
    correlations = []
    for i in range(y_true.shape[1]):
        if np.all(y_true[:, i] == y_true[0, i]) or np.all(y_pred[:, i] == y_pred[0, i]):
            correlations.append(0.0)  # 如果某一列全部相同，相关系数设为0
        else:
            try:
                corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
                correlations.append(corr)
            except:
                correlations.append(0.0)
    return np.mean(correlations)

def smape(y_true, y_pred):
    """计算对称平均绝对百分比误差"""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def evaluate(model, dataloader, run=None, verbose=True, suff='', logger=None):
    """模型评估函数"""
    model.eval()
    # 获取模型所在的设备
    device = next(model.parameters()).device
    loss_fn = nn.MSELoss()
    losses, preds, real, wsis, projs, maes, smapes = [], [], [], [], [], [], []
    
    for image, rna_data, wsi_file_name, tcga_project in tqdm(dataloader, desc="Evaluating"):
        if image.nelement() == 0:  # 使用nelement检查空张量
            continue

        image = image.to(device)
        rna_data = rna_data.to(device)
        wsis.append(wsi_file_name)
        projs.append(tcga_project)

        with torch.no_grad():
            pred = model(image)
        
        preds.append(pred.detach().cpu().numpy())
        loss = loss_fn(pred, rna_data)
        real.append(rna_data.detach().cpu().numpy())
        
        losses.append(loss.detach().cpu().numpy())
        maes.append(mean_absolute_error(rna_data.cpu().numpy(), pred.cpu().numpy()))
        smapes.append(smape(rna_data.cpu().numpy(), pred.cpu().numpy()))
    
    # 计算平均指标
    avg_loss = np.mean(losses) if losses else float('nan')
    avg_mae = np.mean(maes) if maes else float('nan')
    avg_smape = np.mean(smapes) if smapes else float('nan')
    
    # 日志记录
    if run:
        run.log({f'test_loss{suff}': avg_loss})
        run.log({f'test_MAE{suff}': avg_mae})
        run.log({f'test_MAPE{suff}': avg_smape})
    
    if verbose and logger:
        logger.info(f'Test Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, SMAPE: {avg_smape:.4f}')
    
    # 合并结果
    preds = np.concatenate(preds, axis=0) if preds else np.array([])
    real = np.concatenate(real, axis=0) if real else np.array([])
    wsis = np.concatenate(wsis, axis=0) if wsis else np.array([])
    projs = np.concatenate(projs, axis=0) if projs else np.array([])
    
    return preds, real, wsis, projs


def dependent_corr(xy, xz, yz, n, twotailed=True, conf_level=0.95, method='steiger'):
    """
    Calculates the statistic significance between two dependent correlation coefficients
    """
    epsilon = 1e-8
    if method == 'steiger':
        d = xy - xz
        determin = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
        
        # 处理无效的determin值
        if abs(determin) < 1e-8:
            return np.nan, np.nan  # 或返回 (0, 1) 表示无统计显著性
        
        av = (xy + xz)/2
        cube = (1 - yz) * (1 - yz) * (1 - yz)
        
        # 计算分母并确保其为正数
        denominator = ((2 * (n - 1)/(n - 3)) * determin + av * av * cube)
        denominator = max(denominator, epsilon)  # 确保分母不为零
        # 检查分母是否非负
        if denominator <= 0:
            return np.nan, np.nan  # 或返回 (0, 1) 表示无统计显著性
        
        t2 = d * np.sqrt((n - 1) * (1 + yz) / denominator)
        
        # 处理无效的t2值
        if np.isnan(t2) or np.isinf(t2):
            return np.nan, np.nan
        
        p = 1 - t.cdf(abs(t2), n - 3)
        if twotailed:
            p *= 2
        return t2, p
        
    elif method == 'zou':
        L1 = rz_ci(xy, n, conf_level=conf_level)[0]
        U1 = rz_ci(xy, n, conf_level=conf_level)[1]
        L2 = rz_ci(xz, n, conf_level=conf_level)[0]
        U2 = rz_ci(xz, n, conf_level=conf_level)[1]
        rho_r12_r13 = rho_rxy_rxz(xy, xz, yz)
        
        # 添加数值稳定性检查
        xy_minus_L1 = xy - L1
        U2_minus_xz = U2 - xz
        U1_minus_xy = U1 - xy
        xz_minus_L2 = xz - L2
        
        # 计算下界
        term1 = xy_minus_L1**2
        term2 = U2_minus_xz**2
        cross_term = 2 * rho_r12_r13 * xy_minus_L1 * U2_minus_xz
        if term1 + term2 - cross_term < 0:
            lower = np.nan
        else:
            lower = d - np.sqrt(term1 + term2 - cross_term)
        
        # 计算上界
        term3 = U1_minus_xy**2
        term4 = xz_minus_L2**2
        cross_term2 = 2 * rho_r12_r13 * U1_minus_xy * xz_minus_L2
        if term3 + term4 - cross_term2 < 0:
            upper = np.nan
        else:
            upper = d + np.sqrt(term3 + term4 - cross_term2)
        
        return lower, upper
    else:
        raise ValueError('Invalid method specified. Use "steiger" or "zou".')
