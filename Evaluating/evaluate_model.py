import pandas as pd
import numpy as np
import pickle as pl
import os
import sys
import argparse
import logging
import gc
import psutil
import warnings
import time
from datetime import datetime
from scipy import stats
from scipy.stats import t
from sklearn.metrics import mean_squared_error
from statsmodels.stats.multitest import fdrcorrection

# 忽略所有来自pickle模块的NumPy弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, 
                        message=".*numpy.core.numeric is deprecated.*")

# 减少NumPy内存占用
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
np.set_printoptions(precision=4)

# 优化后的相关系数计算函数
def optimized_pearsonr(x, y):
    """
    高度优化的Pearson相关系数计算
    避免创建中间对象，最小化内存使用
    """
    # 检查是否为常数序列
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0, 1.0
    
    n = len(x)
    if n < 2:
        return 0.0, 1.0
    
    # 使用数值稳定的计算方法
    mx = np.mean(x)
    my = np.mean(y)
    
    # 中心化数据
    xm = x - mx
    ym = y - my
    
    # 计算协方差和方差
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
    
    # 处理除零错误
    if r_den < 1e-12:
        return 0.0, 1.0
    
    r = r_num / r_den
    
    # 计算p值 (使用t分布近似)
    if abs(r) == 1.0:
        p = 0.0
    else:
        # 添加数值稳定性检查
        denom = max(1 - r**2, 1e-8)
        df = max(n - 2, 1)
        t_stat = r * np.sqrt(df / denom)
        p = 2 * (1 - t.cdf(abs(t_stat), df=df))
    
    return r, min(max(p, 0.0), 1.0)

def dependent_corr(xy, xz, yz, n, twotailed=True, conf_level=0.95, method='steiger'):
    """
    Calculates the statistic significance between two dependent correlation coefficients
    with enhanced numerical stability
    """
    # 添加数值稳定性检查
    epsilon = 1e-8
    xy = np.clip(xy, -1 + epsilon, 1 - epsilon)
    xz = np.clip(xz, -1 + epsilon, 1 - epsilon)
    yz = np.clip(yz, -1 + epsilon, 1 - epsilon)
    
    if method == 'steiger':
        d = xy - xz
        determin = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
        
        # 处理无效的determin值
        if abs(determin) < 1e-8:
            return np.nan, np.nan
        
        av = (xy + xz) / 2
        cube = (1 - yz) * (1 - yz) * (1 - yz)
        
        # 计算分母并确保其为正数
        denominator = ((2 * (n - 1) / max(n - 3, 1)) * determin + av * av * cube)
        denominator = max(denominator, epsilon)
        
        # 检查分母是否非负
        if denominator <= 0:
            return np.nan, np.nan
        
        t2 = d * np.sqrt(max(n - 1, 1) * (1 + yz) / denominator)
        
        # 处理无效的t2值
        if np.isnan(t2) or np.isinf(t2):
            return np.nan, np.nan
        
        p = 1 - t.cdf(abs(t2), max(n - 3, 1))
        if twotailed:
            p *= 2
        return t2, p
        
    elif method == 'zou':
        # 简化的Zou方法实现
        d = xy - xz
        rho_r12_r13 = ((yz - 0.5 * xy * xz) * (1 - xy**2 - xz**2 - yz**2) +
                      xy * xz * (1 - xy**2 - xz**2 - yz**2)) / ((1 - xy**2) * (1 - xz**2))
        
        # 计算方差
        var_xy = (1 - xy**2)**2 / max(n - 1, 1)
        var_xz = (1 - xz**2)**2 / max(n - 1, 1)
        cov_xy_x极 = rho_r12_r13 * np.sqrt(var_xy * var_xz)
        
        # 计算置信区间
        se_diff = np.sqrt(var_xy + var_xz - 2 * cov_xy_xz)
        z = stats.norm.ppf(1 - (1 - conf_level) / 2)
        
        lower = d - z * se_diff
        upper = d + z * se_diff
        
        return lower, upper
    else:
        raise ValueError('Invalid method specified. Use "steiger" or "zou".')

def log_memory():
    """记录当前内存使用情况"""
    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        return f"内存使用: {mem.rss/(1024**2):.2f} MB (RSS), {mem.vms/(1024**2):.2f} MB (VMS)"
    except:
        return "内存信息不可用"

def setup_analysis_logger(save_path, cancer_type, prediction_type):
    """
    设置分析日志记录器
    """
    # 创建日志文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_path, f"{cancer_type}_{prediction_type}_analysis_{timestamp}.log")
    
    # 创建日志记录器
    logger = logging.getLogger('analysis_logger')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

def analyze_gene_expression(cancer_type, prediction_type, extraction_model, cluster_type, num_folds, module_path, base_dir):
    """
    优化后的基因表达预测结果分析函数
    """
    # 设置模块路径和当前工作目录
    sys.path.insert(0, module_path)
    os.chdir(base_dir)

    # 设置结果保存路径
    results_dir = os.path.join(
        os.path.join(base_dir, "data/result/model/"), 
        extraction_model,
        prediction_type, 
        cancer_type
    )
    save_path = os.path.join(results_dir, "result")
    os.makedirs(save_path, exist_ok=True)
    
    # 设置日志记录器
    logger, log_file = setup_analysis_logger(save_path, cancer_type, prediction_type)
    logger.info(f"开始分析: 癌症类型={cancer_type}, 预测模型={prediction_type}")
    logger.info(f"特征提取模型={extraction_model}, 聚类方法={cluster_type}")
    logger.info(f"日志文件: {log_file}")
    #logger.info(f"当前内存状态: {log_memory()}")

    # 加载测试结果数据
    results_file = os.path.join(results_dir, 'test_results.pkl')
    try:
        # 忽略pickle加载时的NumPy弃用警告
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning,
                                    message=".*numpy.core.numeric is deprecated.*")
            with open(results_file, 'rb') as f:
                test_results = pl.load(f)
        logger.info(f"成功加载测试结果文件: {results_file}")
        #logger.info(f"加载后内存状态: {log_memory()}")
    except Exception as e:
        logger.error(f"加载测试结果文件失败: {e}")
        return

    # 初始化数据结构
    gene_names = test_results['genes']
    real_dfs, pred_dfs, random_dfs = [], [], []
    wsi_samples = []

    logger.info(f"处理 {num_folds} 折交叉验证数据...")
    
    # 处理每个fold的结果数据
    for fold in range(num_folds):
        fold_data = test_results[f'split_{fold}']
        sample_ids = fold_data['wsi_file_name']
        
        # 收集当前fold的数据并转换为float32节省内存
        wsi_samples.extend(sample_ids)
        real_dfs.append(
            pd.DataFrame(
                fold_data['real'].astype(np.float32), 
                index=sample_ids, 
                columns=gene_names
            )
        )
        pred_dfs.append(
            pd.DataFrame(
                fold_data['preds'].astype(np.float32), 
                index=sample_ids, 
                columns=gene_names
            )
        )
        random_dfs.append(
            pd.DataFrame(
                fold_data['random'].astype(np.float32), 
                index=sample_ids, 
                columns=gene_names
            )
        )

    # 合并所有fold的数据
    df_real = pd.concat(real_dfs)
    df_pred = pd.concat(pred_dfs)
    df_random = pd.concat(random_dfs)
    
    # =============== 新增部分：保存真实值和预测值矩阵到CSV ===============
    logger.info("保存真实值和预测值矩阵到CSV文件...")
    real_matrix_file = os.path.join(save_path, "real_expression_matrix.csv")
    pred_matrix_file = os.path.join(save_path, "pred_expression_matrix.csv")
    
    # 使用分块写入避免内存问题
    try:
        df_real.to_csv(real_matrix_file, chunksize=10000)
        logger.info(f"真实值矩阵已保存到: {real_matrix_file}")
    except Exception as e:
        logger.error(f"保存真实值矩阵失败: {e}")
    
    try:
        df_pred.to_csv(pred_matrix_file, chunksize=10000)
        logger.info(f"预测值矩阵已保存到: {pred_matrix_file}")
    except Exception as e:
        logger.error(f"保存预测值矩阵失败: {e}")
        
    # 释放不再需要的内存
    del real_dfs, pred_dfs, random_dfs, test_results
    gc.collect()
    
    #logger.info(f"合并数据后内存状态: {log_memory()}")
    logger.info(f"真实数据形状: {df_real.shape}, 预测数据形状: {df_pred.shape}, 随机数据形状: {df_random.shape}")

    # 验证样本索引一致性
    if not (np.all(df_real.index == df_pred.index) and np.all(df_real.index == df_random.index)):
        logger.warning("样本索引不一致，重新对齐数据")
        df_pred = df_pred.reindex(df_real.index)
        df_random = df_random.reindex(df_real.index)

    # 初始化结果存储
    results = {
        'pred_real_r': [],   # 预测值与真实值的相关系数
        'random_real_r': [],  # 随机值与真实值的相关系数
        'pearson_p': [],      # Pearson相关系数的p值
        'Steiger_p': [],      # Steiger检验的p值
        'rmse_pred': [],      # 预测值的RMSE
        'rmse_random': [],    # 随机值的RMSE
        'rmse_quantile_norm': [],  # 基于四分位距归一化的RMSE
        'rmse_mean_norm': [],      # 基于均值归一化的RMSE
        'gene': []            # 基因名称
    }

    logger.info(f"开始分析 {len(gene_names)} 个基因的预测结果...")
    
    # 高效常数序列检查函数
    def is_constant(arr):
        """高效检查数组是否为常数序列"""
        return np.all(arr == arr[0])
    
    # 分块处理参数
    CHUNK_SIZE = 100  # 分块大小
    total_genes = len(gene_names)
    processed_genes = 0
    
    # 添加中断处理标志
    interrupted = False
    
    try:
        # 分块处理基因
        for chunk_idx in range(0, total_genes, CHUNK_SIZE):
            chunk_start = chunk_idx
            chunk_end = min(chunk_idx + CHUNK_SIZE, total_genes)
            chunk_genes = gene_names[chunk_start:chunk_end]
            
            # 添加进度显示
            progress = (chunk_start / total_genes) * 100
            #logger.info(f"处理基因块 {chunk_start+1}-{chunk_end}/{total_genes} - 进度: {progress:.1f}% - {log_memory()}")
            
            for gene in chunk_genes:
                try:
                    # 提取基因数据
                    real_vals = df_real[gene].values
                    pred_vals = df_pred[gene].values
                    random_vals = df_random[gene].values
                    
                    # 检查是否为常数序列
                    const_check = (
                        is_constant(real_vals) or 
                        is_constant(pred_vals) or 
                        is_constant(random_vals)
                    )
                    
                    # 计算相关系数和p值
                    if const_check or len(real_vals) < 3:
                        pred_r, random_r = 0.0, 0.0
                        pearson_p, steiger_p = 1.0, 1.0
                        pred_random_r = 0.0
                    else:
                        # 使用优化的pearsonr计算
                        pred_r, pearson_p = optimized_pearsonr(real_vals, pred_vals)
                        random_r, _ = optimized_pearsonr(real_vals, random_vals)
                        pred_random_r, _ = optimized_pearsonr(pred_vals, random_vals)
                        
                        # Steiger检验
                        _, steiger_p = dependent_corr(
                            pred_r, random_r, pred_random_r, 
                            len(real_vals), twotailed=False, 
                            conf_level=0.95, method='steiger'
                        )
                    
                    # 计算RMSE指标
                    rmse_pred = np.sqrt(mean_squared_error(real_vals, pred_vals))
                    rmse_random = np.sqrt(mean_squared_error(real_vals, random_vals))
                    
                    # 归一化RMSE
                    q1, q3 = np.quantile(real_vals, 0.25), np.quantile(real_vals, 0.75)
                    rmse_qnorm = rmse_pred / (q3 - q1 + 1e-5)
                    rmse_mnorm = rmse_pred / (np.mean(real_vals) + 1e-5)
                    
                    # 存储结果
                    results['pred_real_r'].append(pred_r)
                    results['random_real_r'].append(random_r)
                    results['pearson_p'].append(pearson_p)
                    results['Steiger_p'].append(steiger_p)
                    results['rmse_pred'].append(rmse_pred)
                    results['rmse_random'].append(rmse_random)
                    results['rmse_quantile_norm'].append(rmse_qnorm)
                    results['rmse_mean_norm'].append(rmse_mnorm)
                    results['gene'].append(gene)
                    
                    processed_genes += 1
                    
                except Exception as e:
                    logger.error(f"处理基因{gene}时出错: {str(e)}")
                    # 添加默认值防止崩溃
                    results['pred_real_r'].append(0.0)
                    results['random_real_r'].append(0.0)
                    results['pearson_p'].append(1.0)
                    results['Steiger_p'].append(1.0)
                    results['rmse_pred'].append(0.0)
                    results['rmse_random'].append(0.0)
                    results['rmse_quantile_norm'].append(0.0)
                    results['rmse_mean_norm'].append(0.0)
                    results['gene'].append(gene)
                    processed_genes += 1
            
            # 显式释放内存
            gc.collect()
            #logger.info(f"基因块处理完成 - 已处理 {processed_genes}/{total_genes} 个基因 - {log_memory()}")
            
            # 每处理完一个块，保存一次临时结果
            temp_results = pd.DataFrame(results)
            temp_file = os.path.join(save_path, f"{prediction_type}_temp_results.csv")
            temp_results.to_csv(temp_file, index=False)
            
    except KeyboardInterrupt:
        logger.warning("检测到键盘中断，正在保存当前结果...")
        interrupted = True
    
    # 如果被中断，保存当前进度
    if interrupted:
        # 创建结果DataFrame
        gene_results = pd.DataFrame(results)
        gene_results.set_index('gene', inplace=True)
        gene_results.sort_values('pred_real_r', ascending=False, inplace=True)

        # 处理缺失值
        gene_results['pred_real_r'] = gene_results['pred_real_r'].fillna(0)
        gene_results['random_real_r'] = gene_results['random_real_r'].fillna(0)
        gene_results['pearson_p'] = gene_results['pearson_p'].fillna(1)
        gene_results['Steiger_p'] = gene_results['Steiger_p'].fillna(1)

        # 多重检验校正
        logger.warning("进行部分多重检验校正...")
        if not gene_results.empty:
            _, fdr_pearson = fdrcorrection(gene_results['pearson_p'])
            gene_results['fdr_pearson_p'] = fdr_pearson

            _, fdr_steiger = fdrcorrection(gene_results['Steiger_p'])
            gene_results['fdr_Steiger_p'] = fdr_steiger

        # 添加癌症类型信息
        gene_results['cancer'] = cancer_type

        # 保存结果
        all_genes_file = os.path.join(save_path, f"{prediction_type}_partial_results_{processed_genes}_genes.csv")
        gene_results.to_csv(all_genes_file)

        # 记录分析结果
        logger.warning(f"分析中断！已保存部分结果: {all_genes_file}")
        logger.warning(f"已处理基因数: {processed_genes}/{total_genes} ({processed_genes/total_genes*100:.1f}%)")
        
        # 关闭日志处理器
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
        
        return gene_results, None, None

    # 正常处理完成后继续
    # 创建结果DataFrame并排序
    gene_results = pd.DataFrame(results)
    gene_results.set_index('gene', inplace=True)
    gene_results.sort_values('pred_real_r', ascending=False, inplace=True)

    # 处理缺失值（当序列为常数时）
    gene_results['pred_real_r'] = gene_results['pred_real_r'].fillna(0)
    gene_results['random_real_r'] = gene_results['random_real_r'].fillna(0)
    gene_results['pearson_p'] = gene_results['pearson_p'].fillna(1)
    gene_results['Steiger_p'] = gene_results['Steiger_p'].fillna(1)

    # 多重检验校正
    logger.info("进行多重检验校正...")
    _, fdr_pearson = fdrcorrection(gene_results['pearson_p'])
    gene_results['fdr_pearson_p'] = fdr_pearson

    _, fdr_steiger = fdrcorrection(gene_results['Steiger_p'])
    gene_results['fdr_Steiger_p'] = fdr_steiger

    # 添加癌症类型信息
    gene_results['cancer'] = cancer_type

    # 筛选显著结果
    logger.info("筛选显著基因...")
    significance_filter = (
        (gene_results['pred_real_r'] > 0) &
        (gene_results['pearson_p'] < 0.05) &
        (gene_results['rmse_pred'] < gene_results['rmse_random']) &
        (gene_results['pred_real_r'] > gene_results['random_real_r']) &
        (gene_results['Steiger_p'] < 0.05) 
        & (gene_results['fdr_Steiger_p'] < 0.2)
    )
    significant_genes = gene_results[significance_filter]

    # 保存结果
    all_genes_file = os.path.join(save_path, "all_genes.csv")
    sig_genes_file = os.path.join(save_path, "sig_genes.csv")

    gene_results.to_csv(all_genes_file)
    significant_genes.to_csv(sig_genes_file)

    # 计算显著基因数量
    sig_gene_count = significant_genes['cancer'].value_counts().reset_index()
    sig_gene_count.columns = ['cancer', 'num_genes']
    
    # 记录分析结果
    logger.info(f"分析完成！癌症类型: {cancer_type}, 预测模型: {prediction_type}")
    logger.info(f"总基因数: {len(gene_results)}, 显著基因数: {sig_gene_count['num_genes'][0]}")
    logger.info(f"所有基因结果保存到: {all_genes_file}")
    logger.info(f"显著基因结果保存到: {sig_genes_file}")
    
    # 生成详细分析报告
    analysis_report = generate_analysis_report(gene_results, significant_genes)
    report_file = os.path.join(save_path, "analysis_report.txt")
    with open(report_file, 'w') as f:
        f.write(analysis_report)
    logger.info(f"分析报告保存到: {report_file}")
    
    # 关闭日志处理器
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    
    return gene_results, significant_genes, sig_gene_count

def generate_analysis_report(gene_results, significant_genes):
    """
    生成详细的分析报告
    """
    report = f"基因表达预测分析报告\n"
    report += "=" * 50 + "\n\n"
    
    # 基本信息
    report += f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"总基因数: {len(gene_results)}\n"
    
    if significant_genes is not None:
        report += f"显著基因数: {len(significant_genes)}\n"
    else:
        report += "显著基因数: 未完成计算\n"
    report += "\n"
    
    # 整体统计信息
    report += "整体统计信息:\n"
    report += "-" * 50 + "\n"
    report += f"平均预测相关性 (pred_real_r): {gene_results['pred_real_r'].mean():.4f}\n"
    report += f"平均随机相关性 (random_real_r): {gene_results['random_real_r'].mean():.4f}\n"
    report += f"平均预测RMSE: {gene_results['rmse_pred'].mean():.4f}\n"
    report += f"平均随机RMSE: {gene_results['rmse_random'].mean():.4f}\n"
    report += f"平均归一化RMSE (四分位距): {gene_results['rmse_quantile_norm'].mean():.4f}\n"
    report += f"平均归一化RMSE (均值): {gene_results['rmse_mean_norm'].mean():.4f}\n"
    report += "\n"
    
    # 最佳预测基因
    report += "最佳预测基因 (按相关性排序):\n"
    report += "-" * 50 + "\n"
    top_genes = gene_results.sort_values('pred_real_r', ascending=False).head(10)
    for i, (gene, row) in enumerate(top_genes.iterrows()):
        report += f"{i+1}. {gene}: "
        report += f"相关性={row['pred_real_r']:.4f}, "
        report += f"p值={row['pearson_p']:.4g}, "
        if 'fdr_pearson_p' in row:
            report += f"FDR={row['fdr_pearson_p']:.4g}, "
        report += f"RMSE={row['rmse_pred']:.4f}\n"
    report += "\n"
    
    # 最显著基因
    if significant_genes is not None and not significant_genes.empty:
        report += "最显著基因 (按Steiger p值排序):\n"
        report += "-" * 50 + "\n"
        sig_top = significant_genes.sort_values('Steiger_p').head(10)
        for i, (gene, row) in enumerate(sig_top.iterrows()):
            report += f"{i+1}. {gene}: "
            report += f"相关性={row['pred_real_r']:.4f} vs 随机={row['random_real_r']:.4f}, "
            report += f"Steiger_p={row['Steiger_p']:.4g}, "
            if 'fdr_Steiger_p' in row:
                report += f"FDR={row['fdr_Steiger_p']:.4g}\n"
        report += "\n"
    
    # 相关性分布
    report += "相关性分布:\n"
    report += "-" * 50 + "\n"
    report += f"相关性 > 0.5: {len(gene_results[gene_results['pred_real_r'] > 0.5])} 个基因\n"
    report += f"相关性 > 0.4: {len(gene_results[gene_results['pred_real_r'] > 0.4])} 个基因\n"
    report += f"相关性 > 0.3: {len(gene_results[gene_results['pred_real_r'] > 0.3])} 个基因\n"
    report += f"相关性 > 0.2: {len(gene_results[gene_results['pred_real_r'] > 0.2])} 个基因\n"
    report += f"相关性 > 0.1: {len(gene_results[gene_results['pred_real_r'] > 0.1])} 个基因\n"
    report += f"相关性 <= 0.1: {len(gene_results[gene_results['pred_real_r'] <= 0.1])} 个基因\n"
    report += "\n"
    
    # 模型性能评估
    report += "模型性能评估 (预测与随机比较):\n"
    report += "-" * 50 + "\n"
    better_corr = len(gene_results[gene_results['pred_real_r'] > gene_results['random_real_r']])
    better_rmse = len(gene_results[gene_results['rmse_pred'] < gene_results['rmse_random']])
    
    report += f"预测相关性优于随机的基因数: {better_corr} ({better_corr/len(gene_results)*100:.1f}%)\n"
    report += f"预测RMSE低于随机的基因数: {better_rmse} ({better_rmse/len(gene_results)*100:.1f}%)\n"
    
    if significant_genes is not None:
        sig_better = len(significant_genes)
        report += f"统计显著优于随机的基因数: {sig_better} ({sig_better/len(gene_results)*100:.1f}%)\n"
    else:
        report += "统计显著优于随机的基因数: 未完成计算\n"
    
    report += "\n" + "=" * 50 + "\n"
    report += "分析结束\n"
    
    return report

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='基因表达预测结果分析工具')
    
    # 添加命令行参数
    parser.add_argument('--cancer_type', type=str, default="BLCA", 
                        help='癌症类型 (默认: BLCA)')
    parser.add_argument("--prediction_type", type=str, default="tformer_lin", 
                        choices=["tformer", "tformer_lin", "Agent", "MLP"], help="预测模型类型")
    parser.add_argument("--extraction_model", type=str, default="vit", choices=["vit", "resnet"], help="特征提取模型")
    parser.add_argument("--cluster_type", type=str, default="KMeans", help="聚类类型")
    parser.add_argument('--folds', type=int, default=5, 
                        help='交叉验证折数 (默认: 5)')
    parser.add_argument('--module_path', type=str, default="/backup/lgx/path_omics/", 
                        help='自定义模块路径 (默认: /backup/lgx/path_omics/)')
    parser.add_argument('--base_dir', type=str, default="/backup/lgx/path_omics_t/", 
                        help='工作基础目录 (默认: /backup/lgx/path_omics_t/)')
    
    # 解析参数
    args = parser.parse_args()
    
    try:
        # 调用主函数
        analyze_gene_expression(
            cancer_type=args.cancer_type,
            prediction_type=args.prediction_type,
            extraction_model=args.extraction_model,
            cluster_type=args.cluster_type,
            num_folds=args.folds,
            module_path=args.module_path,
            base_dir=args.base_dir
        )
    except KeyboardInterrupt:
        print("\n检测到键盘中断，程序已终止")
    except Exception as e:
        print(f"发生未处理异常: {str(e)}")
