
import os
import argparse
import json
from collections import Counter
import heapq

import numpy as np
import torch

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen.utils.random import reseed_everything
from alphagen.utils.qlib_converter import AlphaSAGEToQlibConverter
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
from gan.utils.data import get_data_by_year
from datetime import datetime, timezone

QLIB_PATH = '~/.qlib/qlib_data/cn_data'

# ========== Global variables related to experiment result saving ==========
convergence_history = []  # Convergence data for each generation
pool_results_history = []  # Pool results for each generation
experiment_results_dir = ''  # Experiment results save directory
experiment_name = ''  # Experiment name (used to distinguish with/without LLM seeds)

def _metric(x, y, w):
    key = y[0]

    if key in cache:
        return cache[key]
    token_len = key.count('(') + key.count(')')
    if token_len > 20:
        return -1.

    expr = eval(key)
    try:
        factor = expr.evaluate(data)
        factor = normalize_by_day(factor)
        ic = batch_pearsonr(factor, target_factor)
        ic = torch.nan_to_num(ic).mean().item()
    except OutOfDataRangeError:
        ic = -1.
    except Exception:
        # Fallback: any expression structure/data exception should not crash training
        ic = -1.0
    if np.isnan(ic):
        ic = -1.
    cache[key] = ic
    return ic




def try_single():
    top_key = Counter(cache).most_common(1)[0][0]
    try:
        v_valid = eval(top_key).evaluate(data_valid)
        v_test = eval(top_key).evaluate(data_test)
        ic_test = batch_pearsonr(v_test, target_factor_test)
        ic_test = torch.nan_to_num(ic_test,nan=0,posinf=0,neginf=0).mean().item()
        ic_valid = batch_pearsonr(v_valid, target_factor_valid)
        ic_valid = torch.nan_to_num(ic_valid,nan=0,posinf=0,neginf=0).mean().item()
        ric_test = batch_spearmanr(v_test, target_factor_test)
        ric_test = torch.nan_to_num(ric_test,nan=0,posinf=0,neginf=0).mean().item()
        ric_valid = batch_spearmanr(v_valid, target_factor_valid)
        ric_valid = torch.nan_to_num(ric_valid,nan=0,posinf=0,neginf=0).mean().item()
        return {'ic_test': ic_test, 'ic_valid': ic_valid, 'ric_test': ric_test, 'ric_valid': ric_valid}
    except OutOfDataRangeError:
        print ('Out of data range')
        print(top_key)
        exit()
        return {'ic_test': -1., 'ic_valid': -1., 'ric_test': -1., 'ric_valid': -1.}


def try_pool(capacity):
    pool = AlphaPool(capacity=capacity,
                    stock_data=data,
                    target=target,
                    ic_lower_bound=None)

    exprs = []
    for key in dict(Counter(cache).most_common(capacity)):
        exprs.append(eval(key))
    pool.force_load_exprs(exprs)
    pool._optimize(alpha=5e-3, lr=5e-4, n_iter=2000)

    ic_test, ric_test = pool.test_ensemble(data_test, target)
    ic_valid, ric_valid = pool.test_ensemble(data_valid, target)
    return {'ic_test': ic_test, 'ic_valid': ic_valid, 'ric_test': ric_test, 'ric_valid': ric_valid}




def ev():
    global generation, top_factors_heap, max_generations
    global convergence_history, pool_results_history  # Experiment result records
    generation += 1
    
    # Update heap with capacity 200, save top 200 factors with highest IC values
    # Use min heap, heap top is the minimum IC value
    # Build expression to IC value mapping, keep highest IC value (handle same expression in different iterations)
    if generation <= max_generations - 1:
        expr_to_ic = {}
        for expr, ic in cache.items():
            if expr not in expr_to_ic or ic > expr_to_ic[expr]:
                expr_to_ic[expr] = ic
        
        # Get set of expressions already in heap
        existing_exprs = {expr for _, expr in top_factors_heap}
        
        # Update heap: handle new factors or factors with higher IC values
        for expr, ic in expr_to_ic.items():
            if expr in existing_exprs:
                # If expression already exists, check if IC value needs to be updated
                for i, (old_ic, old_expr) in enumerate(top_factors_heap):
                    if old_expr == expr and ic > old_ic:
                        # Update IC value at this position in heap
                        top_factors_heap[i] = (ic, expr)
                        heapq.heapify(top_factors_heap)  # Re-heapify
                        break
            else:
                # Expression doesn't exist, try to add to heap
                if len(top_factors_heap) < 200:
                    heapq.heappush(top_factors_heap, (ic, expr))
                elif ic > top_factors_heap[0][0]:  # If current IC is greater than minimum IC at heap top
                    heapq.heapreplace(top_factors_heap, (ic, expr))
    
    res = (
        [{'pool': 0, 'res': try_single()}] +
        [{'pool': cap, 'res': try_pool(cap)} for cap in (10, 20, 50, 100)]
    )
    print(res)
    
    # ========== Record convergence data for each generation (for ablation experiments) ==========
    # Get statistical metrics for current generation
    top_factors = Counter(cache).most_common(10)
    best_ic_train = top_factors[0][1] if top_factors else 0
    top10_avg_ic = np.mean([ic for _, ic in top_factors]) if top_factors else 0
    
    # Get validation set metrics for best factor
    best_expr = top_factors[0][0] if top_factors else None
    best_valid_metrics = calculate_factor_valid_metrics(best_expr) if best_expr else {}
    
    # Record convergence history
    convergence_record = {
        'generation': generation,
        'best_ic_train': best_ic_train,
        'top10_avg_ic_train': top10_avg_ic,
        'best_ic_valid': best_valid_metrics.get('ic_valid'),
        'best_rank_ic_valid': best_valid_metrics.get('rank_ic_valid'),
        'cache_size': len(cache),
        'positive_ic_count': sum(1 for _, ic in cache.items() if ic > 0),
        'single_best': res[0]['res'] if res else {}  # try_single result
    }
    convergence_history.append(convergence_record)
    
    # Record Pool results
    pool_record = {
        'generation': generation,
        'pools': {}
    }
    for r in res:
        if r['pool'] > 0:
            pool_record['pools'][f'pool_{r["pool"]}'] = r['res']
    pool_results_history.append(pool_record)
    
    global save_dir
    dir_ = save_dir
    #'/path/to/save/results'
    os.makedirs(dir_, exist_ok=True)
    # if generation % 2 == 0:
    #     with open(f'{dir_}/{generation}.json', 'w') as f:
    #         json.dump({'cache': cache, 'res': res}, f)
        
        # Also save qlib format factors
        #save_qlib_factors(dir_, generation, cache)
    
    # Only on the last iteration, merge and save heap and top 100 factors with highest IC values from last generation
    if generation == max_generations:
        try:
            save_qlib_factors(dir_, generation, cache)
            save_top_factors_heap(dir_, generation, cache)
            print(f'[train_GP] All factors saved to directory: {dir_}')
            
            # ========== Save experiment results (for ablation experiment comparison) ==========
            save_experiment_results()
            
        except Exception as e:
            print(f'[train_GP] Error saving factors: {e}')
            import traceback
            traceback.print_exc()


def save_qlib_factors(dir_: str, generation: int, cache: dict, top_n: int = 100):
    """
    Convert mined factors to qlib format and save, while calculating validation set rank_ic values
    
    Args:
        dir_: Save directory
        generation: Current iteration generation number
        cache: Factor cache {expression: IC value}
        top_n: Save top N factors with highest IC values
    """
    # Create converter
    converter = AlphaSAGEToQlibConverter()
    
    # Get top N factors
    top_factors = Counter(cache).most_common(top_n)
    
    # Filter factors with positive IC
    valid_top_factors = [(expr, ic) for expr, ic in top_factors if ic > 0]
    
    # Batch calculate rank_ic values
    expr_list = [expr for expr, ic in valid_top_factors]
    rank_ic_list = calculate_rank_ic_batch(expr_list)
    
    # Build factor list
    factors = []
    for (expr, ic), rank_ic in zip(valid_top_factors, rank_ic_list):
        factors.append({
            'expression': expr, 
            'ic': ic,
            'rank_ic_valid': rank_ic
        })
    
    print(f'[train_GP] Calculated rank_ic for {len(factors)} factors')
    
    # Convert to qlib format, and filter invalid expressions (e.g., expressions doing Rolling operations on constants)
    converted_factors = converter.convert_batch(factors, filter_invalid=True)
    
    # Build output data
    output_data = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'generation': generation,
        'factors_count': len(converted_factors),
        'source': 'AlphaSAGE_GP',
        'factors': converted_factors,
        # Also save qlib directly usable fields format (only contains valid expressions)
        'qlib_fields': [f['qlib_expression'] for f in converted_factors if f.get('qlib_expression') and f.get('is_valid', True)],
        'qlib_names': [f'ALPHA_{i+1:03d}' for i, f in enumerate(converted_factors) if f.get('qlib_expression') and f.get('is_valid', True)]
    }
    
    # Save qlib format factors
    qlib_output_file = f'{dir_}/{generation}_qlib_factors.json'
    with open(qlib_output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f'[train_GP] Qlib format factors saved: {qlib_output_file}')
    print(f'[train_GP] Successfully converted {len(output_data["qlib_fields"])} factors')


def calculate_rank_ic_for_factor(expr_str: str):
    """
    Calculate rank_ic value for a single factor on validation set
    
    Args:
        expr_str: Factor expression string
        
    Returns:
        float: rank_ic value on validation set, returns None if calculation fails
    """
    try:
        expr = eval(expr_str)
        v_valid = expr.evaluate(data_valid)
        v_valid = normalize_by_day(v_valid)
        ric_valid = batch_spearmanr(v_valid, target_factor_valid)
        ric_valid = torch.nan_to_num(ric_valid, nan=0, posinf=0, neginf=0).mean().item()
        return ric_valid
    except (OutOfDataRangeError, Exception) as e:
        print(f'[train_GP] Failed to calculate rank_ic: {expr_str}, error: {str(e)}')
        return None


def calculate_rank_ic_batch(expr_list: list):
    """
    Batch calculate rank_ic values for multiple factors on validation set
    
    Args:
        expr_list: Factor expression string list
        
    Returns:
        list: rank_ic value for each factor, returns None if calculation fails
    """
    print(f'[train_GP] Starting batch calculation of rank_ic for {len(expr_list)} factors...')
    
    # Batch evaluate all factors
    valid_factors = []
    valid_indices = []
    
    for i, expr_str in enumerate(expr_list):
        try:
            expr = eval(expr_str)
            v_valid = expr.evaluate(data_valid)
            v_valid = normalize_by_day(v_valid)
            valid_factors.append(v_valid)
            valid_indices.append(i)
        except (OutOfDataRangeError, Exception) as e:
            print(f'[train_GP] Factor {i} evaluation failed: {expr_str[:50]}..., error: {str(e)}')
    
    # Initialize result list
    rank_ic_list = [None] * len(expr_list)
    
    if len(valid_factors) == 0:
        print(f'[train_GP] Warning: No valid factors to calculate rank_ic')
        return rank_ic_list
    
    print(f'[train_GP] Successfully evaluated {len(valid_factors)}/{len(expr_list)} factors, starting batch rank_ic calculation...')
    
    # Batch calculate rank_ic for all valid factors
    try:
        # Calculate rank_ic (Spearman correlation coefficient) with target for each factor
        rank_ics = []
        for i, factor_tensor in enumerate(valid_factors):
            ric = batch_spearmanr(factor_tensor, target_factor_valid)
            ric = torch.nan_to_num(ric, nan=0, posinf=0, neginf=0).mean().item()
            rank_ics.append(ric)
            
            # Print progress every 100 factors
            if (i + 1) % 100 == 0:
                print(f'[train_GP] Calculated rank_ic for {i+1}/{len(valid_factors)} factors')
        
        # Fill results to corresponding positions
        for idx, ric_val in zip(valid_indices, rank_ics):
            rank_ic_list[idx] = ric_val
        
        print(f'[train_GP] Batch calculation completed, successfully calculated {len(rank_ics)} rank_ic values')
        
    except Exception as e:
        print(f'[train_GP] Error in batch rank_ic calculation: {str(e)}, falling back to individual calculation')
        # If batch calculation fails, fall back to individual calculation
        for idx in valid_indices:
            try:
                expr = eval(expr_list[idx])
                v_valid = expr.evaluate(data_valid)
                v_valid = normalize_by_day(v_valid)
                ric = batch_spearmanr(v_valid, target_factor_valid)
                ric = torch.nan_to_num(ric, nan=0, posinf=0, neginf=0).mean().item()
                rank_ic_list[idx] = ric
            except Exception as e2:
                print(f'[train_GP] Factor {idx} calculation failed: {str(e2)}')
    
    return rank_ic_list


def calculate_factor_test_metrics(expr_str: str):
    """
    Calculate IC and Rank IC for a single factor on test set
    
    Args:
        expr_str: Factor expression string
        
    Returns:
        dict: {'ic_test': float, 'rank_ic_test': float}, returns None values if calculation fails
    """
    try:
        expr = eval(expr_str)
        v_test = expr.evaluate(data_test)
        v_test = normalize_by_day(v_test)
        
        ic_test = batch_pearsonr(v_test, target_factor_test)
        ic_test = torch.nan_to_num(ic_test, nan=0, posinf=0, neginf=0).mean().item()
        
        ric_test = batch_spearmanr(v_test, target_factor_test)
        ric_test = torch.nan_to_num(ric_test, nan=0, posinf=0, neginf=0).mean().item()
        
        return {'ic_test': ic_test, 'rank_ic_test': ric_test}
    except Exception as e:
        return {'ic_test': None, 'rank_ic_test': None}


def calculate_factor_valid_metrics(expr_str: str):
    """
    Calculate IC and Rank IC for a single factor on validation set
    
    Args:
        expr_str: Factor expression string
        
    Returns:
        dict: {'ic_valid': float, 'rank_ic_valid': float}, returns None values if calculation fails
    """
    try:
        expr = eval(expr_str)
        v_valid = expr.evaluate(data_valid)
        v_valid = normalize_by_day(v_valid)
        
        ic_valid = batch_pearsonr(v_valid, target_factor_valid)
        ic_valid = torch.nan_to_num(ic_valid, nan=0, posinf=0, neginf=0).mean().item()
        
        ric_valid = batch_spearmanr(v_valid, target_factor_valid)
        ric_valid = torch.nan_to_num(ric_valid, nan=0, posinf=0, neginf=0).mean().item()
        
        return {'ic_valid': ic_valid, 'rank_ic_valid': ric_valid}
    except Exception as e:
        return {'ic_valid': None, 'rank_ic_valid': None}


def calculate_correlation_matrix(expr_list: list, top_n: int = 50):
    """
    Calculate correlation matrix for top N factors
    
    Args:
        expr_list: List of factor expression strings
        top_n: Number of top factors to calculate correlation for
        
    Returns:
        dict: Correlation matrix data
    """
    global data_valid
    
    # Limit to top N factors
    expr_list = expr_list[:top_n]
    
    if len(expr_list) == 0:
        return {
            'expressions': [],
            'correlation_matrix': [],
            'top_n': top_n
        }
    
    # Evaluate all factors on validation set
    factors_tensors = []
    valid_exprs = []
    
    for expr_str in expr_list:
        try:
            expr = eval(expr_str)
            factor = expr.evaluate(data_valid)
            factor = normalize_by_day(factor)
            factors_tensors.append(factor)
            valid_exprs.append(expr_str)
        except Exception as e:
            print(f'[train_GP] Failed to evaluate factor for correlation: {expr_str[:50]}..., error: {e}')
            continue
    
    if len(factors_tensors) == 0:
        return {
            'expressions': [],
            'correlation_matrix': [],
            'top_n': top_n
        }
    
    # Calculate pairwise correlation matrix
    n = len(factors_tensors)
    correlation_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                try:
                    # Calculate Pearson correlation
                    corr = batch_pearsonr(factors_tensors[i], factors_tensors[j])
                    corr = torch.nan_to_num(corr, nan=0, posinf=0, neginf=0).mean().item()
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
                except Exception as e:
                    print(f'[train_GP] Failed to calculate correlation between factor {i} and {j}: {e}')
                    correlation_matrix[i, j] = 0.0
                    correlation_matrix[j, i] = 0.0
    
    return {
        'expressions': valid_exprs,
        'correlation_matrix': correlation_matrix.tolist(),
        'top_n': len(valid_exprs)
    }


def save_experiment_results():
    """
    Save experiment results for ablation experiment comparison
    """
    global cache, generation, max_generations, experiment_results_dir, experiment_name
    global convergence_history, pool_results_history
    
    os.makedirs(experiment_results_dir, exist_ok=True)
    print(f'[train_GP] Starting to save experiment results to: {experiment_results_dir}')
    
    # ========== 1. Save convergence history (for line chart) ==========
    convergence_file = os.path.join(experiment_results_dir, 'convergence_history.json')
    with open(convergence_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment_name': experiment_name,
            'total_generations': max_generations,
            'history': convergence_history
        }, f, indent=2, ensure_ascii=False)
    print(f'[train_GP] Convergence history saved: {convergence_file}')
    
    # ========== 2. Save final factor details (for box plot) ==========
    # Get Top-200 factors
    top_factors = Counter(cache).most_common(200)
    final_factors = []
    
    print(f'[train_GP] Calculating complete metrics for Top-200 factors...')
    for i, (expr, ic_train) in enumerate(top_factors):
        if ic_train <= 0:
            continue
        
        factor_info = {
            'rank': i + 1,
            'expression': expr,
            'ic_train': ic_train
        }
        
        # Calculate validation set metrics
        valid_metrics = calculate_factor_valid_metrics(expr)
        factor_info.update(valid_metrics)
        
        # Calculate test set metrics
        test_metrics = calculate_factor_test_metrics(expr)
        factor_info.update(test_metrics)
        
        final_factors.append(factor_info)
        
        if (i + 1) % 50 == 0:
            print(f'[train_GP] Calculated metrics for {i+1}/{len(top_factors)} factors')
    
    final_factors_file = os.path.join(experiment_results_dir, 'final_factors.json')
    with open(final_factors_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment_name': experiment_name,
            'total_factors': len(final_factors),
            'factors': final_factors
        }, f, indent=2, ensure_ascii=False)
    print(f'[train_GP] Final factor details saved: {final_factors_file}')
    
    # ========== 3. Save Pool results (for bar chart) ==========
    # Get Pool results from last generation
    if pool_results_history:
        final_pool_results = pool_results_history[-1] if pool_results_history else {}
    else:
        final_pool_results = {}
    
    pool_results_file = os.path.join(experiment_results_dir, 'pool_results.json')
    with open(pool_results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment_name': experiment_name,
            'final_generation': generation,
            'pool_results': final_pool_results,
            'pool_history': pool_results_history
        }, f, indent=2, ensure_ascii=False)
    print(f'[train_GP] Pool results saved: {pool_results_file}')
    
    # ========== 4. Save factor correlation matrix (for heatmap) ==========
    expr_list = [expr for expr, ic in top_factors if ic > 0]
    corr_data = calculate_correlation_matrix(expr_list, top_n=50)
    
    correlation_file = os.path.join(experiment_results_dir, 'correlation_matrix.json')
    with open(correlation_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment_name': experiment_name,
            **corr_data
        }, f, indent=2, ensure_ascii=False)
    print(f'[train_GP] Correlation matrix saved: {correlation_file}')
    
    # ========== 5. Save experiment summary (for main results table) ==========
    # Calculate summary metrics
    valid_factors = [f for f in final_factors if f.get('ic_train') and f['ic_train'] > 0]
    
    summary = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'total_generations': max_generations,
        'final_generation': generation,
        
        # Factor count statistics
        'total_cache_size': len(cache),
        'valid_factors_count': len(valid_factors),
        'positive_ic_count': sum(1 for _, ic in cache.items() if ic > 0),
        
        # Best factor metrics
        'best_ic_train': max([f['ic_train'] for f in valid_factors]) if valid_factors else None,
        'best_ic_valid': max([f['ic_valid'] for f in valid_factors if f.get('ic_valid')]) if valid_factors else None,
        'best_ic_test': max([f['ic_test'] for f in valid_factors if f.get('ic_test')]) if valid_factors else None,
        'best_rank_ic_valid': max([f['rank_ic_valid'] for f in valid_factors if f.get('rank_ic_valid')]) if valid_factors else None,
        'best_rank_ic_test': max([f['rank_ic_test'] for f in valid_factors if f.get('rank_ic_test')]) if valid_factors else None,
        
        # Top-10 factor average metrics
        'top10_avg_ic_train': np.mean([f['ic_train'] for f in valid_factors[:10]]) if len(valid_factors) >= 10 else None,
        'top10_avg_ic_valid': np.mean([f['ic_valid'] for f in valid_factors[:10] if f.get('ic_valid')]) if len(valid_factors) >= 10 else None,
        'top10_avg_ic_test': np.mean([f['ic_test'] for f in valid_factors[:10] if f.get('ic_test')]) if len(valid_factors) >= 10 else None,
        
        # Pool combination effect
        'pool_results': final_pool_results,
        
        # Convergence information
        'convergence_generation': None,  # Generation when threshold first reached
    }
    
    # Calculate convergence generation (first generation where best_ic_train > 0.03)
    ic_threshold = 0.03
    for record in convergence_history:
        if record.get('best_ic_train', 0) > ic_threshold:
            summary['convergence_generation'] = record['generation']
            break
    
    summary_file = os.path.join(experiment_results_dir, 'summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[train_GP] Experiment summary saved: {summary_file}')
    
    print(f'[train_GP] ========== Experiment results saving completed ==========')


def save_top_factors_heap(dir_: str, generation: int, cache: dict):
    """
    On the last iteration, merge and save factors from heap and top 100 factors by IC value from current cache, and convert to qlib format
    Also calculate rank_ic value for each factor on validation set
    
    Args:
        dir_: Save directory
        generation: Current iteration generation number
        cache: Factor cache {expression: IC value}
    """
    global top_factors_heap
    
    # 创建转换器
    converter = AlphaSAGEToQlibConverter()
    
    # 1. 从堆中提取所有因子，按IC值从高到低排序
    heap_factors = list(top_factors_heap)
    heap_factors.sort(reverse=True)  # 按IC值降序排列
    
    # 2. 获取当前cache中IC值前100的因子
    top_cache_factors = Counter(cache).most_common(100)
    
    # 3. 合并堆中的因子和cache中前100的因子，去重并保留最高IC值
    merged_factors_dict = {}
    
    # 添加堆中的因子
    for ic, expr in heap_factors:
        if ic > 0:  # 只保留正 IC 的因子
            if expr not in merged_factors_dict or ic > merged_factors_dict[expr]:
                merged_factors_dict[expr] = ic
    
    # 添加cache中前100的因子
    for expr, ic in top_cache_factors:
        if ic > 0:  # 只保留正 IC 的因子
            if expr not in merged_factors_dict or ic > merged_factors_dict[expr]:
                merged_factors_dict[expr] = ic
    
    # 转换为因子列表格式，按IC值降序排序，并批量计算每个因子的rank_ic
    sorted_factors = sorted(merged_factors_dict.items(), key=lambda x: x[1], reverse=True)
    expr_list = [expr for expr, ic in sorted_factors]
    
    # 批量计算所有因子的rank_ic
    rank_ic_list = calculate_rank_ic_batch(expr_list)
    
    # 构建因子列表
    factors = []
    for (expr, ic), rank_ic in zip(sorted_factors, rank_ic_list):
        factors.append({
            'expression': expr, 
            'ic': ic,
            'rank_ic_valid': rank_ic
        })
    
    # 转换为 qlib 格式，并过滤无效表达式（如对常量做 Rolling 操作的表达式）
    converted_factors = converter.convert_batch(factors, filter_invalid=True)
    
    # 构建输出数据
    output_data = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'generation': generation,
        'factors_count': len(converted_factors),
        'source': 'AlphaSAGE_GP_Heap_Merged',
        'heap_factors_count': len(heap_factors),
        'cache_top100_count': len(top_cache_factors),
        'merged_unique_count': len(merged_factors_dict),
        'factors': converted_factors,
        # 同时保存 qlib 可直接使用的 fields 格式（只包含有效表达式）
        'qlib_fields': [f['qlib_expression'] for f in converted_factors if f.get('qlib_expression') and f.get('is_valid', True)],
        'qlib_names': [f'ALPHA_{i+1:03d}' for i, f in enumerate(converted_factors) if f.get('qlib_expression') and f.get('is_valid', True)]
    }
    
    # 保存合并后的因子
    heap_output_file = f'{dir_}/{generation}_top200_factors_heap_merged.json'
    with open(heap_output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f'[train_GP] 合并因子已保存: {heap_output_file}')
    print(f'[train_GP] 堆中因子数: {len(heap_factors)}, Cache前100因子数: {len(top_cache_factors)}, 合并去重后: {len(merged_factors_dict)}, 成功转换: {len(output_data["qlib_fields"])} 个因子')





def run(args):
    if args.instruments == 'sp500' or args.instruments == 'nasdaq100':
        QLIB_PATH = '~/.qlib/qlib_data/us_data'
    else:
        QLIB_PATH = '~/.qlib/qlib_data/cn_data'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    reseed_everything(args.seed)

    global data, data_valid, data_test, target, target_factor, target_factor_valid, target_factor_test, cache, generation, save_dir, top_factors_heap, max_generations
    # 实验结果相关的全局变量
    global convergence_history, pool_results_history, experiment_results_dir, experiment_name

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -2) / close - 1

    train_start_time = '2010-01-01'
    train_end_time = f'{args.train_end_year}-12-31'
    valid_start_time = f'{args.train_end_year + 1}-01-01'
    valid_end_time = f'{args.train_end_year + 1}-12-31'
    test_start_time = f'{args.train_end_year + 2}-01-01'
    test_end_time = f'{args.train_end_year + 4}-12-31'

    data = StockData(instrument=args.instruments,
                           start_time=train_start_time,
                           end_time=train_end_time,
                           qlib_path=QLIB_PATH)
    data_valid = StockData(instrument=args.instruments,
                           start_time=valid_start_time,
                           end_time=valid_end_time,
                           qlib_path=QLIB_PATH)
    data_test = StockData(instrument=args.instruments,
                          start_time=test_start_time,
                          end_time=test_end_time,
                          qlib_path=QLIB_PATH)
                          
    save_dir = f'data/gp_{args.instruments}_{args.train_end_year}_{args.freq}_{args.seed}_{datetime.now(timezone.utc).strftime("%m%d_%H%M")}'                      
    #save_dir = f'data/gp_{args.instruments}_{args.train_end_year}_{args.freq}_{args.seed}_{datetime.now().strftime("%m%d%H%M")}'
    #save_dir = f'data/{args.instruments}_{args.train_end_year}_{args.freq}_{args.seed}'

    Metric = make_fitness(function=_metric, greater_is_better=True)
    funcs = [make_function(**func._asdict()) for func in generic_funcs]

    generation = 0
    cache = {}
    top_factors_heap = []  # 容量为200的堆，保存IC值最高的200个因子
    
    # ========== 初始化实验结果记录变量 ==========
    convergence_history = []
    pool_results_history = []

    target_factor = target.evaluate(data)
    target_factor_valid = target.evaluate(data_valid)
    target_factor_test = target.evaluate(data_test)

    # ========== 加载种子因子（来自 mining_feedback["suggested_seeds"]）==========
    seed_factors = []
    seed_factors_original = []  # 保存原始种子因子
    seed_factors_file = getattr(args, 'seed_factors_file', '')
    
    # 设置实验结果保存目录和实验名称

    #experiment_results_dir = f'data/experiment_results_{args.seed}'
    experiment_results_dir = f'AlphaSAGE/data/experiment_results_1_No_LLM_seed'
    # 根据是否使用LLM种子因子设置实验名称
    if seed_factors_file:
        experiment_name = f'with_LLM_seed_{args.instruments}_{args.train_end_year}_seed{args.seed}'
    else:
        experiment_name = f'without_LLM_seed_{args.instruments}_{args.train_end_year}_seed{args.seed}'
    
    print(f'[train_GP] 实验名称: {experiment_name}')
    print(f'[train_GP] 实验结果将保存到: {experiment_results_dir}')
    
    if seed_factors_file:
        seed_factors_original = load_seed_factors(seed_factors_file)
        if seed_factors_original:
            # 扩增种子因子：目标扩增到 300 个，用于更有效地引导 GP
            # 扩增比例约为 50% 的初始种群（1000 * 0.5 = 500，但考虑到验证失败的情况，目标设为 300）
            amplify_target = min(400, int(1000 * 0.5))  # 目标扩增数量
            seed_factors = amplify_seed_factors(
                seed_factors_original, 
                target_count=amplify_target,
                random_state=np.random.RandomState(args.seed)
            )
            
            # 将所有种子因子（原始 + 扩增）预先注入到 cache 中
            inject_seed_factors_to_cache(seed_factors, data, target_factor)
            print(f'[train_GP] 已加载种子因子: 原始={len(seed_factors_original)}, 扩增后={len(seed_factors)}, 将加入初始种群')
    
    # ========== 构建 terminals（基础特征 + 常量，不再包含种子因子）==========
    features = ['open_', 'close', 'high', 'low', 'volume']
    constants = [f'Constant({v})' for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]]
    
    terminals = features + constants
    print(f'[train_GP] Terminals 总数: {len(terminals)} (基础特征: {len(features)}, 常量: {len(constants)})')

    X_train = np.array([terminals])
    y_train = np.array([[1]])

    max_generations = 30  # 总迭代次数
    est_gp = SymbolicRegressor(population_size=1000,
                            generations=max_generations,
                            init_depth=(2, 6),
                            tournament_size=600,
                            stopping_criteria=1.,
                            p_crossover=0.3,
                            p_subtree_mutation=0.1,
                            p_hoist_mutation=0.01,
                            p_point_mutation=0.1,
                            p_point_replace=0.6,
                            max_samples=0.9,
                            verbose=1,
                            parsimony_coefficient=0.,
                            random_state=args.seed,
                            function_set=funcs,
                            metric=Metric,
                            const_range=None,
                            n_jobs=1)
    
    # 如果存在种子因子，使用warm_start方式：先fit一次生成初始种群，然后注入种子因子，再继续fit
    if seed_factors:
            # 第一步：只生成初始种群（generations=1）
            est_gp.set_params(generations=1, warm_start=False)
            est_gp.fit(X_train, y_train, callback=None)  # 第一次fit不使用callback，避免重复计算
            
            # 将种子因子加入初始种群（替换部分fitness较低的个体）
            actually_injected = inject_seed_factors_to_initial_population(
                est_gp, seed_factors, X_train, y_train, terminals, len(features),
                replace_ratio=0.5  # 替换50%的初始种群，使种子因子更有效地引导演化
            )
            print(f'[train_GP] 最终成功注入 {actually_injected}/{len(seed_factors)} 个种子因子到初始种群')
            
            # 第二步：使用warm_start继续演化剩余的generations
            # 重要：将 generation 设置为 1，因为第一次 fit 已经完成了 1 代
            generation = 1
            est_gp.set_params(generations=max_generations, warm_start=True)
            est_gp.fit(X_train, y_train, callback=ev)
    else:
        est_gp.fit(X_train, y_train, callback=ev)
    
    print(est_gp._program.execute(X_train))


def has_future_leakage(expr_str: str) -> bool:
    """
    检查表达式是否存在未来信息泄露
    
    未来信息泄露的情况：
    - Ref(x, -n) 其中 n > 0，表示未来 n 天的数据
    
    Args:
        expr_str: 表达式字符串
        
    Returns:
        bool: True 表示存在未来信息泄露
    """
    import re
    # 匹配 Ref(xxx, -数字) 模式，负数表示未来数据
    ref_pattern = r'Ref\([^,]+,\s*-(\d+)\)'
    matches = re.findall(ref_pattern, expr_str)
    if matches:
        for val in matches:
            if int(val) > 0:  # Ref(x, -5) 中的 5 > 0
                return True
    return False


def load_seed_factors(seed_factors_file: str) -> list:
    """
    从文件加载种子因子，并验证其有效性
    同时检查并过滤掉存在未来信息泄露的因子
    
    Args:
        seed_factors_file: 种子因子 JSON 文件路径
        
    Returns:
        list: 有效的种子因子表达式列表（不含未来信息泄露）
    """
    if not seed_factors_file or not os.path.exists(seed_factors_file):
        print(f'[train_GP] 种子因子文件不存在: {seed_factors_file}')
        return []
    
    try:
        with open(seed_factors_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        seed_factors_raw = data.get('seed_factors', [])
        print(f'[train_GP] 从文件加载 {len(seed_factors_raw)} 个种子因子')
        
        # 验证并过滤有效的种子因子
        valid_seeds = []
        leakage_count = 0
        for sf in seed_factors_raw:
            expr_str = sf.get('expression', '') if isinstance(sf, dict) else str(sf)
            if not expr_str:
                continue
            
            # 检查未来信息泄露
            if has_future_leakage(expr_str):
                print(f'[train_GP] 种子因子存在未来信息泄露，已过滤: {expr_str[:60]}...')
                leakage_count += 1
                continue
                
            # 尝试解析表达式验证有效性
            try:
                _ = eval(expr_str)
                valid_seeds.append(expr_str)
                print(f'[train_GP] 种子因子有效: {expr_str[:60]}...')
            except Exception as e:
                print(f'[train_GP] 种子因子无效: {expr_str[:60]}..., 错误: {e}')
        
        print(f'[train_GP] 验证后有效种子因子数: {len(valid_seeds)}/{len(seed_factors_raw)} (过滤未来信息泄露: {leakage_count})')
        return valid_seeds
        
    except Exception as e:
        print(f'[train_GP] 加载种子因子文件失败: {e}')
        return []


def amplify_seed_factors(seed_factors: list, target_count: int = 200, random_state=None) -> list:
    """
    对种子因子进行扩增，通过变异生成更多变体
    
    扩增策略：
    1. 参数变异：改变时间窗口参数 (10, 20, 30, 40, 50)
    2. 操作符替换：用相似操作符替换 (TsMean <-> TsEMA, Add <-> Sub 等)
    3. 特征替换：替换基础特征 (close <-> vwap, high <-> low 等)
    4. 子表达式交叉：将不同种子因子的子表达式组合
    
    Args:
        seed_factors: 原始种子因子表达式列表
        target_count: 目标扩增数量
        random_state: 随机数生成器
        
    Returns:
        list: 扩增后的种子因子列表（包含原始因子）
    """
    import re
    import random
    
    if random_state is None:
        random_state = random.Random(42)
    elif isinstance(random_state, np.random.RandomState):
        # 如果是 numpy 的 RandomState，转换为 Python 的 random
        random_state = random.Random(random_state.randint(0, 2**31))
    
    if not seed_factors:
        return []
    
    # 时间窗口参数集合
    time_windows = [5, 10, 15, 20, 30, 40, 50, 60]
    
    # 可替换的操作符映射（相似功能）
    operator_replacements = {
        'TsMean': ['TsEMA', 'TsWMA', 'TsMed'],
        'TsEMA': ['TsMean', 'TsWMA'],
        'TsWMA': ['TsMean', 'TsEMA'],
        'TsMed': ['TsMean'],
        'TsStd': ['TsMad', 'TsVar'],
        'TsMad': ['TsStd'],
        'TsVar': ['TsStd'],
        'TsMax': ['TsMin', 'TsArgMax'],
        'TsMin': ['TsMax', 'TsArgMin'],
        'TsRank': ['TsArgMax', 'TsArgMin'],
        'Add': ['Sub'],
        'Sub': ['Add'],
        'Mul': ['Div'],
        'Div': ['Mul'],
        'Greater': ['Less'],
        'Less': ['Greater'],
    }
    
    # 可替换的特征映射
    feature_replacements = {
        'close': ['vwap', 'open_', 'high', 'low'],
        'vwap': ['close', 'open_'],
        'open_': ['close', 'vwap'],
        'high': ['low', 'close'],
        'low': ['high', 'close'],
        'volume': ['close', 'vwap'],
    }
    
    # 用于存储所有变体（包括原始因子）
    all_variants = set(seed_factors)
    
    def mutate_time_window(expr_str: str) -> list:
        """变异时间窗口参数
        
        注意：为避免未来信息泄露，所有时间窗口参数都使用正数
        - Ts 系列操作符：正数表示回看窗口长度
        - Ref 操作符：正数表示往过去偏移（如 Ref(close, 5) = 5天前的close）
                      负数表示往未来偏移（信息泄露！应避免）
        """
        variants = []
        # 匹配形如 TsXxx(..., 数字) 或 Ref(..., 数字) 的模式
        pattern = r'(Ts\w+|Ref)\(([^,]+),\s*(-?\d+)\)'
        matches = list(re.finditer(pattern, expr_str))
        
        if not matches:
            return variants
        
        for match in matches:
            op, args, window = match.groups()
            window_int = int(window)
            
            # 跳过 Ref 的负数参数（这可能导致未来信息泄露）
            if op == 'Ref' and window_int < 0:
                # 将负数 Ref 转换为正数（修复原始表达式的潜在问题）
                # Ref(x, -5) 改为 Ref(x, 5)，避免未来信息
                continue
            
            # 尝试不同的时间窗口（都使用正数，避免未来信息泄露）
            for new_window in time_windows:
                if new_window != abs(window_int):
                    # 所有操作符都使用正数时间窗口
                    new_expr = expr_str[:match.start()] + f'{op}({args}, {new_window})' + expr_str[match.end():]
                    variants.append(new_expr)
        
        return variants
    
    def mutate_operator(expr_str: str) -> list:
        """替换操作符"""
        variants = []
        for op, replacements in operator_replacements.items():
            if op in expr_str:
                for new_op in replacements:
                    new_expr = expr_str.replace(op, new_op, 1)  # 只替换第一个
                    if new_expr != expr_str:
                        variants.append(new_expr)
        return variants
    
    def mutate_feature(expr_str: str) -> list:
        """替换基础特征"""
        variants = []
        for feat, replacements in feature_replacements.items():
            # 匹配独立的特征名（不是操作符的一部分）
            pattern = rf'\b{feat}\b'
            if re.search(pattern, expr_str):
                for new_feat in replacements:
                    new_expr = re.sub(pattern, new_feat, expr_str, count=1)
                    if new_expr != expr_str:
                        variants.append(new_expr)
        return variants
    
    def is_valid_expr(expr_str: str) -> bool:
        """验证表达式是否有效，并检查是否存在未来信息泄露"""
        try:
            # 1. 检查未来信息泄露：Ref 的负数参数
            # Ref(x, -n) 表示未来 n 天的数据，这是信息泄露
            ref_pattern = r'Ref\([^,]+,\s*(-\d+)\)'
            ref_matches = re.findall(ref_pattern, expr_str)
            for ref_val in ref_matches:
                if int(ref_val) < 0:
                    return False  # 存在未来信息泄露，拒绝该表达式
            
            # 2. 验证表达式语法是否正确
            expr = eval(expr_str)
            return expr is not None
        except Exception:
            return False
    
    print(f'[train_GP] 开始扩增种子因子：原始数量={len(seed_factors)}, 目标数量={target_count}')
    
    # 迭代扩增直到达到目标数量
    iteration = 0
    max_iterations = 20  # 防止无限循环
    
    while len(all_variants) < target_count and iteration < max_iterations:
        iteration += 1
        current_factors = list(all_variants)
        new_variants = []
        
        for expr_str in current_factors:
            if len(all_variants) + len(new_variants) >= target_count:
                break
            
            # 1. 时间窗口变异
            window_variants = mutate_time_window(expr_str)
            for v in window_variants:
                if v not in all_variants and is_valid_expr(v):
                    new_variants.append(v)
                    if len(all_variants) + len(new_variants) >= target_count:
                        break
            
            # 2. 操作符变异
            if len(all_variants) + len(new_variants) < target_count:
                op_variants = mutate_operator(expr_str)
                for v in op_variants:
                    if v not in all_variants and v not in new_variants and is_valid_expr(v):
                        new_variants.append(v)
                        if len(all_variants) + len(new_variants) >= target_count:
                            break
            
            # 3. 特征变异
            if len(all_variants) + len(new_variants) < target_count:
                feat_variants = mutate_feature(expr_str)
                for v in feat_variants:
                    if v not in all_variants and v not in new_variants and is_valid_expr(v):
                        new_variants.append(v)
                        if len(all_variants) + len(new_variants) >= target_count:
                            break
        
        # 添加新变体
        all_variants.update(new_variants)
        
        if not new_variants:
            # 如果没有生成新变体，尝试组合变异
            print(f'[train_GP] 迭代 {iteration}: 单一变异无法产生更多变体，尝试组合变异')
            
            # 对已有变体再次进行变异
            for expr_str in random_state.sample(list(all_variants), min(50, len(all_variants))):
                if len(all_variants) >= target_count:
                    break
                
                # 组合变异：先改窗口，再改操作符
                window_variants = mutate_time_window(expr_str)
                for wv in window_variants[:3]:  # 限制数量
                    op_variants = mutate_operator(wv)
                    for ov in op_variants[:2]:
                        if ov not in all_variants and is_valid_expr(ov):
                            all_variants.add(ov)
                            if len(all_variants) >= target_count:
                                break
        
        print(f'[train_GP] 迭代 {iteration}: 当前变体数量={len(all_variants)}')
    
    # 转换为列表并返回
    result = list(all_variants)
    
    # 确保原始种子因子在列表前面
    for sf in reversed(seed_factors):
        if sf in result:
            result.remove(sf)
            result.insert(0, sf)
    
    print(f'[train_GP] 种子因子扩增完成：原始={len(seed_factors)}, 扩增后={len(result)}')
    
    return result[:target_count]  # 确保不超过目标数量


def inject_seed_factors_to_cache(seed_factors: list, data_obj, target_factor_tensor):
    """
    将种子因子预先计算 IC 并注入到 cache 中，
    这样 GP 在变异/交叉时可以利用这些高质量因子作为构建块
    
    Args:
        seed_factors: 种子因子表达式列表
        data_obj: 训练数据对象
        target_factor_tensor: 目标因子张量
    """
    global cache
    
    injected_count = 0
    for expr_str in seed_factors:
        try:
            expr = eval(expr_str)
            factor = expr.evaluate(data_obj)
            factor = normalize_by_day(factor)
            ic = batch_pearsonr(factor, target_factor_tensor)
            ic = torch.nan_to_num(ic).mean().item()
            
            if not np.isnan(ic) and ic > -1:
                cache[expr_str] = ic
                injected_count += 1
                print(f'[train_GP] 种子因子注入 cache: IC={ic:.4f}, {expr_str[:50]}...')
        except Exception as e:
            print(f'[train_GP] 种子因子计算IC失败: {expr_str[:50]}..., 错误: {e}')
    
    print(f'[train_GP] 成功注入 {injected_count}/{len(seed_factors)} 个种子因子到 cache')


def convert_alphagen_to_gplearn_program(expr_str, function_set, terminals, n_features):
    """
    将 AlphaSAGE 表达式字符串转换为 gplearn 的 program 列表
    
    Args:
        expr_str: AlphaSAGE 表达式字符串，如 "Add(Ref(close, -2), close)"
        function_set: gplearn 的函数集合列表
        terminals: terminals 列表（特征名和常量字符串）
        n_features: 特征数量（不包括常量）
    
    Returns:
        program: gplearn 格式的 program 列表（前序遍历的扁平树），节点只应包含：
          - gplearn 的 _Function（来自 function_set）
          - int（特征索引，范围 [0, n_features)）
          - float（常量）
        注意：alphagen_generic 里 rolling/rolling_binary 操作符的窗口已经“固化”在函数名里
        （如 TsMean10 / TsCorr30），其 arity 分别是 1/2，因此 **program 列表里不应再附加 delta_time**。
    """
    # 解析表达式字符串为 AlphaSAGE 表达式对象
    expr = eval(expr_str)
    
    # 构建操作符名称到 gplearn 函数对象的映射
    func_name_map = {}
    for func in function_set:
        func_name_map[func.name] = func
    
    # 构建特征名称到 terminal 索引的映射
    feature_to_idx = {}
    for i, term in enumerate(terminals):
        if not term.startswith('Constant'):
            # 特征：映射到索引
            feature_to_idx[term] = i
            # 处理 open_ 的特殊情况
            if term == 'open_':
                feature_to_idx['open'] = i
    
    # 构建常量值到 terminal 索引的映射（常量在 terminals 中的位置）
    const_to_idx = {}
    const_values = []
    for i, term in enumerate(terminals):
        if term.startswith('Constant'):
            try:
                value = float(term.split('(')[1].split(')')[0])
                const_to_idx[value] = n_features + len(const_values)
                const_values.append(value)
            except:
                pass
    
    # 递归转换表达式树为 program 列表
    program_list = []
    
    def expr_to_program(expr_obj):
        """递归转换表达式对象为 program 列表"""
        from alphagen.data.expression import (
            Feature, Constant, UnaryOperator, BinaryOperator, 
            RollingOperator, PairRollingOperator
        )
        
        # 处理 Feature（终端节点）
        if isinstance(expr_obj, Feature):
            feature_name = expr_obj._feature.name.lower()
            # 映射特征名到 terminal 索引
            idx = feature_to_idx.get(feature_name, 0)
            program_list.append(idx)
            return
        
        # 处理 Constant（终端节点）
        if isinstance(expr_obj, Constant):
            value = expr_obj._value
            # 查找常量在 terminals 中的位置
            if value in const_to_idx:
                idx = const_to_idx[value]
                program_list.append(idx)
            else:
                # 如果常量不在 terminals 中，直接使用 float 值
                program_list.append(float(value))
            return
        
        # 处理操作符
        op_class = type(expr_obj)
        op_name = op_class.__name__
        
        # 处理 RollingOperator（需要匹配时间窗口；窗口固化在函数名中，arity=1）
        if isinstance(expr_obj, RollingOperator):
            delta_time = expr_obj._delta_time
            # 查找匹配时间窗口的函数（10, 20, 30, 40, 50）
            # 找到最接近的时间窗口
            windows = [10, 20, 30, 40, 50]
            closest_window = min(windows, key=lambda x: abs(x - abs(delta_time)))
            
            func_name = f'{op_name}{closest_window}'
            if func_name in func_name_map:
                func = func_name_map[func_name]
                program_list.append(func)
                # 添加操作数
                expr_to_program(expr_obj._operand)
            else:
                # 如果找不到匹配的函数，使用一个占位符一元函数（arity=1）
                unary_funcs = [f for f in function_set if f.arity == 1]
                if unary_funcs:
                    program_list.append(unary_funcs[0])  # 使用第一个一元函数作为占位符
                    expr_to_program(expr_obj._operand)
                else:
                    # 如果连一元函数都没有，至少添加操作数（虽然不完整，但不会崩溃）
                    expr_to_program(expr_obj._operand)
            return
        
        # 处理 PairRollingOperator（需要匹配时间窗口；窗口固化在函数名中，arity=2）
        if isinstance(expr_obj, PairRollingOperator):
            delta_time = expr_obj._delta_time
            windows = [10, 20, 30, 40, 50]
            closest_window = min(windows, key=lambda x: abs(x - abs(delta_time)))
            
            func_name = f'{op_name}{closest_window}'
            if func_name in func_name_map:
                func = func_name_map[func_name]
                program_list.append(func)
                # 添加两个操作数
                expr_to_program(expr_obj._lhs)
                expr_to_program(expr_obj._rhs)
            else:
                # 如果找不到匹配的函数，使用一个占位符二元函数（arity=2）
                binary_funcs = [f for f in function_set if f.arity == 2]
                if binary_funcs:
                    program_list.append(binary_funcs[0])  # 使用第一个二元函数作为占位符
                    expr_to_program(expr_obj._lhs)
                    expr_to_program(expr_obj._rhs)
                else:
                    # 如果连二元函数都没有，至少添加操作数
                    expr_to_program(expr_obj._lhs)
                    expr_to_program(expr_obj._rhs)
            return
        
        # 处理 UnaryOperator
        if isinstance(expr_obj, UnaryOperator):
            if op_name in func_name_map:
                func = func_name_map[op_name]
                program_list.append(func)
                expr_to_program(expr_obj._operand)
            else:
                # 如果找不到映射，使用第一个一元函数作为占位符
                unary_funcs = [f for f in function_set if f.arity == 1]
                if unary_funcs:
                    program_list.append(unary_funcs[0])
                    expr_to_program(expr_obj._operand)
            return
        
        # 处理 BinaryOperator
        if isinstance(expr_obj, BinaryOperator):
            if op_name in func_name_map:
                func = func_name_map[op_name]
                program_list.append(func)
                expr_to_program(expr_obj._lhs)
                expr_to_program(expr_obj._rhs)
            else:
                # 如果找不到映射，使用第一个二元函数作为占位符
                binary_funcs = [f for f in function_set if f.arity == 2]
                if binary_funcs:
                    program_list.append(binary_funcs[0])
                    expr_to_program(expr_obj._lhs)
                    expr_to_program(expr_obj._rhs)
            return
    
    # 开始转换
    expr_to_program(expr)
    return program_list


class SeedFactorProgram:
    """
    包装类，用于将种子因子表达式字符串包装成类似_Program的对象
    这样可以在GP的初始种群中使用种子因子，并参与遗传操作
    
    注意：这个类需要模拟gplearn的_Program对象的基本接口，包括program属性
    """
    def __init__(self, expr_str, raw_fitness=None, est_gp=None, terminals=None, n_features=None):
        self.expr_str = expr_str
        self.raw_fitness_ = raw_fitness if raw_fitness is not None else cache.get(expr_str, -1.0)
        self.fitness_ = self.raw_fitness_
        self.parents = {'method': 'Seed Factor', 'parent_idx': None, 'parent_nodes': []}
        self.length_ = expr_str.count('(') + expr_str.count(')')
        self.depth_ = self._estimate_depth()
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None
        self.est_gp = est_gp  # 保存est_gp引用以便访问必要属性
        self._n_features = n_features  # 保存 n_features 以便后续使用
        
        # 将 AlphaSAGE 表达式转换为 gplearn 的 program 列表
        if est_gp is not None and terminals is not None and n_features is not None:
            try:
                function_set = est_gp._function_set
                self.program = convert_alphagen_to_gplearn_program(
                    expr_str, function_set, terminals, n_features
                )
                
                # 验证 program 是否完整
                if not self._validate_program_structure(self.program, function_set):
                    print(f'[SeedFactorProgram] 警告：转换后的 program 结构不完整: {expr_str[:50]}...')
                    # 如果验证失败，尝试创建一个简单的有效 program
                    unary_funcs = [f for f in function_set if f.arity == 1]
                    if unary_funcs:
                        self.program = [unary_funcs[0], 0]  # 简单的有效占位符
                    else:
                        self.program = [0]  # 单个终端节点
                
                # 更新 length_ 为 program 的实际长度
                self.length_ = len(self.program)
            except Exception as e:
                print(f'[SeedFactorProgram] 转换 program 失败: {expr_str[:50]}..., 错误: {e}')
                import traceback
                traceback.print_exc()
                # 如果转换失败，创建一个占位符 program
                unary_funcs = [f for f in est_gp._function_set if f.arity == 1] if est_gp._function_set else []
                if unary_funcs:
                    self.program = [unary_funcs[0], 0]  # 简单的有效占位符
                else:
                    self.program = [0]
        else:
            # 延迟初始化，稍后设置
            self.program = None
            self._terminals = terminals
            self._n_features = n_features
    
    def _estimate_depth(self):
        """估算表达式的深度"""
        depth = 0
        max_depth = 0
        for char in self.expr_str:
            if char == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ')':
                depth -= 1
        return max_depth
    
    def _validate_program_structure(self, program, function_set):
        """验证 program 结构是否完整（类似 gplearn 的 validate_program）"""
        if not program or len(program) == 0:
            return False
        
        # 检查函数节点：通过是否有 arity 属性来判断
        terminals = [0]
        for node in program:
            # 检查是否是函数节点
            if hasattr(node, 'arity'):
                arity = node.arity
                terminals.append(arity)
            else:
                # 终端节点（int 或 float）
                if len(terminals) == 0:
                    return False
                terminals[-1] -= 1
                while len(terminals) > 0 and terminals[-1] == 0:
                    terminals.pop()
                    if len(terminals) > 0:
                        terminals[-1] -= 1
        
        # 验证结果：terminals 应该等于 [-1]（表示所有函数都有足够的参数）
        return terminals == [-1]
    
    def __str__(self):
        """返回表达式字符串，这样_metric函数可以识别它"""
        return self.expr_str
    
    def execute(self, X):
        """执行表达式，返回因子值"""
        try:
            expr = eval(self.expr_str)
            factor = expr.evaluate(data)
            factor = normalize_by_day(factor)
            # 转换为numpy数组格式，与gplearn兼容
            if hasattr(factor, 'cpu'):
                factor_np = factor.cpu().numpy()
            else:
                factor_np = np.array(factor)
            # 确保返回一维数组
            if len(factor_np.shape) > 1:
                factor_np = factor_np.flatten()
            return factor_np
        except Exception as e:
            print(f'[SeedFactorProgram] 执行失败: {self.expr_str[:50]}..., 错误: {e}')
            return np.zeros(X.shape[0]) if len(X.shape) > 0 else np.array([0.0])
    
    def fitness(self, parsimony_coefficient=None):
        """计算适应度"""
        if parsimony_coefficient is None:
            parsimony_coefficient = 0.0  # 默认不使用parsimony
        penalty = parsimony_coefficient * self.length_ * (1 if self.raw_fitness_ >= 0 else -1)
        return self.raw_fitness_ - penalty
    
    def get_all_indices(self, n_samples=None, max_samples=None, random_state=None):
        """模拟_Program的get_all_indices方法"""
        # 返回所有样本的索引（不使用子采样）
        if n_samples is None:
            return np.arange(self._n_samples) if self._n_samples else np.array([]), np.array([])
        indices = np.arange(n_samples)
        return indices, np.array([])
    
    def reproduce(self):
        """复制自身（用于reproduction操作）

        gplearn 期望遗传算子返回的是“扁平 program 列表”，而不是 Program 对象。
        _Program.reproduce() 的语义也是返回 copy(self.program)。
        """
        from copy import copy
        return copy(self.program) if self.program is not None else []
    
    def get_subtree(self, random_state, program=None):
        """获取随机子树（与 gplearn 原生实现一致）
        
        算法原理：
        - 前序遍历的扁平表示中，子树是连续的
        - stack 表示"还需要多少个节点才能完成当前子树"
        - 初始 stack=1（需要找一个完整子树）
        - 遇到函数节点时 stack += arity（函数需要 arity 个子节点）
        - 每处理一个节点 end += 1
        - 当 stack == end - start 时，子树刚好完整
        """
        if program is None:
            program = self.program
        
        if program is None or len(program) == 0:
            return 0, 1
        
        # 选择函数节点（90%概率）或终端节点（10%概率）
        # 遵循 Koza (1992) 的方法
        func_indices = [i for i, node in enumerate(program) 
                       if hasattr(node, 'arity')]
        
        if len(func_indices) == 0:
            # 如果没有函数节点（只有终端），返回整个 program
            return 0, len(program)
        
        # 90% 概率选择函数节点，10% 概率选择终端节点
        if random_state.uniform() < 0.9:
            start = func_indices[random_state.randint(len(func_indices))]
        else:
            # 选择随机位置
            start = random_state.randint(len(program))
        
        # 找到子树的范围（与 gplearn 原生实现一致）
        stack = 1
        end = start
        while stack > end - start:
            if end >= len(program):
                # 防止越界，返回到 program 末尾
                break
            node = program[end]
            # 只有函数节点增加 stack，终端节点不改变 stack
            if hasattr(node, 'arity'):
                stack += node.arity
            end += 1
        
        return start, end
    
    def crossover(self, donor, random_state):
        """执行交叉操作"""
        from copy import copy
        
        if self.program is None or len(self.program) == 0:
            # 如果 program 为空，返回自身
            return copy(self.program) if self.program else [], [], []
        
        # 获取子树
        start, end = self.get_subtree(random_state)
        # 确保 end > start
        if end <= start:
            end = min(start + 1, len(self.program))
        removed = list(range(start, end))
        
        # 处理 donor（可能是 _Program 对象或 SeedFactorProgram 或列表）
        if hasattr(donor, 'program'):
            donor_program = donor.program
        elif isinstance(donor, list):
            donor_program = donor
        else:
            # 如果无法处理，返回自身
            return copy(self.program), removed, []
        
        if donor_program is None or len(donor_program) == 0:
            return copy(self.program), removed, []
        
        # 获取 donor 的子树
        donor_start, donor_end = self.get_subtree(random_state, donor_program)
        # 确保 donor_end > donor_start
        if donor_end <= donor_start:
            donor_end = min(donor_start + 1, len(donor_program))
        donor_removed = list(set(range(len(donor_program))) - set(range(donor_start, donor_end)))
        
        # 执行交叉
        new_program = copy(self.program[:start]) + copy(donor_program[donor_start:donor_end]) + copy(self.program[end:])
        
        # 验证新 program 的结构
        if not self._validate_program_structure(new_program, self.est_gp._function_set if self.est_gp else []):
            # 如果交叉后的 program 不完整，返回自身（reproduction）
            return copy(self.program), [], []
        
        return new_program, removed, donor_removed
    
    def subtree_mutation(self, random_state):
        """执行子树变异
        
        生成一个随机子树（chicken）来替换当前程序的一个随机子树
        """
        # 生成随机子树作为 donor
        chicken = self._generate_random_subtree(random_state)
        
        # 使用 crossover 实现 subtree mutation
        return self.crossover(chicken, random_state)
    
    def _generate_random_subtree(self, random_state, max_depth=4):
        """生成一个随机子树
        
        Args:
            random_state: 随机数生成器
            max_depth: 子树的最大深度
            
        Returns:
            list: 随机子树的 program 列表
        """
        if self.est_gp is None or not self.est_gp._function_set:
            # 如果没有函数集，返回一个简单的终端
            n_features = self._n_features if self._n_features else 5
            return [random_state.randint(n_features)]
        
        n_features = self._n_features
        if n_features is None:
            if hasattr(self.est_gp, 'n_features_in_'):
                n_features = self.est_gp.n_features_in_
            elif hasattr(self.est_gp, 'n_features'):
                n_features = self.est_gp.n_features
            else:
                n_features = 5
        
        function_set = self.est_gp._function_set
        
        # 尝试使用 gplearn 的 _Program 类生成随机程序
        try:
            from gplearn._program import _Program
            temp_program = _Program(
                function_set=function_set,
                arities=self.est_gp._arities,
                init_depth=(2, max_depth),
                init_method='half and half',
                n_features=n_features,
                const_range=self.est_gp.const_range,
                metric=self.est_gp._metric,
                p_point_replace=self.est_gp.p_point_replace,
                parsimony_coefficient=self.est_gp.parsimony_coefficient,
                random_state=random_state
            )
            return temp_program.program
        except Exception:
            pass
        
        # 如果无法使用 _Program，手动生成随机子树
        return self._build_random_tree(random_state, function_set, n_features, max_depth)
    
    def _build_random_tree(self, random_state, function_set, n_features, max_depth, current_depth=0):
        """递归构建随机树
        
        Args:
            random_state: 随机数生成器
            function_set: 函数集合
            n_features: 特征数量
            max_depth: 最大深度
            current_depth: 当前深度
            
        Returns:
            list: 随机树的 program 列表（前序遍历）
        """
        program = []
        
        # 如果达到最大深度或随机决定生成终端
        if current_depth >= max_depth or (current_depth > 0 and random_state.uniform() < 0.3):
            # 生成终端节点（特征索引）
            terminal = random_state.randint(n_features)
            return [terminal]
        
        # 生成函数节点
        func = function_set[random_state.randint(len(function_set))]
        program.append(func)
        
        # 递归生成子节点
        for _ in range(func.arity):
            child = self._build_random_tree(
                random_state, function_set, n_features, max_depth, current_depth + 1
            )
            program.extend(child)
        
        return program
    
    def hoist_mutation(self, random_state):
        """执行提升变异"""
        from copy import copy
        
        if self.program is None or len(self.program) == 0:
            return copy(self.program) if self.program else [], []
        
        # 获取子树
        start, end = self.get_subtree(random_state)
        if end <= start or end > len(self.program):
            # 如果子树无效，返回自身
            return copy(self.program), []
        
        subtree = copy(self.program[start:end])
        
        # 如果子树太短，无法进行提升变异
        if len(subtree) <= 1:
            return copy(self.program), []
        
        # 获取子树的子树
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        if sub_end <= sub_start or sub_end > len(subtree):
            return copy(self.program), []
        
        hoist = copy(subtree[sub_start:sub_end])
        
        # 确定被移除的节点
        removed = list(set(range(start, end)) - set(range(start + sub_start, start + sub_end)))
        
        # 执行提升
        new_program = copy(self.program[:start]) + hoist + copy(self.program[end:])
        
        # 验证新 program 的结构
        if not self._validate_program_structure(new_program, self.est_gp._function_set if self.est_gp else []):
            # 如果提升后的 program 不完整，返回自身（reproduction）
            return copy(self.program), []
        
        return new_program, removed
    
    def point_mutation(self, random_state):
        """执行点变异"""
        from copy import copy
        
        new_program = copy(self.program) if self.program else []
        mutated = []
        
        if not self.est_gp:
            return new_program, mutated
        
        # 随机选择要变异的节点
        for i in range(len(new_program)):
            if random_state.uniform() < (self.est_gp.p_point_replace if hasattr(self.est_gp, 'p_point_replace') else 0.1):
                node = new_program[i]
                
                # 检查是否是函数节点（通过是否有 arity 属性）
                if hasattr(node, 'arity'):
                    # 替换为相同 arity 的其他函数
                    same_arity_funcs = [f for f in self.est_gp._function_set if f.arity == node.arity]
                    if same_arity_funcs:
                        new_program[i] = same_arity_funcs[random_state.randint(len(same_arity_funcs))]
                        mutated.append(i)
                elif isinstance(node, int):
                    # 替换为其他特征或常量
                    # 获取 n_features：优先使用保存的值
                    n_features = self._n_features
                    if n_features is None:
                        if hasattr(self.est_gp, 'n_features_in_'):
                            n_features = self.est_gp.n_features_in_
                        elif hasattr(self.est_gp, 'n_features'):
                            n_features = self.est_gp.n_features
                        else:
                            n_features = 5  # 默认值
                    
                    if self.est_gp.const_range is not None:
                        # 可以是特征或常量
                        if random_state.uniform() < 0.5:
                            new_program[i] = random_state.randint(n_features)
                        else:
                            new_program[i] = random_state.uniform(*self.est_gp.const_range)
                    else:
                        # 只能是特征
                        new_program[i] = random_state.randint(n_features)
                    mutated.append(i)
                elif isinstance(node, float):
                    # 替换为其他常量
                    if self.est_gp.const_range is not None:
                        new_program[i] = random_state.uniform(*self.est_gp.const_range)
                        mutated.append(i)
        
        # 验证新 program 的结构
        if not self._validate_program_structure(new_program, self.est_gp._function_set if self.est_gp else []):
            # 如果变异后的 program 不完整，返回自身（reproduction）
            return copy(self.program) if self.program else [], []
        
        return new_program, mutated


def inject_seed_factors_to_initial_population(est_gp, seed_factors, X_train, y_train, terminals, n_features, replace_ratio=0.1):
    """
    将种子因子注入到GP的初始种群中，替换部分随机个体
    
    Args:
        est_gp: SymbolicRegressor实例，已经fit过一次
        seed_factors: 种子因子表达式字符串列表
        X_train: 训练数据X
        y_train: 训练数据y
        replace_ratio: 替换初始种群的比例（0-1之间），最多替换这么多比例的个体
    """
    if not seed_factors or not hasattr(est_gp, '_programs') or len(est_gp._programs) == 0:
        print(f'[train_GP] 无法注入种子因子：种子因子为空或GP未初始化')
        return 0
    
    initial_population = est_gp._programs[0]
    if initial_population is None or len(initial_population) == 0:
        print(f'[train_GP] 无法注入种子因子：初始种群为空')
        return 0
    
    # 计算要替换的个体数量（至少替换种子因子数量，但不超过replace_ratio比例）
    n_replace_by_ratio = int(len(initial_population) * replace_ratio)
    n_replace = min(len(seed_factors), n_replace_by_ratio, len(initial_population))
    if n_replace == 0:
        n_replace = min(len(seed_factors), len(initial_population))
    
    print(f'[train_GP] 准备将 {n_replace} 个种子因子注入初始种群（总数: {len(initial_population)}，替换比例: {replace_ratio:.1%}）')
    
    # 随机选择要替换的个体索引（优先替换fitness较低的个体）
    if hasattr(est_gp, 'random_state') and est_gp.random_state is not None:
        random_state = est_gp.random_state
    else:
        random_state = np.random.RandomState(42)
    
    # 按fitness排序，优先替换fitness较低的个体
    fitness_scores = [p.raw_fitness_ if hasattr(p, 'raw_fitness_') and p.raw_fitness_ is not None else -999 
                      for p in initial_population]
    sorted_indices = np.argsort(fitness_scores)  # 升序排列，fitness低的在前
    
    # 选择fitness最低的n_replace个个体进行替换
    replace_indices = sorted_indices[:n_replace]
    
    # 创建种子因子程序对象并替换
    replaced_count = 0
    for i, idx in enumerate(replace_indices):
        if i >= len(seed_factors):
            break
        
        expr_str = seed_factors[i]
        # 获取该种子因子的IC值（应该已经在cache中，因为inject_seed_factors_to_cache已经计算过）
        seed_ic = cache.get(expr_str, None)
        
        if seed_ic is None:
            # 如果不在cache中，尝试计算（这种情况不应该发生，因为inject_seed_factors_to_cache已经处理过）
            print(f'[train_GP] 警告：种子因子 {i} 不在cache中，尝试计算...')
            try:
                expr = eval(expr_str)
                factor = expr.evaluate(data)
                factor = normalize_by_day(factor)
                seed_ic = batch_pearsonr(factor, target_factor)
                seed_ic = torch.nan_to_num(seed_ic).mean().item()
                cache[expr_str] = seed_ic
            except Exception as e:
                print(f'[train_GP] 计算种子因子 {i} IC失败: {e}，跳过该因子')
                continue  # IC计算失败，跳过该因子不注入
        
        # 跳过IC值无效的因子（如之前计算失败被设为-1.0的）
        if seed_ic <= -1.0 or np.isnan(seed_ic):
            print(f'[train_GP] 种子因子 {i} IC无效 ({seed_ic})，跳过该因子')
            continue
        
        # 创建种子因子程序对象，传入 terminals 和 n_features
        seed_program = SeedFactorProgram(
            expr_str, raw_fitness=seed_ic, est_gp=est_gp,
            terminals=terminals, n_features=n_features
        )
        
        # 设置必要的属性以模拟_Program对象
        if hasattr(initial_population[0], '_n_samples'):
            seed_program._n_samples = initial_population[0]._n_samples
        if hasattr(initial_population[0], '_max_samples'):
            seed_program._max_samples = initial_population[0]._max_samples
        
        # 替换初始种群中的个体
        old_fitness = fitness_scores[idx]
        initial_population[idx] = seed_program
        replaced_count += 1
        print(f'[train_GP] 已替换初始种群个体 {idx}: 旧fitness={old_fitness:.4f} -> 新fitness={seed_ic:.4f}, {expr_str[:60]}...')
    
    print(f'[train_GP] 成功将 {replaced_count} 个种子因子注入初始种群')
    return replaced_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', type=str, default='csi300')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-end-year', type=int, default=2020)
    parser.add_argument('--freq', type=str, default='day')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--seed-factors-file', type=str, default='',
                        help='种子因子 JSON 文件路径，用于指导 GP 初始种群')
    args = parser.parse_args()
    run(args)
