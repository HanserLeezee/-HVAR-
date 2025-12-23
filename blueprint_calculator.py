#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽图纸概率计算器 v2.0 - 基于Beta-Binomial分布模型
修正了数据解析错误，重新拟合模型

功能:
1. 输入期望图纸数量，返回所需PT的概率分布
2. 输入当前状态，计算达到目标还需的PT分布  
3. 输入当前状态，计算对应结果的概率
"""

import numpy as np
from scipy import stats
from scipy.special import gammaln, betaln
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 概率模型参数 (v2.0 修正版) ====================
# 修正数据解析后重新拟合的Beta分布参数
ALPHA = 7.6481          # Beta分布α参数
BETA_PARAM = 55.2280    # Beta分布β参数
MEAN_SUCCESS_RATE = ALPHA / (ALPHA + BETA_PARAM)  # 约0.1216 (12.16%)

# 基础游戏参数
BASE_SCORE = 12000     # 基础积分
SCORE_PER_DRAW = 180   # 每次抽取消耗积分


def print_header():
    """打印程序标题"""
    print("=" * 60)
    print("      抽图纸概率计算器 v2.0 - 修正版Beta分布模型")
    print("=" * 60)
    print(f"\n模型参数:")
    print(f"  - Beta分布: α = {ALPHA:.2f}, β = {BETA_PARAM:.2f}")
    print(f"  - 单次成功率均值: {MEAN_SUCCESS_RATE:.4f} ({MEAN_SUCCESS_RATE*100:.2f}%)")
    print(f"  - 每次抽取消耗: {SCORE_PER_DRAW} PT")
    print(f"  - 基础积分: {BASE_SCORE} PT")
    print("=" * 60)


def pt_to_draws(pt):
    """将PT转换为抽取次数"""
    return int((pt - BASE_SCORE) / SCORE_PER_DRAW)


def draws_to_pt(draws):
    """将抽取次数转换为PT"""
    return draws * SCORE_PER_DRAW + BASE_SCORE


def log_comb(n, k):
    """计算 log(C(n,k)) 使用 gammaln 避免溢出"""
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def calculate_blueprint_distribution(n_draws, max_blueprints=None):
    """
    计算给定抽取次数下，获得各数量图纸的概率分布
    使用Beta-Binomial分布（考虑成功率的不确定性）
    """
    if max_blueprints is None:
        max_blueprints = min(n_draws, 500)
    
    probs = np.zeros(max_blueprints + 1)
    
    for k in range(max_blueprints + 1):
        log_prob = (
            log_comb(n_draws, k) +
            betaln(k + ALPHA, n_draws - k + BETA_PARAM) -
            betaln(ALPHA, BETA_PARAM)
        )
        probs[k] = np.exp(log_prob)
    
    return probs


def option1_pt_for_target_blueprints():
    """选项1: 输入期望图纸数量，返回所需PT的概率分布"""
    print("\n" + "=" * 60)
    print("选项1: 输入期望图纸数量，计算所需PT的概率分布")
    print("=" * 60)
    
    try:
        target = int(input("请输入期望获得的图纸数量: "))
        if target <= 0:
            print("错误：图纸数量必须大于0")
            return
    except ValueError:
        print("错误：请输入有效的整数")
        return
    
    print(f"\n正在计算获得 {target} 张图纸所需PT的概率分布...")
    
    expected_draws = int(target / MEAN_SUCCESS_RATE)
    min_draws = max(target, int(expected_draws * 0.3))
    max_draws = int(expected_draws * 2.5)
    
    draws_range = np.arange(min_draws, max_draws + 1)
    cumulative_probs = []
    
    for n in draws_range:
        probs = calculate_blueprint_distribution(n, max_blueprints=target + 50)
        prob_at_least_target = 1 - np.sum(probs[:target])
        cumulative_probs.append(prob_at_least_target)
    
    cumulative_probs = np.array(cumulative_probs)
    pt_range = draws_to_pt(draws_range)
    
    percentiles = [0.25, 0.50, 0.75, 0.90, 0.95]
    percentile_pts = {}
    
    for p in percentiles:
        idx = np.searchsorted(cumulative_probs, p)
        if idx < len(pt_range):
            percentile_pts[p] = pt_range[idx]
        else:
            percentile_pts[p] = pt_range[-1]
    
    print("\n" + "-" * 40)
    print(f"获得至少 {target} 张图纸所需PT估算:")
    print("-" * 40)
    for p in percentiles:
        pt = percentile_pts[p]
        print(f"  {int(p*100):>3}%概率达成: {pt:>10,.0f} PT ({pt/10000:.1f}w)")
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(pt_range / 10000, cumulative_probs, 'b-', linewidth=2)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50%')
    ax1.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90%')
    ax1.axvline(x=percentile_pts[0.50]/10000, color='r', linestyle=':', alpha=0.5)
    ax1.axvline(x=percentile_pts[0.90]/10000, color='orange', linestyle=':', alpha=0.5)
    ax1.set_xlabel('PT (wan)', fontsize=12)
    ax1.set_ylabel(f'P(blueprints >= {target})', fontsize=12)
    ax1.set_title(f'PT needed for {target} blueprints (CDF)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    pdf_approx = np.diff(cumulative_probs)
    pt_mid = (pt_range[:-1] + pt_range[1:]) / 2
    ax2.fill_between(pt_mid / 10000, pdf_approx, alpha=0.4, color='blue')
    ax2.plot(pt_mid / 10000, pdf_approx, 'b-', linewidth=1.5)
    ax2.axvline(x=percentile_pts[0.50]/10000, color='r', linestyle='--', 
                label=f'Median: {percentile_pts[0.50]/10000:.1f}w')
    ax2.set_xlabel('PT (wan)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('PT Distribution (PDF)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'option1_target{target}_result.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {filename}")
    plt.close()


def option2_additional_pt_needed():
    """选项2: 输入当前图纸和PT，计算达到目标还需的PT分布"""
    print("\n" + "=" * 60)
    print("选项2: 输入当前状态，计算达到目标还需的PT分布")
    print("=" * 60)
    
    try:
        current_blueprints = int(input("请输入当前已有的图纸数量: "))
        current_pt = int(input("请输入当前已有的PT (直接输入数字，如90000): "))
        target_blueprints = int(input("请输入期望获得的图纸数量: "))
        
        if target_blueprints <= current_blueprints:
            print("错误：目标图纸数量必须大于当前图纸数量")
            return
        if current_pt < BASE_SCORE:
            print(f"错误：PT不能小于基础值 {BASE_SCORE}")
            return
    except ValueError:
        print("错误：请输入有效的整数")
        return
    
    needed_blueprints = target_blueprints - current_blueprints
    print(f"\n当前状态: {current_blueprints} 张图纸, {current_pt} PT ({current_pt/10000:.1f}w)")
    print(f"目标: {target_blueprints} 张图纸")
    print(f"还需获得: {needed_blueprints} 张图纸")
    print(f"\n正在计算...")
    
    expected_additional_draws = int(needed_blueprints / MEAN_SUCCESS_RATE)
    min_draws = max(needed_blueprints, int(expected_additional_draws * 0.3))
    max_draws = int(expected_additional_draws * 2.5)
    
    draws_range = np.arange(min_draws, max_draws + 1)
    cumulative_probs = []
    
    for n in draws_range:
        probs = calculate_blueprint_distribution(n, max_blueprints=needed_blueprints + 50)
        prob_at_least_target = 1 - np.sum(probs[:needed_blueprints])
        cumulative_probs.append(prob_at_least_target)
    
    cumulative_probs = np.array(cumulative_probs)
    
    additional_pt_range = draws_range * SCORE_PER_DRAW
    total_pt_range = current_pt + additional_pt_range
    
    percentiles = [0.25, 0.50, 0.75, 0.90, 0.95]
    percentile_additional_pts = {}
    percentile_total_pts = {}
    
    for p in percentiles:
        idx = np.searchsorted(cumulative_probs, p)
        if idx < len(additional_pt_range):
            percentile_additional_pts[p] = additional_pt_range[idx]
            percentile_total_pts[p] = total_pt_range[idx]
        else:
            percentile_additional_pts[p] = additional_pt_range[-1]
            percentile_total_pts[p] = total_pt_range[-1]
    
    print("\n" + "-" * 55)
    print(f"从 {current_blueprints} 张达到 {target_blueprints} 张图纸:")
    print("-" * 55)
    print(f"{'概率':<8} {'还需PT':<18} {'总PT':<18}")
    print("-" * 55)
    for p in percentiles:
        add_pt = percentile_additional_pts[p]
        total_pt = percentile_total_pts[p]
        print(f"{int(p*100):>3}%    {add_pt:>8,.0f} ({add_pt/10000:>5.1f}w)    {total_pt:>8,.0f} ({total_pt/10000:>5.1f}w)")
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.plot(additional_pt_range / 10000, cumulative_probs, 'b-', linewidth=2)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50%')
    ax1.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90%')
    ax1.set_xlabel('Additional PT (wan)', fontsize=12)
    ax1.set_ylabel(f'P(get {needed_blueprints} more)', fontsize=12)
    ax1.set_title(f'{current_blueprints} -> {target_blueprints}: Additional PT', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    ax2 = axes[1]
    ax2.plot(total_pt_range / 10000, cumulative_probs, 'g-', linewidth=2)
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50%')
    ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90%')
    ax2.axvline(x=current_pt/10000, color='purple', linestyle=':', linewidth=2, 
                label=f'Current: {current_pt/10000:.1f}w')
    ax2.set_xlabel('Total PT (wan)', fontsize=12)
    ax2.set_ylabel(f'P(reach {target_blueprints})', fontsize=12)
    ax2.set_title(f'Total PT for {target_blueprints} blueprints', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    filename = f'option2_{current_blueprints}to{target_blueprints}_result.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {filename}")
    plt.close()


def option3_probability_of_current_state():
    """选项3: 计算当前状态(X图纸, Y PT)的概率"""
    print("\n" + "=" * 60)
    print("选项3: 计算当前状态的概率")
    print("=" * 60)
    
    try:
        current_blueprints = int(input("请输入当前已有的图纸数量 X: "))
        current_pt = int(input("请输入当前已有的PT Y (直接输入数字，如90000): "))
        
        if current_pt < BASE_SCORE:
            print(f"错误：PT不能小于基础值 {BASE_SCORE}")
            return
        if current_blueprints < 0:
            print("错误：图纸数量不能为负")
            return
    except ValueError:
        print("错误：请输入有效的整数")
        return
    
    n_draws = pt_to_draws(current_pt)
    
    if n_draws <= 0:
        print(f"\n当前PT {current_pt} 对应抽取次数为0，未进行任何抽取")
        print(f"图纸为 {current_blueprints} 张的概率: {'100%' if current_blueprints == 0 else '0%'}")
        return
    
    if current_blueprints > n_draws:
        print(f"\n错误：抽取 {n_draws} 次不可能获得 {current_blueprints} 张图纸")
        return
    
    print(f"\n当前状态分析:")
    print(f"  - 图纸数量: {current_blueprints}")
    print(f"  - 总PT: {current_pt} ({current_pt/10000:.2f}w)")
    print(f"  - 抽取次数: {n_draws}")
    print(f"  - 实际成功率: {current_blueprints/n_draws:.4f} ({current_blueprints/n_draws*100:.2f}%)")
    
    probs = calculate_blueprint_distribution(n_draws, max_blueprints=min(n_draws, 500))
    
    if current_blueprints < len(probs):
        exact_prob = probs[current_blueprints]
    else:
        exact_prob = 0
    
    prob_less = np.sum(probs[:current_blueprints]) if current_blueprints > 0 else 0
    prob_less_or_equal = prob_less + exact_prob
    prob_more = 1 - prob_less_or_equal
    
    print("\n" + "-" * 50)
    print("概率分析结果:")
    print("-" * 50)
    print(f"  P(X = {current_blueprints}) = {exact_prob:.6f} ({exact_prob*100:.4f}%)")
    print(f"  P(X < {current_blueprints}) = {prob_less:.6f} ({prob_less*100:.4f}%)")
    print(f"  P(X ≤ {current_blueprints}) = {prob_less_or_equal:.6f} ({prob_less_or_equal*100:.4f}%)")
    print(f"  P(X > {current_blueprints}) = {prob_more:.6f} ({prob_more*100:.4f}%)")
    
    print("\n" + "-" * 50)
    print("运气评估:")
    print("-" * 50)
    expected = n_draws * MEAN_SUCCESS_RATE
    print(f"  期望图纸数: {expected:.1f}")
    print(f"  实际图纸数: {current_blueprints}")
    diff = current_blueprints - expected
    if diff > 0:
        print(f"  比期望多: {diff:.1f} 张 (运气好!)")
    elif diff < 0:
        print(f"  比期望少: {-diff:.1f} 张 (运气不佳)")
    else:
        print(f"  与期望持平 (正常运气)")
    
    percentile = prob_less_or_equal * 100
    if percentile > 80:
        luck_assessment = "非常好运"
    elif percentile > 60:
        luck_assessment = "运气较好"
    elif percentile > 40:
        luck_assessment = "运气正常"
    elif percentile > 20:
        luck_assessment = "运气较差"
    else:
        luck_assessment = "非常倒霉"
    
    print(f"  百分位数: {percentile:.1f}% (超过了{percentile:.1f}%的人)")
    print(f"  运气评价: {luck_assessment}")
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    max_k = min(int(expected * 2) + 10, len(probs) - 1)
    k_range = np.arange(0, max_k + 1)
    ax1.bar(k_range, probs[:max_k + 1], alpha=0.6, color='blue', label='Distribution')
    ax1.axvline(x=current_blueprints, color='red', linestyle='--', linewidth=2, 
                label=f'Current: {current_blueprints}')
    ax1.axvline(x=expected, color='green', linestyle=':', linewidth=2, 
                label=f'Expected: {expected:.1f}')
    ax1.set_xlabel('Blueprints', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title(f'Distribution after {n_draws} draws', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if current_blueprints <= max_k:
        ax1.bar(current_blueprints, probs[current_blueprints], color='red', alpha=0.8)
    
    ax2 = axes[1]
    cdf = np.cumsum(probs[:max_k + 1])
    ax2.plot(k_range, cdf, 'b-', linewidth=2)
    ax2.axvline(x=current_blueprints, color='red', linestyle='--', linewidth=2,
                label=f'Current: {current_blueprints}')
    ax2.axhline(y=prob_less_or_equal, color='red', linestyle=':', alpha=0.7)
    ax2.scatter([current_blueprints], [prob_less_or_equal], color='red', s=100, zorder=5)
    ax2.set_xlabel('Blueprints', fontsize=12)
    ax2.set_ylabel('CDF P(X <= k)', fontsize=12)
    ax2.set_title(f'Percentile: {percentile:.1f}%', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    filename = f'option3_{current_blueprints}bp_{current_pt}pt_result.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {filename}")
    plt.close()


def main():
    """主程序"""
    print_header()
    
    while True:
        print("\n" + "=" * 60)
        print("请选择功能:")
        print("=" * 60)
        print("1. 输入期望图纸数量，计算所需PT的概率分布")
        print("2. 输入当前状态，计算达到目标还需的PT分布")
        print("3. 输入当前状态，计算对应结果的概率")
        print("0. 退出程序")
        print("-" * 60)
        
        choice = input("请输入选项 (0-3): ").strip()
        
        if choice == '1':
            option1_pt_for_target_blueprints()
        elif choice == '2':
            option2_additional_pt_needed()
        elif choice == '3':
            option3_probability_of_current_state()
        elif choice == '0':
            print("\n感谢使用，再见！")
            break
        else:
            print("无效选项，请重新输入")


if __name__ == "__main__":
    main()
