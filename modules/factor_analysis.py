import numpy as np
import pandas as pd
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer
from scipy.stats import zscore

def drop_low_mean(df: pd.DataFrame, threshold: float):
    # 只保留数值列
    df_num = df.select_dtypes(include=["number"])
    mean_values = df_num.mean()
    to_drop = mean_values[mean_values < threshold].index
    return df_num.drop(columns=to_drop), list(to_drop)

def drop_high_corr(df: pd.DataFrame, threshold: float):
    # 只保留数值列
    df_num = df.select_dtypes(include=["number"])
    c = df_num.corr(method="pearson").abs()
    upper = c.where(np.triu(np.ones(c.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df_num.drop(columns=to_drop), to_drop, c

def eigenvalues_for(df: pd.DataFrame):
    fa0 = FactorAnalyzer(rotation=None)
    fa0.fit(df)
    EigenValue, value = fa0.get_eigenvalues()
    return EigenValue

def select_factor_number(eigs, adjust=0):
    return max(1, int((eigs > 1).sum()) - int(adjust))

def retain_max_abs_value_row(row):
    row = pd.to_numeric(row, errors="coerce")
    abs_row = row.abs()
    max_index = abs_row.idxmax(skipna=True)
    return row.where(row.index == max_index)

def run_factor_pipeline(features_df: pd.DataFrame,
                        mean_threshold: float = 0.0,
                        corr_threshold: float = 0.95,
                        communality: float = 0.4,
                        method: str = "uls",
                        rotation: str = "promax",
                        cum_explain: float = 0.60,
                        loop_input: str = "/"):
    # 1. 复制数据并清洗（移除全空列）
    Data = features_df.copy()
    Data = Data.loc[:, Data.notna().any()]

    # 🚨 关键修改：只保留数值型列
    Data = Data.select_dtypes(include=["number"])

    # 移除方差为 0 的列
    Data = Data.loc[:, Data.var() > 0]

    # 删除全空列
    Data = Data.dropna(axis=1, how="all")

    # 填充残余 NaN
    Data = Data.fillna(0)

    # 替换 inf/-inf
    Data = Data.replace([np.inf, -np.inf], 0)

    if Data.shape[1] == 0:
        raise ValueError("上传的数据中没有可用的数值型特征，请检查文件。")

    # 2. 均值筛选
    Data, drop_mean = drop_low_mean(Data, mean_threshold)

    # 3. 高相关特征筛选
    Data, drop_corr, corr_mat = drop_high_corr(Data, corr_threshold)

    # 4. 准备循环剔除 communality 过低项
    if loop_input == "/":
        loop_times = -1
    elif loop_input == "0":
        loop_times = 0
    else:
        loop_times = int(loop_input)
    current_loop = 0

    fa = FactorAnalyzer(method=method, rotation=rotation)
    fa.fit(Data)
    new_Data = Data.copy()
    dropped_features_list = []

    while True and loop_input != "0" and (loop_times == -1 or current_loop < loop_times):
        current_loop += 1
        communalities = pd.DataFrame(fa.get_communalities(), index=new_Data.columns)
        features_comm = list(communalities[communalities[0] > communality].index)
        dropped = list(set(new_Data.columns) - set(features_comm))
        dropped_features_list.extend(dropped)
        if len(features_comm) == len(new_Data.columns):
            break
        new_Data = new_Data[features_comm]
        fa.fit(new_Data)

    # KMO / Bartlett
    kmo_all, kmo_model = calculate_kmo(new_Data)
    chi_square_value, p_value = calculate_bartlett_sphericity(new_Data)
    num_vars = new_Data.shape[1]
    df_bartlett = num_vars * (num_vars - 1) // 2

    # Eigenvalues and number of factors
    eigs = eigenvalues_for(new_Data)
    factor_number = select_factor_number(np.array(eigs), adjust=0)

    # 提取因子
    fa = FactorAnalyzer(n_factors=factor_number, method=method, rotation=rotation)
    fa.fit(new_Data)

    fac_loadings = pd.DataFrame(
        fa.loadings_,
        columns=[f"FAC{i}" for i in range(1, factor_number+1)],
        index=new_Data.columns
    )

    # 因子载荷相关
    loadings_corr = fac_loadings.corr(method="pearson")

    # 解释方差表
    idx = ["SS Loadings", "Proportion Variance", "Cumulative Variance"]
    df_variance = pd.DataFrame(
        data=fa.get_factor_variance(),
        index=idx,
        columns=[f"FAC{i}" for i in range(1, factor_number+1)]
    )

    # 构建模式矩阵（保留每行绝对值最大项）
    pattern_matrix = fac_loadings.copy().fillna(0)
    pattern_matrix = pattern_matrix.apply(retain_max_abs_value_row, axis=1).dropna(how="all")
    sort_columns = pattern_matrix.columns.tolist()
    pattern_matrix = pattern_matrix.sort_values(by=sort_columns, ascending=[False]*len(sort_columns))

    # 计算 Z 分数
    normalized_data = new_Data.apply(zscore, axis=0)
    ordered_columns = pattern_matrix.index
    normalized_data = normalized_data[ordered_columns]

    # 正负号 dict
    data_dict = {col: {idx: ("positive" if val > 0 else "negative") for idx, val in pattern_matrix[col].dropna().items()}
                 for col in pattern_matrix.columns}

    factor_score = pd.DataFrame(index=normalized_data.index, columns=pattern_matrix.columns)
    for i in factor_score.index:
        for key, mapping in data_dict.items():
            adjusted_values = []
            for col, sign in mapping.items():
                if col in normalized_data.columns:
                    v = normalized_data.loc[i, col]
                    adjusted_values.append(v if sign == "positive" else -v)
            if adjusted_values:
                factor_score.at[i, key] = sum(adjusted_values) / len(adjusted_values)
    factor_score = factor_score.astype(float)

    results = dict(
        data_after_filters=new_Data,
        dropped_low_mean=drop_mean,
        dropped_high_corr=drop_corr,
        corr_matrix=corr_mat,
        eigenvalues=eigs,
        kmo=kmo_model,
        bartlett=(chi_square_value, p_value),
        df_bartlett=df_bartlett,
        variance_table=df_variance,
        pattern_matrix=pattern_matrix,
        factor_scores=factor_score,
        loadings_corr=loadings_corr,
    )
    return results