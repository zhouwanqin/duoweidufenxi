import streamlit as st
import pandas as pd
import numpy as np
import warnings
from modules.utils import chinese_font_setup, save_df_download, plot_scree, kmo_bartlett_text
from modules.factor_analysis import run_factor_pipeline

# 忽略 sklearn 的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

st.title("② 因子分析")

chinese_font_setup()

st.markdown("可以直接使用上一页输出的特征矩阵，也可手动上传。")
up = st.file_uploader("（可选）上传特征矩阵 Excel（不上传则使用上一页结果）", type=["xlsx"])

with st.sidebar:
    st.header("特征筛选参数")
    mean_threshold = st.number_input("均值阈值（移除均值过低的特征）", min_value=0.0, max_value=1.0, value=0.0, step=0.01,
                                    help="对已归一化/标准化特征常设为 0。若有稀疏特征可设为 0.01~0.05 以剔除。")
    corr_threshold = st.number_input("相关性阈值（|r| > 阈值则剔除后者）", min_value=0.5, max_value=0.99, value=0.95, step=0.01,
                                    help="过高相关的特征会引发多重共线性。默认 0.95。")

    st.header("因子分析参数")
    method = st.selectbox("因子提取方法", ["uls", "ml", "minres", "pa"], index=0,
                          help="uls 最小二乘；ml 最大似然；minres 最小残差；pa 主轴法。")
    rotation = st.selectbox("旋转方法", ["promax", "varimax", "oblimin", "quartimax", "none"], index=0,
                            help="常用 promax（斜交）或 varimax（正交）。")
    communality = st.number_input("communality 阈值", min_value=0.1, max_value=0.9, value=0.4, step=0.05,
                                  help="低于该值的特征会在循环中被剔除。")
    cum_explain = st.slider("累计解释率（停止条件）", min_value=0.50, max_value=0.95, value=0.60, step=0.01,
                            help="当任一因子的累计解释率超过该值时停止迭代（提示用途）。")
    loop_times = st.text_input("循环次数（整数 / 0 / / ）", value="/",
                               help="输入整数：固定循环次数；0：不循环；/：按停止条件迭代。")

run_btn = st.button("🚀 开始因子分析", type="primary", width="stretch")

if run_btn:
    # 获取数据
    if up is not None:
        df = pd.read_excel(up)
    else:
        if 'features_df' not in st.session_state:
            st.error("未找到特征矩阵，请先在 ① 页面生成或在此上传。")
            st.stop()
        df = st.session_state['features_df']

    st.write("数据预览：", df.shape)
    st.dataframe(df.head(15), width="stretch")

    with st.status("运行因子分析流水线……", expanded=True) as status:
        try:
            results = run_factor_pipeline(
                features_df=df,
                mean_threshold=float(mean_threshold),
                corr_threshold=float(corr_threshold),
                communality=float(communality),
                method=method,
                rotation=(None if rotation == "none" else rotation),
                cum_explain=float(cum_explain),
                loop_input=loop_times.strip(),
            )
        except ValueError as ve:
            st.error(str(ve))
            st.stop()
        except Exception as e:
            st.exception(e)
            st.stop()

        status.update(label="分析完成。", state="complete")

    st.success("✅ 因子分析完成")
    # 展示结果
    st.subheader("KMO/Bartlett")
    st.code(kmo_bartlett_text(results['kmo'], results['bartlett'], results['df_bartlett']), language="text")

    st.subheader("碎石图")
    fig = plot_scree(results['eigenvalues'])
    st.pyplot(fig)

    st.subheader("因子贡献率表")
    st.dataframe(results['variance_table'], width="stretch")
    save_df_download(results['variance_table'], "因子贡献率表.xlsx", "💾 下载：因子贡献率表")

    st.subheader("模式矩阵（载荷，经筛选后排序）")
    st.dataframe(results['pattern_matrix'].head(50), width="stretch")
    save_df_download(results['pattern_matrix'], "模式矩阵.xlsx", "💾 下载：模式矩阵")

    st.subheader("各文本因子得分")
    st.dataframe(results['factor_scores'].head(50), width="stretch")
    save_df_download(results['factor_scores'], "各文本因子得分.xlsx", "💾 下载：各文本因子得分")

    st.subheader("因子载荷相关性表")
    st.dataframe(results['loadings_corr'], width="stretch")
    save_df_download(results['loadings_corr'], "因子相关性表.xlsx", "💾 下载：因子相关性表")

    st.info("如需复现旧版行为，可在侧边栏调节参数并重新运行。")