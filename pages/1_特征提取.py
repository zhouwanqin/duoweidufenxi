
import streamlit as st
import os, tempfile, zipfile, io, re, math
import pandas as pd
from modules.utils import extract_zip_to_dir, chinese_font_setup, save_df_download, load_feature_column_names
from modules.text_features import build_vocabulary_df, build_features_for_corpus

st.title("① 特征提取")

# 内置资源路径（不显示给用户）
VOCAB_DIR = "/home/zhuyu/zhouwanqin/streamlit_text_factor_app/词单20241107"
FEATURE_EXCEL = "/home/zhuyu/zhouwanqin/streamlit_text_factor_app/20241107_语言特征整理.xlsx"

chinese_font_setup()

with st.expander("📦 上传分词后的 zip 文件（必选）", expanded=True):
    uploaded = st.file_uploader("只需上传一个 zip，包含已分词且带词性标注的 .txt 文件", type=["zip"])

with st.sidebar:
    st.header("切片设置")
    do_slice = st.toggle("是否进行文本切片", value=True, help="勾选后将每篇文本按设定长度切分为多个片段。")
    slice_len = st.number_input("切片长度（字符）", min_value=200, max_value=5000, value=1600, step=100,
                                help="每个切片的字符数（建议 800~2000 之间）。仅当启用切片时生效。")
    st.markdown("---")
    st.header("特征表结构")
    st.caption("自动读取内置语言特征表格，第一列为特征清单名。")

col1, col2 = st.columns([1,1])
with col1:
    st.markdown("**运行说明**")
    st.write(
        """
        - 本页将从 zip 解压文本 → （可选）切片 → 统计语言学特征 → 输出 **特征矩阵**。  
        - 支持较大语料，处理过程显示进度条。  
        - 若文本非常多，请耐心等待。
        """
    )

with col2:
    st.markdown("**当前配置**")
    st.write(f"- 词单目录：已内置")
    st.write(f"- 语言特征表：已内置")
    st.write(f"- 切片：{'开启' if do_slice else '关闭'}（长度：{slice_len}）")

start = st.button("🚀 开始提取特征", type="primary", use_container_width=True, disabled=(uploaded is None))

if start:
    if uploaded is None:
        st.error("请先上传 zip 文件。")
        st.stop()
    # 解压
    with st.status("解压并扫描文件中……", expanded=False) as status:
        tmpdir = tempfile.mkdtemp(prefix="corpus_")
        extract_zip_to_dir(uploaded, tmpdir)
        # 收集 txt
        txt_paths = []
        for root, _, files in os.walk(tmpdir):
            for f in files:
                if f.lower().endswith(".txt"):
                    txt_paths.append(os.path.join(root, f))
        if not txt_paths:
            st.error("zip 内未找到 .txt 文件，请检查。")
            st.stop()
        status.update(label=f"发现 {len(txt_paths)} 个文本文件。", state="complete")

    # 载入词单与特征列
    with st.status("载入词单与语言特征表……", expanded=False) as status:
        try:
            df_vocab = build_vocabulary_df(VOCAB_DIR)
            feature_columns = load_feature_column_names(FEATURE_EXCEL)
        except Exception as e:
            st.exception(e)
            st.stop()
        status.update(label=f"词单与特征列载入完成（特征列数：{len(feature_columns)}）。", state="complete")

    # 计算特征
    with st.status("计算语言学特征中……（可能耗时）", expanded=True) as status:
        try:
            features_df, logs = build_features_for_corpus(
                txt_paths=txt_paths,
                df_vocabulary=df_vocab,
                feature_excel_path=FEATURE_EXCEL,
                do_slice=do_slice,
                slice_length=int(slice_len),
            )
        except Exception as e:
            st.exception(e)
            st.stop()

        st.write("处理日志：")
        for line in logs:
            st.markdown(f"- {line}")
        status.update(label="统计完成。", state="complete")

    # 保存与展示
    st.success("✅ 特征矩阵已生成")
    st.write(f"形状：{features_df.shape[0]} × {features_df.shape[1]}")
    st.dataframe(features_df.head(20), use_container_width=True)

    st.session_state['features_df'] = features_df

    save_df_download(features_df, filename="features_matrix.xlsx", label="💾 下载特征矩阵（Excel）")
    st.info("接下来 → 进入左侧页面 **② 因子分析**。")
