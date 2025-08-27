
import streamlit as st

st.set_page_config(
    page_title="文本因子分析助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("📚 文本因子分析助手")
    st.markdown("**流程：** 上传分词后的 zip → ① 特征提取 → ② 因子分析")
    st.markdown("---")
    st.caption("提示：上传的 zip 应包含分词且带词性标注的 .txt 文件（如 词/pos 形式）。")

st.title("欢迎使用：文本因子分析助手")
st.markdown(
    """
    **使用方法：**  
    1. 在左侧页面选择 **① 特征提取**，上传分词后的 zip（可选是否切片）。  
    2. 点击开始处理，等待生成 **特征矩阵**。  
    3. 进入 **② 因子分析**，调整参数，查看与下载结果。  
    """
)

st.info("所有词单与语言特征表已内置，无需手动设置。")
st.success("准备就绪 👉 请从左侧进入页面 ① 开始处理。")
