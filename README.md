
# 文本因子分析助手（Streamlit）

## 使用方式
1. 安装依赖：`pip install -r requirements.txt`
2. 运行：`streamlit run app.py`
3. 在页面 **① 特征提取** 上传分词后的 `zip`（包含 `.txt`，且为“词/词性”格式）。
4. 选择是否切片（及长度），点击开始生成 **特征矩阵**。
5. 进入 **② 因子分析**，调整参数并查看结果（碎石图、模式矩阵、因子得分等）。

> 词单与语言特征表已内置路径：
> - 词单：/home/zhuyu/zhouwanqin/streamlit_text_factor_app/词单20241107
> - 语言特征表：/home/zhuyu/zhouwanqin/streamlit_text_factor_app/20241107_语言特征整理.xlsx

## 结构
- `app.py`：应用入口与说明
- `pages/1_特征提取.py`：上传 zip、特征提取与导出
- `pages/2_因子分析.py`：因子分析参数设置、结果展示与导出
- `modules/utils.py`：通用工具（中文字体、下载、绘图等）
- `modules/text_features.py`：特征工程（切片、词表匹配、正则统计、基础特征）
- `modules/factor_analysis.py`：因子分析流水线（筛选、FA、模式矩阵、因子得分）
- `requirements.txt`：依赖
