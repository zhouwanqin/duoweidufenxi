
import os, io, zipfile, tempfile, re, math
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def extract_zip_to_dir(uploaded_file, target_dir: str):
    # Extract uploaded zip (Streamlit UploadedFile) to target_dir.
    if isinstance(uploaded_file, (str, bytes)):
        with zipfile.ZipFile(uploaded_file, 'r') as zf:
            zf.extractall(target_dir)
    else:
        zbytes = uploaded_file.read()
        with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
            zf.extractall(target_dir)

def chinese_font_setup():
    # Configure matplotlib to display Chinese and minus sign.
    plt.rcParams["font.sans-serif"] = ["SimHei", "STHeiti", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

def save_df_download(df: pd.DataFrame, filename: str, label: str):
    bio = io.BytesIO()
    df.to_excel(bio, index=True)
    st.download_button(label=label, data=bio.getvalue(), file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def load_feature_column_names(feature_excel_path: str):
    # Read the internal feature excel and return the list of feature column names.
    xls = pd.ExcelFile(feature_excel_path)
    df = xls.parse()
    candidates = ["20241101整理", "特征清单", "features", "FeatureList"]
    col = None
    for c in candidates:
        if c in df.columns:
            col = c
            break
    if col is None:
        col = df.columns[0]
    col_values = [v for v in df[col].tolist()[1:] if pd.notna(v)]
    return col_values

def plot_scree(eigenvalues):
    chinese_font_setup()
    fig, ax = plt.subplots(figsize=(6,4), dpi=140)
    xs = list(range(1, len(eigenvalues)+1))
    ax.plot(xs, eigenvalues, marker='o')
    ax.scatter(xs, eigenvalues)
    ax.set_title("Scree Plot ")
    ax.set_xlabel("factor count")
    ax.set_ylabel("eigenvalue")
    ax.grid(True, alpha=0.3)
    return fig

def kmo_bartlett_text(kmo_model, bartlett_tuple, df_bartlett):
    chi_square_value, p_value = bartlett_tuple
    return (f"KMO: {kmo_model:.3f}\n"
            f"p_value: {p_value}\n"
            f"chi_square: {chi_square_value:.3f}\n"
            f"df: {df_bartlett}")
