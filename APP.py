import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib

# 加载模型
model_path = "stacking_regressor_model.pkl"
stacking_regressor = joblib.load(model_path)

# 设置页面配置和标题
st.set_page_config(layout="wide", page_title="Stacking 模型预测与 SHAP 可视化", page_icon="📊")

st.title("📊 Stacking 模型预测与 SHAP 可视化分析")
st.write("""
通过输入特征值进行模型预测，并结合 SHAP 分析结果，了解特征对模型预测的贡献。
""")

# 左侧侧边栏输入区域
st.sidebar.header("特征输入区域")
st.sidebar.write("请输入特征值：")

# 定义特征输入范围
DOSE = st.sidebar.number_input("特征 DOSE (范围: 150-1800)", min_value=150, max_value=1800, value=300)
BMI = st.sidebar.number_input("特征 BMI (范围: 10.81-40.23)", min_value=10.81, max_value=40.23, value=15.34)
CL = st.sidebar.number_input("特征 CL (范围: 0.90-2.30)", min_value=0.90, max_value=2.30, value=1.55)



# 添加预测按钮
predict_button = st.sidebar.button("进行预测")

# 主页面用于结果展示
if predict_button:
    st.header("预测结果")
    try:
        # 将输入特征转换为模型所需格式
        input_array = np.array([DOSE, BMI, CL]).reshape(1, -1)

        # 模型预测
        prediction = stacking_regressor.predict(input_array)[0]

        # 显示预测结果
        st.success(f"预测结果：{prediction:.2f}")
    except Exception as e:
        st.error(f"预测时发生错误：{e}")

# 可视化展示
st.header("SHAP 可视化分析")
st.write("""
以下图表展示了模型的 SHAP 分析结果，包括第一层基学习器、第二层元学习器以及整个 Stacking 模型的特征贡献。
""")

# 第一层基学习器 SHAP 可视化
st.subheader("1. 第一层基学习器")
st.write("基学习器（RandomForest、XGB、LGBM 等）的特征贡献分析。")
first_layer_img = "summary_plot.png"
try:
    img1 = Image.open(first_layer_img)
    st.image(img1, caption="第一层基学习器的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到第一层基学习器的 SHAP 图像文件。")

# 第二层元学习器 SHAP 可视化
st.subheader("2. 第二层元学习器")
st.write("元学习器（Linear Regression）的输入特征贡献分析。")
meta_layer_img = "SHAP Contribution Analysis for the Meta-Learner in the Second Layer of Stacking Regressor.png"
try:
    img2 = Image.open(meta_layer_img)
    st.image(img2, caption="第二层元学习器的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到第二层元学习器的 SHAP 图像文件。")

# 整体 Stacking 模型 SHAP 可视化
st.subheader("3. 整体 Stacking 模型")
st.write("整个 Stacking 模型的特征贡献分析。")
overall_img = "Based on the overall feature contribution analysis of SHAP to the stacking model.png"
try:
    img3 = Image.open(overall_img)
    st.image(img3, caption="整体 Stacking 模型的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到整体 Stacking 模型的 SHAP 图像文件。")

# 页脚
st.markdown("---")
st.header("总结")
st.write("""
通过本页面，您可以：
1. 使用输入特征值进行实时预测。
2. 直观地理解第一层基学习器、第二层元学习器以及整体 Stacking 模型的特征贡献情况。
这些分析有助于深入理解模型的预测逻辑和特征的重要性。
""")
