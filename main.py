import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ========================
# 1️⃣ Title + Subtitle
# ========================
st.set_page_config(page_title="Samsung Predictor", layout="centered")
st.title("📱 Guess Your Samsung!")
st.subheader("กรอกข้อมูลของคุณ แล้วเราจะทำนาย Samsung รุ่นที่เหมาะกับคุณ")

st.markdown("---")

# ========================
# 2️⃣ โหลด dataset
# ========================
try:
    df = pd.read_pickle("Cleansamsung.pkl")
except:
    # Dataset ตัวอย่าง
    data = {
        'งบประมาณ (บาท)': [0,1,2,3],
        'การใช้งานหลัก': [1,2,3,4],
        'ความสำคัญของกล้อง': [1,2,3,1],
        'ขนาดหน้าจอ': [1,2,3,1],
        'ต้องการ S Pen': [1,2,1,2],
        'ชอบจอพับได้': [1,2,1,2],
        'ใช้ Samsung รุ่นไหนอยู่ปัจจุบัน': ['Galaxy A', 'Galaxy S', 'Galaxy Z', 'Galaxy Note']
    }
    df = pd.DataFrame(data)

st.write("📊 Samsung Data Clean")
st.dataframe(df)

# ========================
# 3️⃣ แยก Features / Target
# ========================
X = df[['งบประมาณ (บาท)','การใช้งานหลัก','ความสำคัญของกล้อง',
        'ขนาดหน้าจอ','ต้องการ S Pen','ชอบจอพับได้']]
y = df['ใช้ Samsung รุ่นไหนอยู่ปัจจุบัน']

# ========================
# 4️⃣ สร้างและฝึกโมเดล
# ========================
model = DecisionTreeClassifier()
model.fit(X, y)

st.markdown("---")

# ========================
# 5️⃣ Streamlit UI: input
# ========================
st.subheader("📝 กรอกข้อมูลของคุณ")

col1, col2, col3 = st.columns(3)
with col1:
    budget = st.number_input("💰 งบประมาณ (0-3)", min_value=0, max_value=3)
    main_use = st.number_input("🖥️ การใช้งานหลัก (1-4)", min_value=1, max_value=4)
with col2:
    camera = st.number_input("📷 ความสำคัญของกล้อง (1-3)", min_value=1, max_value=3)
    screen = st.number_input("📺 ขนาดหน้าจอ (1-3)", min_value=1, max_value=3)
with col3:
    s_pen = st.number_input("✏️ ต้องการ S Pen (1-2)", min_value=1, max_value=2)
    foldable = st.number_input("📐 ชอบจอพับได้ (1-2)", min_value=1, max_value=2)

st.markdown("<br>", unsafe_allow_html=True)

# ========================
# 6️⃣ ปุ่มทำนาย
# ========================
if st.button("🔮 ทำนาย Samsung รุ่น", use_container_width=True):
    user_input = pd.DataFrame({
        'งบประมาณ (บาท)': [budget],
        'การใช้งานหลัก': [main_use],
        'ความสำคัญของกล้อง': [camera],
        'ขนาดหน้าจอ': [screen],
        'ต้องการ S Pen': [s_pen],
        'ชอบจอพับได้': [foldable]
    })
    predicted_model = model.predict(user_input)
    
    st.markdown("---")
    st.success(f"🎉 แนะนำ Samsung รุ่น: **{predicted_model[0]}**", icon="✅")