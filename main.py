import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ========================
# 1️⃣ Page Config + CSS
# ========================
st.set_page_config(page_title="Samsung Predictor", layout="centered")

# --- Custom CSS ---
st.markdown(
    """
    <style>
    /* พื้นหลังทั้งหน้า */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Card input */
    .card {
        background: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    /* ปุ่ม */
    button[kind="primary"] {
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========================
# 2️⃣ Title (Styled)
# ========================
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        padding: 18px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        text-align: center;
        margin-bottom: 20px;
    ">
        <h1 style="
            color: white;
            font-weight: bold;
            font-size: 36px;
            margin: 0;
        ">
            📱 Guess Your Samsung!
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("กรอกข้อมูลของคุณ แล้วเราจะทำนาย Samsung รุ่นที่เหมาะกับคุณ")
st.markdown("---")

# ========================
# 3️⃣ Dataset
# ========================
try:
    df = pd.read_pickle("Cleansamsung.pkl")
except:
    data = {
        'งบประมาณ (บาท)': [0, 1, 2, 3],
        'การใช้งานหลัก': [1, 2, 3, 4],
        'ความสำคัญของกล้อง': [1, 2, 3, 1],
        'ขนาดหน้าจอ': [1, 2, 3, 1],
        'ต้องการ S Pen': [0, 2, 0, 2],
        'ชอบจอพับได้': [0, 2, 0, 2],
        'ใช้ Samsung รุ่นไหนอยู่ปัจจุบัน': ['Galaxy A', 'Galaxy S', 'Galaxy Z', 'Galaxy Note']
    }
    df = pd.DataFrame(data)

#with st.expander("📊 ดู Dataset ที่ใช้ฝึกโมเดล"):
#    st.dataframe(df)

X = df[['งบประมาณ (บาท)', 'การใช้งานหลัก', 'ความสำคัญของกล้อง',
        'ขนาดหน้าจอ', 'ต้องการ S Pen', 'ชอบจอพับได้']]
y = df['ใช้ Samsung รุ่นไหนอยู่ปัจจุบัน']

model = DecisionTreeClassifier()
model.fit(X, y)

# ========================
# 4️⃣ Input Section
# ========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📝 กรอกข้อมูลของคุณ")

col1, col2, col3 = st.columns(3)

with col1:
    budget = st.radio(
        "💰 งบประมาณ",
        options=[0, 1, 3, 2],
        format_func=lambda x: {
            0: "0 – 9,999 บาท",
            1: "10,000 – 25,000 บาท",
            3: "25,001 – 40,000 บาท",
            2: "40,001 – 60,000 บาท"
        }[x]
    )
    main_use = st.radio(
        "🖥️ การใช้งานหลัก",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "ใช้ทั่วไป",
            2: "ทำงาน",
            3: "เล่นเกม",
            4: "ถ่ายรูป"
        }[x]
    )

with col2:
    camera = st.radio(
        "📷 ความสำคัญของกล้อง",
        options=[1, 3, 2],
        format_func=lambda x: {
            1: "สูง",
            3: "กลาง",
            2: "ต่ำ"
        }[x]
    )
    screen = st.radio(
        "📺 ขนาดหน้าจอ",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "เล็ก (<6.0\")",
            2: "กลาง (6.0 – 6.7\")",
            3: "ใหญ่ (>6.7\")"
        }[x]
    )

with col3:
    s_pen = st.radio(
        "✏️ ต้องการ S Pen",
        options=[0, 2],
        format_func=lambda x: {
            0: "ไม่ต้องการ",
            2: "ต้องการ"
        }[x]
    )
    foldable = st.radio(
        "📐 ชอบจอพับได้",
        options=[0, 2],
        format_func=lambda x: {
            0: "ไม่ชอบ",
            2: "ชอบ"
        }[x]
    )
st.markdown("</div>", unsafe_allow_html=True)

# ========================
# 5️⃣ Prediction
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

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.success(f"🎉 แนะนำ Samsung รุ่น: **{predicted_model[0]}**", icon="✅")
    st.markdown("</div>", unsafe_allow_html=True)