import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import os

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="NexusMind — Personality Decoder v01",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');


html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080808;
    color: #f0f0f0;
}
.stApp { background-color: #080808; }

h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

.hero {
    text-align: center;
    padding: 2.5rem 0 1rem 0;
}
.hero h1 {
    font-size: 3rem;
    background: linear-gradient(135deg, #00ff87, #60efff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}
.hero .version {
    font-family: 'Space Mono', monospace;
    color: #00ff87;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    margin-bottom: 0.5rem;
}
.hero p {
    font-family: 'Space Mono', monospace;
    color: #555;
    font-size: 0.82rem;
    letter-spacing: 0.05em;
}

.accuracy-badge {
    display: inline-block;
    background: #0d1f0d;
    border: 1px solid #00ff87;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #00ff87;
    margin-top: 0.5rem;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #444;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    margin-top: 2rem;
}

.result-card {
    background: linear-gradient(135deg, #0d0d0d, #111);
    border: 1px solid #1a1a1a;
    border-left: 4px solid #00ff87;
    border-radius: 12px;
    padding: 2rem;
    margin-top: 1.5rem;
}
.result-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00ff87, #60efff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.confidence {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #666;
    margin-bottom: 1rem;
}
.result-desc {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.9;
    color: #bbb;
}

.name-input label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #555 !important;
}

.footer {
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #00ff87;
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid #1a1a1a;
    opacity: 0.7;
}

.stButton > button {
    background: linear-gradient(135deg, #00ff87, #60efff);
    color: #080808;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.stMetric {
    background: linear-gradient(135deg, #0d0d0d, #111);
    border: 1px solid #1a1a1a;
    border-left: 4px solid #00ff87;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}

.stSelectbox {
    background: #0d0d0d;
}

[data-testid="stMetricValue"] {
    color: #00ff87 !important;
    font-size: 1.8rem !important;
}

[data-testid="stMetricLabel"] {
    color: #888 !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv('personality_c.csv')
    #df = pd.read_csv('personalityy.csv')

    le = LabelEncoder()
    y = le.fit_transform(df['personality'])
    X = df.drop('personality', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train_scaled, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_scaled))

    return model, scaler, le, X.columns.tolist(), round(acc * 100, 1)

model, scaler, le, feature_cols, accuracy = train_model()

# ─────────────────────────────────────────
# DESCRIPTIONS
# ─────────────────────────────────────────
descriptions = {
    "INTROVERT": {
        "emoji": "🔋",
        "desc": "You're a deep diver who recharges in silence.\nYou don't hate people — you just find them expensive to maintain.\nOne good conversation > ten small talks.\nSolitude isn't loneliness. It's your workshop."
    },
    "EXTROVERT": {
        "emoji": "⚡",
        "desc": "You're a human spark plug. You run on buzz, banter, and energy exchange.\nYou feel most alive when there's noise, people, and movement around you.\nAlone time feels like a phone on 1% — you recharge with people."
    },
    "AMBIVERT": {
        "emoji": "🎭",
        "desc": "You're the balanced one.\nSocial when the vibe calls for it. Quiet when it doesn't.\nYou adapt. You read the room. You're the most flexible personality type."
    },
    "OMNIVERT": {
        "emoji": "🌀",
        "desc": "You swing between extremes — and that's not a flaw.\nHighly social sometimes, completely withdrawn at others.\nContext drives everything for you. You contain multitudes."
    }
}

# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="version">V 0 1 &nbsp;•&nbsp; N E X U S M I N D</div>
    <h1>🧠 PERSONALITY<br>DECODER</h1>
    <p>I don't judge. I just notice patterns.</p>
    <div class="accuracy-badge">Model Accuracy: {accuracy}%</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.info("Adjust sliders based on your REAL habits — not ideal ones.")
# ─────────────────────────────────────────
# NAME INPUT
# ─────────────────────────────────────────
st.subheader(' Your Name...')
name = st.text_input("", placeholder="Enter your name (optional)", label_visibility="collapsed")

# ─────────────────────────────────────────
# SLIDERS
# ─────────────────────────────────────────
st.subheader("Your Daily Rhythm")

col1, col2 = st.columns(2)

with col1:
    sleep = st.slider("🌙 Sleep hours per night", 1.0, 12.0, 7.0, 0.2)
    study = st.slider("📚 Study hours per day", 0.0, 12.0, 3.0, 0.2)
    screen = st.slider("📱 Screen time per day (hrs)", 0.0, 12.0, 4.0, 0.2)
    focus = st.slider("🎯 Focus level", 1.0, 10.0, 5.0, 0.5, help="1 = easily distracted, 10 = laser focused")
    stress = st.slider("😤 Stress level", 1.0, 10.0, 5.0, 0.5, help="1 = very calm, 10 = very stressed")

with col2:
    social = st.slider("👥 Social energy level", 1.0, 10.0, 5.0, 0.2, help="1 = drained by people, 10 = energized by people")
    energy = st.slider("⚡ Daily energy level", 1.0, 10.0, 5.0, 0.2)
    outings = st.slider("🌆 Outings per week", 0, 14, 2)
    communication = st.slider("🗣️ Communication level", 1.0, 10.0, 5.0, 0.5, help="1 = very quiet, 10 = very expressive")
    alone_time = st.slider("🏠 Alone time preference", 1.0, 10.0, 5.0, 0.5, help="1 = hate being alone, 10 = love being alone")

# ─────────────────────────────────────────
# FEATURE CHART
# ─────────────────────────────────────────
#st.markdown('<div class="section-label">// Your pattern</div>', unsafe_allow_html=True)
st.subheader("📊 Your Pattern")
input_values = [sleep, study, social, energy, screen, outings, focus, stress, communication, alone_time]
normalized = [v/12 if i in [0,1,2,3,4] else v/14 if i == 5 else v/10 for i, v in enumerate(input_values)]
labels = ['Sleep', 'Study', 'Social', 'Energy', 'Screen', 'Outings', 'Focus', 'Stress', 'Comms', 'Alone']

fig, ax = plt.subplots(figsize=(8, 2.5))
fig.patch.set_facecolor('#0d0d0d')
ax.set_facecolor('#0d0d0d')

bars = ax.barh(labels, normalized, color='#00ff87', alpha=0.8, height=0.6)
ax.set_xlim(0, 1)
ax.tick_params(colors='#555', labelsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#222')
ax.spines['left'].set_color('#222')
ax.xaxis.label.set_color('#444')
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ─────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Decode My Personality"):
    with st.spinner("Crunching your patterns..."):
        time.sleep(1.5)

    input_df = pd.DataFrame([[sleep, study, social, energy, screen, outings, focus, stress, communication, alone_time]],
                             columns=feature_cols)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)[0]
    confidence = round(max(probabilities) * 100, 1)
    personality = le.inverse_transform(prediction)[0].upper()


    probs = model.predict_proba(input_df)[0]
    pred_index = np.argmax(probs)
    pred_label = le.inverse_transform([pred_index])[0]

    sorted_idx = np.argsort(probs)[::-1]
    second_label = le.inverse_transform([sorted_idx[1]])[0]

    confidence = round(probs[pred_index] * 100, 1)

    info = descriptions.get(personality, {"emoji": "🔍", "desc": "Interesting pattern."})

    greeting = f"**{name}**, you are a..." if name else "You are a..."

    st.markdown(f"*{greeting}*")
    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">{info['emoji']} {personality}</div>
        <div class="confidence">Confidence: {confidence}%</div>
        <div class="result-desc">{info['desc'].replace(chr(10), '<br>')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    probs = model.predict_proba(input_df)[0]

    pred_idx = np.argmax(probs)
    sorted_idx = np.argsort(probs)[::-1]

    personality = le.inverse_transform([pred_idx])[0]
    secondary = le.inverse_transform([sorted_idx[1]])[0]

    confidence = round(probs[pred_idx] * 100, 1)

    # ─────────────────────────────
    # REASONING ENGINE
    # ─────────────────────────────
    reasons = []

    if social <= 4:
        reasons.append("low social energy")
    elif social >= 7:
        reasons.append("high social engagement")

    if alone_time >= 7:
        reasons.append("strong preference for solitude")
    if 0 <= alone_time <= 4:
        reasons.append("thrives in social settings")
    elif 5 <= alone_time <= 6:
        reasons.append("balanced between socializing and solitude")


    if outings <= 2:
        reasons.append("low social activity")
    elif 3 <= outings <= 5:
        reasons.append("moderate social activity")
    elif outings > 5:
        reasons.append("high social activity") 

    if communication <= 4:
        reasons.append("reserved communication style")
    if communication == 5:
        reasons.append("moderate communication style")
    if communication >5:
        reasons.append("expressive communication style")

    if stress >= 7:
        reasons.append("high stress levels")
    elif 4 < stress < 7:
        reasons.append("moderate stress levels")
    elif stress <=4:
        reasons.append("low stress levels")

    if focus >= 7:
        reasons.append("you maintain strong internal focus even in stimulating environments")

    st.write("\n\n")
   
    #st.info(f"Secondary tendency: {secondary}")
    #st.write(f"Confidence: {confidence}%")
    probs_dict = dict(zip(le.classes_, probabilities))
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    secondary = sorted_probs[1][0]

    st.info(f"Secondary tendency: {secondary}")

    st.markdown("### 🧠 Why?")
    for r in reasons:
        st.write(f"• {r}")


    probs = model.predict_proba(input_scaled)[0]

    labels = le.classes_

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    fig2.patch.set_facecolor('#0d0d0d')
    ax2.set_facecolor('#0d0d0d')

    colors = ['#00ff87', '#60efff', '#ff6a00', '#ffcc00']
    wedges, texts, autotexts = ax2.pie(probs, labels=[l.upper() for l in labels], autopct='%1.1f%%', startangle=90, wedgeprops={"linewidth":2, 'edgecolor': '#080808'}, colors=colors, textprops={'color': '#ccc', 'fontsize': 8})
    for autotext in autotexts:
        autotext.set_color('#080808')
        autotext.set_fontweight('bold')

    ax2.legend(wedges, [l.upper() for l in labels], title="Personalities", loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=8, title_fontsize=10, frameon=False, ncol=2, labelcolor='#ccc', framealpha=0)

    ax2.set_title("Personality Breakdown", color='#00ff87', fontsize=12,
                  fontfamily='monospace', pad=15)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


    st.caption("⚠️ Personality is complex. This model detects patterns, not absolute truths.")
   
    entry = {
        'name': name if name else 'Anonymous',
        'personality': personality,
        'confidence': confidence,
        'sleep': sleep, 'study': study, 'social': social,
        'energy': energy, 'screen': screen, 'outings': outings,
        'focus': focus, 'stress': stress,
        'communication': communication, 'alone_time': alone_time
    }

    entries_file = 'entries.csv'
    if os.path.exists(entries_file):
        entries_df = pd.read_csv(entries_file)
        entries_df = pd.concat([entries_df, pd.DataFrame([entry])], ignore_index=True)
    else:
        entries_df = pd.DataFrame([entry])
    entries_df.to_csv(entries_file, index=False)




st.markdown("""
<div class="footer">
    Built by Ajibola  &nbsp;•&nbsp; Guided by faith &nbsp;•&nbsp; NexusMind 🧠 v01.
</div>
""", unsafe_allow_html=True)
