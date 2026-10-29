
import base64, json
from io import BytesIO

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from features import extract_features
from segment import segment_eyelid

st.set_page_config(page_title="AnemiaScan", page_icon="🩸", layout="wide")

# ---------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700;800&family=Figtree:wght@400;500;600&display=swap');
:root{--ink:#1D1B2F;--plum:#2B2145;--rose:#C23A57;--blush:#FBE6EB;--teal:#1F8A8A;--amber:#E8A33D;--line:#E4E1EC;--mute:#6B6880;}
html, body, [class*="css"], .stApp{font-family:'Figtree',system-ui,sans-serif;}
.stApp{background:#F6F5F9;}
#MainMenu, footer, header{visibility:hidden;}
.block-container{padding-top:1.4rem;max-width:1150px;}
h1,h2,h3{font-family:'Bricolage Grotesque','Figtree',sans-serif;color:var(--ink);letter-spacing:-.01em;}

.hero{position:relative;overflow:hidden;background:var(--plum);border-radius:22px;padding:2.2rem 2.4rem;margin-bottom:1.2rem;color:#fff;}
.hero h1{color:#fff;font-size:3rem;line-height:1.05;margin:0 0 .5rem 0;font-weight:800;}
.hero p{color:#D9D3EC;font-size:1.08rem;max-width:520px;margin:0 0 1.1rem 0;line-height:1.5;}
.hero .badge{display:inline-block;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.28);color:#fff;border-radius:999px;padding:.32rem .9rem;font-size:.88rem;font-weight:500;}
.hero svg{position:absolute;right:-30px;top:-10px;width:430px;height:auto;opacity:.95;}
@media (max-width:760px){.hero svg{opacity:.18;}.hero h1{font-size:2.3rem;}}

.card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:1.2rem 1.3rem;margin-bottom:1rem;}
.card h3{margin:.1rem 0 .5rem 0;font-size:1.2rem;}
.card p, .card li{color:#3a3750;font-size:.98rem;line-height:1.55;}
.muted{color:var(--mute);font-size:.9rem;}

.imgbox{background:#fff;border:1px solid var(--line);border-radius:14px;padding:.6rem;text-align:center;}
.imgbox img{max-width:100%;max-height:260px;border-radius:9px;}
.imgbox .cap{font-size:.85rem;color:var(--mute);margin-top:.35rem;}

.verdict{border-radius:18px;padding:1.3rem 1.5rem;border:1px solid;margin-bottom:1rem;}
.verdict h2{margin:0 0 .25rem 0;font-size:1.7rem;}
.verdict p{margin:0;font-size:1rem;line-height:1.5;}
.v-high{background:#FCEBEE;border-color:#EFA9B6;} .v-high h2{color:#9E2440;} .v-high p{color:#5b2430;}
.v-low{background:#E6F4F3;border-color:#97D0CD;} .v-low h2{color:#146666;} .v-low p{color:#1f4a49;}
.v-mid{background:#FFF4DF;border-color:#EDC987;} .v-mid h2{color:#946011;} .v-mid p{color:#5a4318;}

.scale-wrap{background:#fff;border:1px solid var(--line);border-radius:16px;padding:1.1rem 1.3rem 1rem 1.3rem;margin-bottom:1rem;}
.scale-title{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.9rem;}
.scale-title b{font-family:'Bricolage Grotesque',sans-serif;font-size:2.1rem;color:var(--ink);}
.scale{position:relative;height:30px;border-radius:999px;background:linear-gradient(90deg,#B3263F 0%,#D9566B 28%,#EE8D98 55%,#F7C2C6 80%,#FCE7E6 100%);border:1px solid #d9cfd6;}
.tick{position:absolute;top:-5px;bottom:-5px;width:2px;background:rgba(29,27,47,.35);}
.marker{position:absolute;top:-9px;width:6px;height:48px;border-radius:4px;background:var(--ink);box-shadow:0 0 0 3px #fff;transform:translateX(-50%);}
.scale-labels{display:flex;justify-content:space-between;margin-top:.7rem;font-size:.85rem;color:var(--mute);}

.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.8rem;margin:.4rem 0 1rem 0;}
.step{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1rem;}
.step .n{width:30px;height:30px;border-radius:50%;background:var(--rose);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;margin-bottom:.5rem;}
.step b{display:block;color:var(--ink);margin-bottom:.2rem;font-family:'Bricolage Grotesque',sans-serif;font-size:1.05rem;}
.step span{color:#4a4760;font-size:.92rem;line-height:1.45;}

.stat{background:#fff;border:1px solid var(--line);border-radius:14px;padding:.9rem 1rem;text-align:left;}
.stat b{display:block;font-family:'Bricolage Grotesque',sans-serif;font-size:1.8rem;color:var(--rose);}
.stat span{color:var(--mute);font-size:.88rem;}

.notice{background:#F0EDF8;border-left:4px solid var(--plum);border-radius:10px;padding:.8rem 1rem;color:#2c2745;font-size:.95rem;margin:.2rem 0 1rem 0;}
.empty{background:#fff;border:2px dashed #CFC8E2;border-radius:18px;padding:2.2rem 1.5rem;text-align:center;color:#4a4760;}
.empty b{font-family:'Bricolage Grotesque',sans-serif;font-size:1.35rem;color:var(--ink);display:block;margin-bottom:.4rem;}

section[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line);}
.stTabs [data-baseweb="tab-list"]{gap:.4rem;}
.stTabs [data-baseweb="tab"]{background:#fff;border:1px solid var(--line);border-radius:999px;padding:.35rem 1rem;color:var(--ink);}
.stTabs [aria-selected="true"]{background:var(--ink);color:#fff;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{display:none;}
</style>
"""

def html(s):
    """Render HTML safely: strip indentation so Markdown does not treat it as a code block."""
    st.markdown("\n".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)

def to_b64(img):
    buf = BytesIO(); img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def image_box(img, caption):
    html(f'<div class="imgbox"><img src="data:image/png;base64,{to_b64(img)}"/><div class="cap">{caption}</div></div>')

CRESCENT = """
<svg viewBox="0 0 430 260" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<defs><linearGradient id="g" x1="0" y1="1" x2="1" y2="0">
<stop offset="0" stop-color="#F4A3AE"/><stop offset=".55" stop-color="#D9566B"/><stop offset="1" stop-color="#8E1F3A"/></linearGradient></defs>
<path d="M10 170 C 120 250, 300 210, 420 20 C 330 150, 150 160, 10 170 Z" fill="url(#g)"/>
<path d="M40 150 C 140 215, 290 180, 395 40" fill="none" stroke="#fff" stroke-opacity=".35" stroke-width="2" stroke-dasharray="4 7"/>
<circle cx="395" cy="40" r="5" fill="#fff" fill-opacity=".85"/>
</svg>
"""

# ---------------------------------------------------------------- model
@st.cache_resource
def load_model():
    return joblib.load("anemia_model.joblib")

@st.cache_data
def load_metrics():
    try:
        return json.load(open("metrics.json"))["MLP (ANN)"]
    except Exception:
        return None

model, metrics = load_model(), load_metrics()
LOW, HIGH = 0.35, 0.65     # below LOW: less likely | above HIGH: possible anemia | between: unclear

# ---------------------------------------------------------------- page
st.markdown(CSS, unsafe_allow_html=True)
html(f"""
<div class="hero">
{CRESCENT}
<h1>AnemiaScan</h1>
<p>A quick first check for anemia from a photo of the inner eyelid, made for places where blood tests are hard to reach.</p>
<span class="badge">Screening aid, not a diagnosis</span>
</div>
""")

with st.sidebar:
    st.markdown("### 📸 Taking a good photo")
    st.markdown("- Use daylight, no flash\n- Turn off beauty filters\n- Pull the lower eyelid down gently\n- Get close so the red inner lid fills the frame\n- Keep the image sharp and still")
    st.markdown("---")
    st.markdown("### Model on unseen test images")
    if metrics:
        st.metric("Accuracy", f"{metrics['accuracy']:.0%}")
        st.metric("Anemic cases caught", f"{metrics['sensitivity']:.0%}")
   

tab_scan, tab_how, tab_about = st.tabs(["🔍 Scan", "⚙️ How it works", "ℹ️ About and limits"])

# ---------------------------------------------------------------- scan tab
with tab_scan:
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown("### Add an eyelid photo")
        source = st.radio("Photo source", ["Upload a photo", "Use camera"], horizontal=True, label_visibility="collapsed")
        if source == "Upload a photo":
            file = st.file_uploader("Eyelid photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        else:
            file = st.camera_input("Take a photo", label_visibility="collapsed")

    with right:
        if not file:
            html("""
            <div class="empty"><b>Your result will appear here</b>
            Add a close-up photo of the inner lower eyelid to start. The app finds the eyelid, reads its colour and gives a screening score.</div>
            """)
        else:
            img = ImageOps.exif_transpose(Image.open(file)).convert("RGBA")
            img.thumbnail((1000, 1000))
            seg, mask = segment_eyelid(img)
            coverage = float(mask.mean())
            if mask.sum() > 0:
                redness = float(cv2.cvtColor(np.array(seg)[..., :3], cv2.COLOR_RGB2LAB)[..., 1][mask > 0].mean())
            else:
                redness = 0.0
            found = 0.02 <= coverage <= 0.92 and redness >= 133   # 128 = neutral; real eyelids are clearly redder

            c1, c2 = st.columns(2)
            with c1:
                image_box(img.convert("RGB"), "Your photo")
            with c2:
                canvas = Image.new("RGBA", seg.size, (236, 232, 245, 255)); canvas.alpha_composite(seg)
                image_box(canvas.convert("RGB"), "Eyelid found by the app")
            st.write("")

            if not found:
                html("""<div class="verdict v-mid"><h2>Eyelid not found</h2>
                <p>The app could not pick out a clear red eyelid area. Retake the photo closer, in daylight, with the lower eyelid pulled down.</p></div>""")
            else:
                prob = float(model.predict_proba(extract_features(seg)[None, :])[0, 1])
                if prob >= HIGH:
                    cls, title, msg = "v-high", "Possible anemia", "The eyelid colour looks paler than usual. Please get a haemoglobin blood test to find out for sure."
                elif prob <= LOW:
                    cls, title, msg = "v-low", "Anemia less likely", "The eyelid colour looks typical. This is not a guarantee. If you feel tired, dizzy or weak, still see a health worker."
                else:
                    cls, title, msg = "v-mid", "Unclear result", "The score is in the middle. Retake the photo in better light, or get a blood test."
                html(f'<div class="verdict {cls}"><h2>{title}</h2><p>{msg}</p></div>')

                pos = max(1.0, min(99.0, prob * 100))
                html(f"""
                <div class="scale-wrap">
                <div class="scale-title"><span class="muted">Anemia likelihood score</span><b>{prob:.0%}</b></div>
                <div class="scale"><div class="tick" style="left:{LOW*100:.0f}%"></div><div class="tick" style="left:{HIGH*100:.0f}%"></div>
                <div class="marker" style="left:{pos:.1f}%"></div></div>
                <div class="scale-labels"><span>Less likely</span><span>Unclear</span><span>Possible anemia</span></div>
                </div>
                """)
                if coverage > 0.6:
                    st.warning("The app may have picked up skin or background along with the eyelid. Crop closer to the red inner lid for a more reliable score.")
                html("""<div class="notice">This is a screening aid, not a diagnosis. Only a blood test can confirm anemia.</div>""")

# ---------------------------------------------------------------- how it works
with tab_how:
    st.markdown("### From photo to score")
    html("""
    <div class="steps">
    <div class="step"><div class="n">1</div><b>Photo</b><span>A phone picture of the inner lower eyelid.</span></div>
    <div class="step"><div class="n">2</div><b>Find the eyelid</b><span>OpenCV picks out the reddest region and removes skin and background.</span></div>
    <div class="step"><div class="n">3</div><b>Read the colour</b><span>21 numbers describe the average colour and its variation in RGB, HSV and Lab.</span></div>
    <div class="step"><div class="n">4</div><b>Neural network</b><span>A small scikit-learn MLP compares those numbers with what it learned from labelled eyelids.</span></div>
    <div class="step"><div class="n">5</div><b>Score</b><span>You get a likelihood score and a plain-language message.</span></div>
    </div>
    """)
    html("""<div class="card"><h3>Why colour?</h3>
    <p>Haemoglobin gives blood its red colour. When it is low, the inner eyelid tends to look paler, which is why health workers sometimes check it by eye. The model learns this pattern from images instead.</p></div>""")

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown("### Know the limits")
    if metrics:
        a, b, c = st.columns(3)
        a.markdown(f'<div class="stat"><b>{metrics["accuracy"]:.0%}</b><span>test accuracy</span></div>', unsafe_allow_html=True)
        b.markdown(f'<div class="stat"><b>{metrics["sensitivity"]:.0%}</b><span>anemic cases caught</span></div>', unsafe_allow_html=True)
        c.markdown(f'<div class="stat"><b>{metrics["auc"]:.2f}</b><span>AUC</span></div>', unsafe_allow_html=True)
        st.write("")
    html("""
    <div class="card"><h3>What to keep in mind</h3>
    <ul>
    <li>Healthy people are sometimes flagged, so a flag means "get a blood test", not "you have anemia".</li>
    <li>The model learned from about 520 eyelids from Ghana, so it may behave differently on other populations and phone cameras.</li>
    <li>Lighting, flash and filters change eyelid colour and can change the score.</li>
    <li>The training labels are only anemic or non-anemic, with no haemoglobin values.</li>
    </ul></div>
    <div class="card"><h3>Who should use it</h3>
    <p>Community health workers and families who want a quick first check before deciding on a blood test. It never replaces a doctor or a laboratory test.</p></div>
    """)