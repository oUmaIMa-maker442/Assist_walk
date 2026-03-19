import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st
import cv2
import numpy as np
import tempfile
import json
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from vision_module import VisionModule
from src.step1_acquisition import acquire_from_pil, acquire_from_video_frame

# ═══════════════════════════════════════════════════════════
# CONFIG PAGE
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title='AssistWalk — Vision Module',
    page_icon='👁',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ═══════════════════════════════════════════════════════════
# CSS PROFESSIONNEL
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Variables ── */
:root {
    --blue-900: #0f172a;
    --blue-800: #1e293b;
    --blue-700: #1d3461;
    --blue-600: #1e3a8a;
    --blue-500: #1d4ed8;
    --blue-400: #3b82f6;
    --blue-300: #93c5fd;
    --blue-100: #dbeafe;
    --accent:   #06b6d4;
    --success:  #10b981;
    --warning:  #f59e0b;
    --danger:   #ef4444;
    --text:     #e2e8f0;
    --muted:    #94a3b8;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--blue-900) !important;
    color: var(--text) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    border-right: 1px solid rgba(59,130,246,0.2) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e293b, #1d3461) !important;
    border: 1px solid rgba(59,130,246,0.3) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(29,78,216,0.3);
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: var(--blue-300) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--blue-600), var(--blue-500)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 12px rgba(29,78,216,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(29,78,216,0.5) !important;
}

/* ── Radio ── */
[data-testid="stRadio"] label {
    background: #1e293b !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    margin: 4px !important;
    transition: all 0.2s;
}

/* ── Success/Warning/Error boxes ── */
.stSuccess {
    background: rgba(16,185,129,0.1) !important;
    border-left: 3px solid var(--success) !important;
    border-radius: 6px !important;
}
.stWarning {
    background: rgba(245,158,11,0.1) !important;
    border-left: 3px solid var(--warning) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #1e293b !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    border-radius: 10px !important;
}

/* ── Divider ── */
hr { border-color: rgba(59,130,246,0.2) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--blue-800); }
::-webkit-scrollbar-thumb { background: var(--blue-500); border-radius: 3px; }

/* ── Custom cards ── */
.aw-card {
    background: linear-gradient(135deg, #1e293b, #1d3461);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: all 0.2s;
}
.aw-card:hover {
    border-color: rgba(59,130,246,0.6);
    box-shadow: 0 4px 20px rgba(29,78,216,0.2);
}
.aw-badge-danger  { background:rgba(239,68,68,0.15);  color:#fca5a5; border:1px solid rgba(239,68,68,0.4);  padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.aw-badge-warning { background:rgba(245,158,11,0.15); color:#fcd34d; border:1px solid rgba(245,158,11,0.4); padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.aw-badge-ok      { background:rgba(16,185,129,0.15); color:#6ee7b7; border:1px solid rgba(16,185,129,0.4); padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.aw-voice {
    background: linear-gradient(135deg, rgba(6,182,212,0.1), rgba(29,78,216,0.1));
    border: 1px solid rgba(6,182,212,0.4);
    border-radius: 10px;
    padding: 16px 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    color: #67e8f9;
    margin: 8px 0;
}
.aw-header {
    background: linear-gradient(135deg, #1d3461, #1e3a8a);
    border-bottom: 2px solid rgba(59,130,246,0.4);
    padding: 20px 28px;
    border-radius: 14px;
    margin-bottom: 24px;
}
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

# Priorité et couleurs par classe
PRIORITY_MAP = {
    'person': ('HAUTE', 'danger', '#ef4444'),
    'car': ('HAUTE', 'danger', '#ef4444'),
    'truck': ('HAUTE', 'danger', '#ef4444'),
    'bus': ('HAUTE', 'danger', '#ef4444'),
    'motorcycle': ('HAUTE', 'danger', '#ef4444'),
    'bicycle': ('HAUTE', 'danger', '#ef4444'),
    'traffic light': ('HAUTE', 'danger', '#f97316'),
    'stop sign': ('HAUTE', 'danger', '#f97316'),
    'chair': ('MOYENNE', 'warning', '#f59e0b'),
    'bench': ('MOYENNE', 'warning', '#f59e0b'),
    'dog': ('BASSE', 'ok', '#10b981'),
    'cat': ('BASSE', 'ok', '#10b981'),
}

def get_priority(class_name):
    return PRIORITY_MAP.get(class_name, ('BASSE', 'ok', '#10b981'))

def estimate_distance(bbox, img_width, img_height):
    """Estimation de distance basée sur la taille de la bounding box."""
    x1, y1, x2, y2 = bbox
    box_area = (x2 - x1) * (y2 - y1)
    img_area  = img_width * img_height
    ratio = box_area / img_area if img_area > 0 else 0

    if ratio > 0.30:   return "~1m", "🔴"
    elif ratio > 0.10: return "~2-3m", "🟠"
    elif ratio > 0.04: return "~4-6m", "🟡"
    elif ratio > 0.01: return "~7-10m", "🟢"
    else:              return ">10m", "🔵"

def generate_voice_message(objects, text_boxes_count):
    """Génère le message vocal AssistWalk."""
    if not objects and text_boxes_count == 0:
        return "Aucun obstacle détecté. Voie libre."

    messages = []
    high = [o for o in objects if get_priority(o['class'])[0] == 'HAUTE']
    med  = [o for o in objects if get_priority(o['class'])[0] == 'MOYENNE']

    for obj in high[:2]:  # max 2 objets haute priorité
        dist, _ = estimate_distance(obj['bbox'], 1000, 1000)
        if obj['class'] == 'person':
            messages.append(f"Attention, personne détectée à {dist}")
        elif obj['class'] in ['car', 'truck', 'bus']:
            messages.append(f"Attention, {obj['class']} détecté à {dist}")
        elif obj['class'] == 'stop sign':
            messages.append("Panneau STOP détecté")
        elif obj['class'] == 'traffic light':
            messages.append("Feu de circulation détecté")
        else:
            messages.append(f"Attention, {obj['class']} détecté à {dist}")

    for obj in med[:1]:
        messages.append(f"{obj['class']} sur votre chemin")

    if text_boxes_count > 0:
        messages.append(f"{text_boxes_count} zone(s) de texte détectée(s)")

    return ". ".join(messages) + "."

def draw_annotations(pil_img, objects, text_boxes):
    """Dessine les bounding boxes colorées par priorité."""
    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = pil_img.size

    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        _, _, color = get_priority(obj['class'])
        dist, icon = estimate_distance(obj['bbox'], w, h)

        # Box principale
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Rectangle label
        label = f"{obj['class']} {int(obj['confidence']*100)}% {icon}{dist}"
        draw.rectangle([x1, max(0,y1-24), x1+len(label)*7+8, y1], fill=color)
        draw.text((x1+4, max(0,y1-22)), label, fill='white')

    for (x1, y1, x2, y2) in text_boxes:
        draw.rectangle([x1, y1, x2, y2], outline='#06b6d4', width=2)
        draw.rectangle([x1, y1, x1+30, y1+16], fill='#06b6d4')
        draw.text((x1+3, y1+1), "TXT", fill='white')

    return annotated

def results_to_csv(results_list):
    """Convertit les résultats en CSV."""
    lines = ["frame,classe,confiance,x1,y1,x2,y2,distance,priorite"]
    for r in results_list:
        frame = r.get('frame_num', 0)
        for obj in r.get('objects', []):
            x1,y1,x2,y2 = obj['bbox']
            dist, _ = estimate_distance(obj['bbox'], 1000, 1000)
            prio, _, _ = get_priority(obj['class'])
            lines.append(f"{frame},{obj['class']},{obj['confidence']},{x1},{y1},{x2},{y2},{dist},{prio}")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════
# CHARGEMENT DU MODULE
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_module():
    return VisionModule()

vision = load_module()

# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size:2.5rem;">👁</div>
        <div style="font-size:1.3rem; font-weight:700; color:#93c5fd; letter-spacing:0.05em;">AssistWalk</div>
        <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em;">Vision Module</div>
    </div>
    <hr style="border-color:rgba(59,130,246,0.2); margin:10px 0 20px;">
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ Paramètres")

    confidence_threshold = st.slider(
        "Seuil de confiance YOLO",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
        help="Plus haut = moins de faux positifs"
    )

    show_distance = st.toggle("📏 Afficher distance estimée", value=True)
    show_voice    = st.toggle("🔊 Préview message vocal", value=True)
    show_json     = st.toggle("📤 Afficher JSON → Fatiha", value=True)

    st.markdown("<hr style='border-color:rgba(59,130,246,0.2);'>", unsafe_allow_html=True)
    st.markdown("#### 📋 À propos")
    st.markdown("""
    <div style="font-size:0.8rem; color:#64748b; line-height:1.8;">
        <b style="color:#93c5fd;">Auteure :</b> Oumaima Lahkiar<br>
        <b style="color:#93c5fd;">Module :</b> Vision Pipeline<br>
        <b style="color:#93c5fd;">Modèles :</b> YOLOv8n + CRAFT<br>
        <b style="color:#93c5fd;">Projet :</b> PFA Applied AI 2026
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(59,130,246,0.2);'>", unsafe_allow_html=True)

    # Session stats
    if 'total_analyses' not in st.session_state:
        st.session_state.total_analyses   = 0
        st.session_state.total_objects    = 0
        st.session_state.total_text_zones = 0

    st.markdown("#### 📊 Session Stats")
    st.metric("Analyses", st.session_state.total_analyses)
    st.metric("Objets détectés", st.session_state.total_objects)
    st.metric("Zones texte", st.session_state.total_text_zones)

# ═══════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="aw-header">
    <div style="display:flex; align-items:center; gap:16px;">
        <span style="font-size:2rem;">👁</span>
        <div>
            <div style="font-size:1.6rem; font-weight:700; color:#e2e8f0; letter-spacing:-0.02em;">
                AssistWalk <span style="color:#3b82f6;">Vision</span>
            </div>
            <div style="font-size:0.85rem; color:#64748b;">
                Module de détection visuelle · YOLO + CRAFT · Oumaima Lahkiar
            </div>
        </div>
        <div style="margin-left:auto; text-align:right;">
            <div style="font-size:0.75rem; color:#64748b;">Seuil actif</div>
            <div style="font-size:1.2rem; font-weight:700; color:#06b6d4;">""" + f"{int(confidence_threshold*100)}%" + """</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# MODE SELECTOR
# ═══════════════════════════════════════════════════════════
mode = st.radio(
    '**Mode d\'analyse :**',
    ['🖼️  Image statique', '🎬  Vidéo'],
    horizontal=True
)
st.divider()

# ════════════════════════════════════════════════════════════
# ██  MODE IMAGE
# ════════════════════════════════════════════════════════════
if '🖼️' in mode:

    uploaded = st.file_uploader(
        'Importer une image (JPG, PNG)',
        type=['jpg', 'jpeg', 'png'],
        label_visibility='collapsed'
    )

    if uploaded:
        pil_img  = Image.open(uploaded).convert('RGB')
        image_np = acquire_from_pil(pil_img)
        img_w, img_h = pil_img.size

        # ── Analyse ───────────────────────────────────────
        start_time = time.time()
        with st.spinner(''):
            st.markdown("""
            <div style="text-align:center; padding:20px; color:#3b82f6; font-size:0.9rem; letter-spacing:0.1em;">
                ⟳ &nbsp; ANALYSE EN COURS · YOLO + CRAFT · PLEASE WAIT
            </div>""", unsafe_allow_html=True)
            results = vision.analyze(image_np)
        elapsed = time.time() - start_time

        # Mettre à jour les stats session
        st.session_state.total_analyses   += 1
        st.session_state.total_objects    += len(results['objects'])
        st.session_state.total_text_zones += len(results['text_regions'])

        # ── Métriques dashboard ───────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Objets détectés",  len(results['objects']))
        c2.metric("📝 Zones texte",       len(results['text_regions']))
        c3.metric("⚡ Temps d'analyse",   f"{elapsed:.2f}s")
        high_count = len([o for o in results['objects'] if get_priority(o['class'])[0]=='HAUTE'])
        c4.metric("🔴 Priorité haute",    high_count)

        st.divider()

        # ── Images côte à côte ────────────────────────────
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown("**Image originale**")
            st.image(pil_img, use_column_width=True)

        with col2:
            annotated = draw_annotations(pil_img, results['objects'], results['text_boxes'])
            st.markdown("**Résultats annotés**")
            st.image(annotated, use_column_width=True)
            st.markdown(
                '<span class="aw-badge-danger">🔴 Haute priorité</span> &nbsp;'
                '<span class="aw-badge-warning">🟠 Moyenne</span> &nbsp;'
                '<span class="aw-badge-ok">🔵 Texte CRAFT</span>',
                unsafe_allow_html=True
            )

        st.divider()

        # ── Tableau des objets ────────────────────────────
        col3, col4 = st.columns([3, 2], gap="medium")

        with col3:
            st.markdown(f"#### 📦 Objets détectés ({len(results['objects'])})")

            if results['objects']:
                for obj in results['objects']:
                    prio, badge_type, color = get_priority(obj['class'])
                    dist, icon = estimate_distance(obj['bbox'], img_w, img_h)
                    x1,y1,x2,y2 = obj['bbox']

                    badge_class = f"aw-badge-{badge_type}"
                    dist_info = f"<br><span class='mono' style='color:#64748b;'>Position : ({x1},{y1}) → ({x2},{y2}) &nbsp;|&nbsp; {icon} Distance estimée : {dist}</span>" if show_distance else ""

                    st.markdown(f"""
                    <div class="aw-card">
                        <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
                            <span style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">{obj['class'].upper()}</span>
                            <span class="{badge_class}">{prio}</span>
                            <span style="margin-left:auto; color:#93c5fd; font-weight:600;">{int(obj['confidence']*100)}%</span>
                        </div>
                        <div style="background:rgba(59,130,246,0.1); border-radius:4px; height:6px; margin:6px 0;">
                            <div style="background:linear-gradient(90deg,{color},{color}88); width:{int(obj['confidence']*100)}%; height:100%; border-radius:4px;"></div>
                        </div>
                        {dist_info}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="aw-card" style="text-align:center; color:#64748b;">
                    Aucun objet pertinent détecté dans cette image.
                </div>""", unsafe_allow_html=True)

        with col4:
            st.markdown(f"#### 📝 Zones texte CRAFT ({len(results['text_regions'])})")
            if results['text_regions']:
                for i, crop in enumerate(results['text_regions']):
                    st.image(crop, caption=f'Zone {i+1}', use_column_width=True)
            else:
                st.markdown("""
                <div class="aw-card" style="text-align:center; color:#64748b;">
                    Aucune zone de texte détectée.
                </div>""", unsafe_allow_html=True)

        # ── Message vocal ─────────────────────────────────
        if show_voice:
            st.divider()
            st.markdown("#### 🔊 Message vocal généré")
            voice_msg = generate_voice_message(results['objects'], len(results['text_boxes']))
            st.markdown(f'<div class="aw-voice">💬 &nbsp; "{voice_msg}"</div>', unsafe_allow_html=True)
            st.caption("Ce message sera lu à voix haute via TextToSpeech dans l'app Android.")

        # ── Export ────────────────────────────────────────
        st.divider()
        st.markdown("#### 💾 Export des résultats")
        exp1, exp2, exp3 = st.columns(3)

        # Export JSON
        json_data = json.dumps({
            'timestamp':  datetime.now().isoformat(),
            'image':      uploaded.name,
            'objects':    results['objects'],
            'text_boxes': results['text_boxes'],
            'nb_regions': len(results['text_regions']),
            'voice_message': generate_voice_message(results['objects'], len(results['text_boxes']))
        }, indent=2)
        exp1.download_button(
            label="📥 JSON",
            data=json_data,
            file_name=f"assistwalk_{datetime.now().strftime('%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

        # Export CSV
        csv_data = results_to_csv([{'frame_num': 0, 'objects': results['objects']}])
        exp2.download_button(
            label="📊 CSV",
            data=csv_data,
            file_name=f"assistwalk_{datetime.now().strftime('%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Export image annotée
        buf = BytesIO()
        annotated.save(buf, format='PNG')
        exp3.download_button(
            label="🖼️ Image annotée",
            data=buf.getvalue(),
            file_name=f"assistwalk_annotated_{datetime.now().strftime('%H%M%S')}.png",
            mime="image/png",
            use_container_width=True
        )

        # ── JSON pour Fatiha ──────────────────────────────
        if show_json:
            st.divider()
            with st.expander("📤 JSON transmis au module de Fatiha (Analyse & Interaction)"):
                st.json({
                    'objects':       results['objects'],
                    'text_boxes':    results['text_boxes'],
                    'nb_regions':    len(results['text_regions']),
                    'voice_preview': generate_voice_message(results['objects'], len(results['text_boxes']))
                })

# ════════════════════════════════════════════════════════════
# ██  MODE VIDÉO
# ════════════════════════════════════════════════════════════
elif '🎬' in mode:

    uploaded_video = st.file_uploader(
        'Importer une vidéo',
        type=['mp4', 'avi', 'mov'],
        label_visibility='collapsed'
    )

    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name

        cap          = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        duration     = total_frames / fps if fps > 0 else 0

        # Info vidéo
        i1, i2, i3 = st.columns(3)
        i1.metric("🎞️ Total frames", total_frames)
        i2.metric("⚡ FPS", f"{fps:.0f}")
        i3.metric("⏱️ Durée", f"{duration:.1f}s")

        st.divider()

        interval = st.slider(
            '**Analyser 1 frame toutes les N frames**',
            min_value=10, max_value=60, value=30, step=5,
            help="30 = analyse ~1 frame/seconde à 30fps"
        )

        if st.button('▶️  Lancer l\'analyse vidéo', use_container_width=False):
            progress_bar = st.progress(0)
            status_col1, status_col2 = st.columns(2)
            status_text  = status_col1.empty()
            obj_counter  = status_col2.empty()

            all_results  = []
            frame_count  = 0
            start_time   = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % interval == 0:
                    status_text.markdown(f"🔍 `Frame {frame_count}/{total_frames}`")
                    rgb    = acquire_from_video_frame(frame)
                    result = vision.analyze(rgb)

                    # Annoter la frame
                    pil_frame = Image.fromarray(rgb)
                    annotated_frame = draw_annotations(pil_frame, result['objects'], result['text_boxes'])

                    # Générer message vocal
                    voice = generate_voice_message(result['objects'], len(result['text_boxes']))

                    all_results.append({
                        'frame_num':  frame_count,
                        'frame_img':  annotated_frame,
                        'objects':    result['objects'],
                        'text_boxes': result['text_boxes'],
                        'nb_text':    len(result['text_regions']),
                        'voice':      voice
                    })

                    total_obj = sum(len(r['objects']) for r in all_results)
                    obj_counter.markdown(f"🎯 **{total_obj}** objets cumulés")
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

                frame_count += 1

            cap.release()
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            elapsed = time.time() - start_time
            progress_bar.progress(1.0)
            status_text.empty()
            obj_counter.empty()

            # Mettre à jour session stats
            st.session_state.total_analyses   += len(all_results)
            total_obj_all = sum(len(r['objects']) for r in all_results)
            total_txt_all = sum(r['nb_text'] for r in all_results)
            st.session_state.total_objects    += total_obj_all
            st.session_state.total_text_zones += total_txt_all

            st.success(f"✅ Analyse terminée en {elapsed:.1f}s — {len(all_results)} frames analysées")

            # ── Dashboard résumé vidéo ─────────────────────
            st.divider()
            st.markdown("#### 📊 Dashboard — Résumé vidéo")

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Frames analysées", len(all_results))
            d2.metric("Total objets", total_obj_all)
            d3.metric("Total zones texte", total_txt_all)
            frames_with_danger = len([r for r in all_results if any(get_priority(o['class'])[0]=='HAUTE' for o in r['objects'])])
            d4.metric("🔴 Frames dangereuses", frames_with_danger)

            st.divider()

            # ── Résultats par frame ────────────────────────
            st.markdown(f"#### 🎞️ Résultats par frame ({len(all_results)} frames)")

            for r in all_results:
                nb_high = len([o for o in r['objects'] if get_priority(o['class'])[0]=='HAUTE'])
                danger_icon = "🔴" if nb_high > 0 else ("🟡" if r['objects'] else "🟢")

                with st.expander(
                    f"{danger_icon}  Frame {r['frame_num']}  —  "
                    f"{len(r['objects'])} objet(s)  —  {r['nb_text']} zone(s) texte"
                ):
                    col_a, col_b = st.columns([2, 1], gap="medium")

                    with col_a:
                        st.image(r['frame_img'], use_column_width=True)

                    with col_b:
                        if r['objects']:
                            for obj in r['objects']:
                                prio, badge_type, color = get_priority(obj['class'])
                                dist, icon = estimate_distance(obj['bbox'], 640, 480)
                                st.markdown(f"""
                                <div class="aw-card" style="padding:10px 14px; margin-bottom:8px;">
                                    <b style="color:#e2e8f0;">{obj['class'].upper()}</b>
                                    <span class="aw-badge-{badge_type}" style="margin-left:8px;">{prio}</span><br>
                                    <span class="mono" style="color:#93c5fd;">{int(obj['confidence']*100)}% confiance</span>
                                    {"<br><span class='mono' style='color:#64748b;'>" + icon + " " + dist + "</span>" if show_distance else ""}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown('<div style="color:#64748b; padding:10px;">Aucun objet détecté</div>', unsafe_allow_html=True)

                        st.caption(f"Zones texte : {r['nb_text']}")

                        if show_voice:
                            st.markdown(f'<div class="aw-voice" style="font-size:0.8rem; padding:10px;">💬 "{r["voice"]}"</div>', unsafe_allow_html=True)

                        if show_json:
                            st.markdown("**JSON → Fatiha**")
                            st.json({
                                'frame':   r['frame_num'],
                                'objects': r['objects'],
                                'text_boxes': r['text_boxes'],
                                'nb_regions': r['nb_text'],
                                'voice': r['voice']
                            })

            # ── Export global ──────────────────────────────
            st.divider()
            st.markdown("#### 💾 Export global")
            ex1, ex2 = st.columns(2)

            full_json = json.dumps({
                'timestamp':    datetime.now().isoformat(),
                'video':        uploaded_video.name,
                'total_frames': len(all_results),
                'results':      [{
                    'frame':      r['frame_num'],
                    'objects':    r['objects'],
                    'text_boxes': r['text_boxes'],
                    'voice':      r['voice']
                } for r in all_results]
            }, indent=2)

            ex1.download_button(
                label="📥 Export JSON complet",
                data=full_json,
                file_name=f"assistwalk_video_{datetime.now().strftime('%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

            csv_data = results_to_csv(all_results)
            ex2.download_button(
                label="📊 Export CSV complet",
                data=csv_data,
                file_name=f"assistwalk_video_{datetime.now().strftime('%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )