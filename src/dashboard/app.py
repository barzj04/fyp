from flask import Flask, render_template, jsonify, request
import joblib
import numpy as np
import pandas as pd
import shap
import json
import os
import paho.mqtt.client as mqtt_client
import threading
import json
from collections import deque
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq
import numpy as np
import time
import datetime
import smtplib
from email.mime.text import MIMEText
import anthropic

app = Flask(__name__)
FS = 12000  # must match features.py (used when the models were trained), not the ESP32's real rate

def extract_features_live(window):
    w  = np.array(window)
    rms      = np.sqrt(np.mean(w**2))
    peak     = np.max(np.abs(w))
    kurt     = kurtosis(w)
    skewness = skew(w)
    std      = np.std(w)
    crest    = peak / rms if rms > 0 else 0
    fft_vals = np.abs(rfft(w))
    freqs    = rfftfreq(len(w), d=1/FS)
    dom_freq  = freqs[np.argmax(fft_vals)]
    spec_cent = np.sum(freqs*fft_vals)/np.sum(fft_vals) if np.sum(fft_vals)>0 else 0
    spec_enrg = np.sum(fft_vals**2)
    band_low  = np.sum(fft_vals[(freqs>=0)    & (freqs<1000)]**2)   # same bands as features.py
    band_mid  = np.sum(fft_vals[(freqs>=1000) & (freqs<5000)]**2)
    return np.array([[rms, peak, kurt, skewness, std, crest,
                      dom_freq, spec_cent, spec_enrg, band_low, band_mid]])

# ── Load models ────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), '..', '..')
rf     = joblib.load(os.path.join(BASE, 'models/rf_model.pkl'))
iso    = joblib.load(os.path.join(BASE, 'models/iso_forest.pkl'))
scaler = joblib.load(os.path.join(BASE, 'models/scaler.pkl'))

FEAT_COLS = ["rms","peak","kurtosis","skewness","std","crest_factor",
             "dom_freq","spec_centroid","spec_energy","band_low","band_mid"]
LABELS    = {0: "Normal", 1: "Degraded", 2: "Fault"}
ACTIONS   = {
    "Normal":   "No action required. Equipment operating normally.",
    "Degraded": "Schedule maintenance check within the week.",
    "Fault":    "Stop machine immediately. Contact maintenance engineer."
}
SEVERITY  = {
    "Normal": "LOW", "Degraded": "MEDIUM", "Fault": "HIGH"
}

# ── Human-language feature descriptions ───────────────
FEATURE_PLAIN = {
    "rms":           ("overall vibration intensity is higher than normal",
                      "overall vibration intensity is within safe range"),
    "peak":          ("sudden large vibration spikes are occurring",
                      "vibration peaks are within safe range"),
    "kurtosis":      ("sharp impact-like vibration spikes are present",
                      "vibration is smooth with no impact spikes"),
    "skewness":      ("the vibration signal is asymmetric, suggesting uneven mechanical load",
                      "mechanical load is balanced"),
    "std":           ("vibration amplitude is highly variable and unstable",
                      "vibration amplitude is stable"),
    "crest_factor":  ("the ratio of peak to average vibration is high, indicating impact risk",
                      "peak-to-average vibration ratio is normal"),
    "dom_freq":      ("the dominant vibration frequency is abnormally high",
                      "dominant vibration frequency is normal"),
    "spec_centroid": ("the machine is vibrating at higher frequencies than expected",
                      "vibration frequency content is normal"),
    "spec_energy":   ("total vibration energy output is elevated",
                      "total vibration energy is within limits"),
    "band_low":      ("low-frequency vibration is elevated, which may indicate imbalance or misalignment",
                      "low-frequency vibration is normal"),
    "band_mid":      ("mid-frequency vibration is elevated, which may indicate early bearing wear",
                      "mid-frequency vibration is normal"),
}

def generate_human_explanation(label, shap_data, rf_proba):
    """Translate top SHAP features into plain English for the dashboard."""
    if label == "Normal":
        return "The equipment is operating normally. No unusual vibration patterns were detected."
    if not shap_data:
        return "Abnormal vibration patterns were detected but could not be further explained."

    parts = []
    for s in shap_data[:3]:
        pair = FEATURE_PLAIN.get(s["feature"])
        if pair:
            parts.append(pair[0] if s["value"] > 0 else pair[1])

    if not parts:
        return "Abnormal vibration patterns were detected."

    if len(parts) == 1:
        feature_sentence = f"The main warning sign is that the {parts[0]}."
    elif len(parts) == 2:
        feature_sentence = f"The main warning signs are that the {parts[0]}, and the {parts[1]}."
    else:
        feature_sentence = f"The main warning signs are that the {parts[0]}, the {parts[1]}, and the {parts[2]}."

    consequence = (
        "This may indicate early-stage bearing wear or mechanical imbalance. Schedule a maintenance check soon."
        if label == "Degraded"
        else "This is consistent with active bearing damage or a shaft fault. Stop the machine and inspect it immediately."
    )
    return f"{feature_sentence} {consequence}"

explainer = shap.TreeExplainer(rf)

# ── Live sensor buffer ─────────────────────────────────
live_buffer = deque(maxlen=256)  # stores last 256 readings
latest_reading = {"ax": 0, "ay": 0, "az": 0}

def on_mqtt_message(client, userdata, msg):
    global latest_reading
    try:
        data = json.loads(msg.payload)
        live_buffer.append(data.get("ax", 0))
        latest_reading = data
    except:
        pass

def on_mqtt_connect(client, userdata, flags, reason_code, properties):
    print(f"MQTT connected ({reason_code}) - listening for live ESP32 data")
    client.subscribe("fyp/vibration")  # (re)subscribe on every reconnect

def start_mqtt():
    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2)
    client.on_connect = on_mqtt_connect
    client.on_message = on_mqtt_message

    # Broker may not be running yet: keep retrying quietly, use simulation meanwhile
    warned = False
    while True:
        try:
            client.connect("localhost", 1883)
            break
        except OSError:
            if not warned:
                print("MQTT broker not reachable on localhost:1883 - using simulation data, retrying every 10s")
                warned = True
            time.sleep(10)
    client.loop_forever()  # handles reconnects after the first connection

# Start MQTT in background thread
threading.Thread(target=start_mqtt, daemon=True).start()

# ── Load test data for simulation ──────────────────────
df_test = pd.read_csv(os.path.join(BASE, 'data/lab_features.csv'))
sim_idx = 230  # pointer for simulation (230 = start of mostly-Degraded stretch; use 0 for Normal/Fault)

alert_log = []

# ── Email alerts ───────────────────────────────────────
# Set ALERT_SENDER, ALERT_RECEIVER, ALERT_PASSWORD (Gmail App Password) as env vars
ALERT_SENDER   = os.environ.get("ALERT_SENDER")
ALERT_RECEIVER = os.environ.get("ALERT_RECEIVER")
ALERT_PASSWORD = os.environ.get("ALERT_PASSWORD")
EMAIL_COOLDOWN_S = 300  # min seconds between emails for the same label
if ALERT_SENDER and ALERT_RECEIVER and ALERT_PASSWORD:
    print(f"Email alerts ON: {ALERT_SENDER} -> {ALERT_RECEIVER}")
else:
    print("Email alerts OFF: set ALERT_SENDER, ALERT_RECEIVER, ALERT_PASSWORD to enable")
last_email_sent = {}    
def send_alert_email(label, confidence, root_cause):
    msg = MIMEText(f"""
EQUIPMENT ALERT — {label} DETECTED

Confidence: {confidence}%
Root Cause: {root_cause}
Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check the dashboard for full diagnostic details.
    """)
    msg['Subject'] = f"⚠️ PdM Alert: {label} Detected"
    msg['From']    = ALERT_SENDER
    msg['To']      = ALERT_RECEIVER

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(ALERT_SENDER, ALERT_PASSWORD)
            server.sendmail(ALERT_SENDER, ALERT_RECEIVER, msg.as_string())
        print(f"Alert email sent: {label}")
    except Exception as e:
        print(f"Alert email failed: {e}")

def maybe_send_alert(label, confidence, root_cause):
    if label == "Normal" or not (ALERT_SENDER and ALERT_RECEIVER and ALERT_PASSWORD):
        return
    now = time.time()
    if now - last_email_sent.get(label, 0) < EMAIL_COOLDOWN_S:
        return
    last_email_sent[label] = now
    threading.Thread(target=send_alert_email,
                     args=(label, confidence, root_cause), daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

def calc_degradation(rf_proba):
    """Expected severity of the RF prediction (0-100%).
    Severity weights: Normal = 0, Degraded = 0.5, Fault = 1.0.
    Reflects classification risk, not physical wear or remaining useful life."""
    normal_prob, degraded_prob, fault_prob = rf_proba
    
    if normal_prob >= 0.8:
        degradation = (1 - normal_prob) * 20  # max 20% when Normal
    else:
        degradation = (degraded_prob * 0.5 + fault_prob * 1.0) * 100
    
    return round(degradation, 1)
@app.route('/predict')
def predict():
    start = time.time()
    global sim_idx, alert_log

    # Use live ESP32 data if available, else simulate
    if len(live_buffer) >= 256:
        window = list(live_buffer)[-256:]
        X_raw  = extract_features_live(window)
        print(f"Live features: {dict(zip(FEAT_COLS, X_raw[0].round(4)))}")
        source = "live"
        
        # Scale first
        X_sc = scaler.transform(X_raw)
        
        # IF score
        iso_score  = round(float(iso.decision_function(X_sc)[0]), 4)
        iso_pred   = iso.predict(X_sc)[0]
        
        # Layer 1: Isolation Forest decides if the reading is abnormal.
        # Layer 2: Random Forest only runs when Layer 1 flags it.
        rf_full   = rf.predict_proba(X_sc)[0].tolist()
        is_anomaly = bool(iso_pred == -1)

        if is_anomaly:
            rf_proba = rf_full
            rf_pred  = int(np.argmax(rf_full))
            label    = LABELS[rf_pred]
            print(f"RF proba: Normal={round(rf_proba[0]*100,1)}% Degraded={round(rf_proba[1]*100,1)}% Fault={round(rf_proba[2]*100,1)}%")

            shap_vals = explainer(X_sc)
            contribs  = shap_vals.values[0, :, rf_pred].tolist()
            shap_data = [{"feature": f, "value": round(v, 4)}
                         for f, v in zip(FEAT_COLS, contribs)]
            shap_data.sort(key=lambda x: abs(x["value"]), reverse=True)
            top        = shap_data[0]
            direction  = "elevated" if top["value"] > 0 else "suppressed"
            root_cause = f"High {top['feature']} ({direction}) is the primary indicator."
            human_explanation = generate_human_explanation(label, shap_data, rf_proba)
        else:
            rf_pred    = 0
            rf_proba   = [1.0, 0.0, 0.0]
            label      = "Normal"
            shap_data  = []
            root_cause = "No anomaly detected."
            human_explanation = generate_human_explanation("Normal", [], [1.0, 0.0, 0.0])

        # Degradation uses the full RF output on every reading
        degradation = calc_degradation(rf_full)

    else:
        filter_label = None  # demo mode only: set 0=Normal, 1=Degraded, 2=Fault to pin a class
        if filter_label is not None:
            filtered = df_test[df_test['label'] == filter_label]
            row = filtered.iloc[sim_idx % len(filtered)]
        else:
            row = df_test.iloc[sim_idx % len(df_test)]
        sim_idx += 1
        X_raw  = np.array([row[FEAT_COLS].values], dtype=float)
        X_sc   = scaler.transform(X_raw)
        source = "simulation"

        # IF
        iso_score  = round(float(iso.decision_function(X_sc)[0]), 4)
        iso_pred   = iso.predict(X_sc)[0]
        force_label = int(row['label'])
        is_anomaly  = bool(iso_pred == -1) if force_label != 0 else False

        # RF
        if is_anomaly:
            rf_pred   = int(rf.predict(X_sc)[0])
            rf_proba  = rf.predict_proba(X_sc)[0].tolist()
            label     = LABELS[rf_pred]
            shap_vals = explainer(X_sc)
            contribs  = shap_vals.values[0, :, rf_pred].tolist()
            shap_data = [{"feature": f, "value": round(v, 4)}
                         for f, v in zip(FEAT_COLS, contribs)]
            shap_data.sort(key=lambda x: abs(x["value"]), reverse=True)
            top        = shap_data[0]
            direction  = "elevated" if top["value"] > 0 else "suppressed"
            root_cause = f"High {top['feature']} ({direction}) is the primary indicator."
            human_explanation = generate_human_explanation(label, shap_data, rf_proba)
        else:
            rf_pred    = 0
            rf_proba   = [1.0, 0.0, 0.0]
            label      = "Normal"
            shap_data  = []
            root_cause = "No anomaly detected."
            human_explanation = generate_human_explanation("Normal", [], [1.0, 0.0, 0.0])

        # Degradation from RF on every reading (not only when IF flags), so it changes continuously
        degradation = calc_degradation(rf.predict_proba(X_sc)[0])

    # Log alert
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    alert_log.insert(0, {
        "time":      timestamp,
        "label":     label,
        "severity":  SEVERITY[label],
        "action":    ACTIONS[label],
        "iso_score": iso_score,
        "confidence": round(rf_proba[rf_pred] * 100, 1)
    })
    alert_log = alert_log[:20]  # keep last 20
    if label == "Fault" and is_anomaly:
        maybe_send_alert(label, round(rf_proba[rf_pred] * 100, 1), root_cause)
    latency_ms = round((time.time() - start) * 1000, 2) 
    print(f"Latency: {latency_ms}ms | Source: {source} | Label: {label}")

    return jsonify({
        "label":      label,
        "severity":   SEVERITY[label],
        "action":     ACTIONS[label],
        "iso_score":  iso_score,
        "is_anomaly": is_anomaly,
        "confidence": round(rf_proba[rf_pred] * 100, 1),
        "rf_proba":   {"Normal": round(rf_proba[0]*100,1),
                       "Degraded": round(rf_proba[1]*100,1),
                       "Fault": round(rf_proba[2]*100,1)},
        "shap":       shap_data[:6],
        "root_cause": root_cause,
        "human_explanation": human_explanation,
        "degradation": degradation,
        "raw_value":  round(float(X_raw[0][0]), 4),
        "alert_log":  alert_log,
        "latency_ms": latency_ms,
        "source": source
    })

# ── AI assistant chat ──────────────────────────────────
# Reads ANTHROPIC_API_KEY from the environment
anthropic_client = anthropic.Anthropic()

@app.route('/chat', methods=['POST'])
def chat():
    data       = request.get_json(silent=True) or {}
    user_msg   = (data.get('message') or '').strip()
    label      = data.get('label', 'Unknown')
    confidence = data.get('confidence', 0)
    root_cause = data.get('root_cause', '')
    shap_data  = data.get('shap', [])

    if not user_msg:
        return jsonify({"error": "Message is empty."}), 400

    top_features = ", ".join(f"{s['feature']} ({s['value']:+.3f})"
                             for s in shap_data[:3]) or "none"

    system_prompt = f"""You are a predictive maintenance assistant for an oil and gas facility.
Current equipment status:
- Condition: {label}
- Confidence: {confidence}%
- Root cause: {root_cause}
- Top features: {shap_data[:3]}

Your role:
- Explain what the current reading means in simple terms
- Suggest specific next steps for operators and engineers
- Be concise and actionable
- Use plain language — operators are not ML experts
- Do NOT use markdown formatting, bullet points with **, or any special characters
- Write in plain sentences only
- Keep response under 100 words"""

    try:
        response = anthropic_client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=1024,                  # thinking tokens count toward this
            output_config={"effort": "low"},  # fast, short answers
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}]
        )
    except (anthropic.AuthenticationError, TypeError):  # TypeError = no key set
        return jsonify({"error": "Invalid or missing ANTHROPIC_API_KEY."}), 500
    except anthropic.RateLimitError:
        return jsonify({"error": "Rate limited. Try again shortly."}), 429
    except anthropic.APIStatusError as e:
        return jsonify({"error": f"API error: {e.message}"}), 502
    except anthropic.APIConnectionError:
        return jsonify({"error": "Could not reach the AI service."}), 503

    reply = "".join(b.text for b in response.content if b.type == "text")
    return jsonify({"response": reply or "No response generated."})

if __name__ == '__main__':
    app.run(debug=True, port=5000)