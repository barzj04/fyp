import paho.mqtt.client as mqtt
import json, csv, os, datetime, threading

# ── Change this before each session ───────────────────
LABEL       = 2     # 0=Normal, 1=Degraded, 2=Fault
WEIGHT      = "15g"    
SESSION_NUM = 1

     # 1, 2, or 3
DURATION    = 60     # seconds (5 min = 300, 10 min = 600)

# ── Config ─────────────────────────────────────────────
BROKER  = "localhost"
TOPIC   = "fyp/vibration"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename  = f"data/lab/label{LABEL}_{WEIGHT}_session{SESSION_NUM}_{timestamp}.csv"
os.makedirs("data/lab", exist_ok=True)

csvfile = open(filename, 'w', newline='')
writer  = csv.writer(csvfile)
writer.writerow(['timestamp','ax','ay','az','gx','gy','gz','label'])
count = [0]
client = mqtt.Client()

def stop_recording():
    print(f"\n\n⏱ {DURATION}s complete — stopping...")
    csvfile.close()
    print(f"✅ Saved {count[0]:,} samples → {filename}")
    client.disconnect()

def on_connect(c, userdata, flags, rc):
    print(f"✅ Connected to MQTT broker")
    c.subscribe(TOPIC)
    print(f"Recording → {filename}")
    print(f"Label: {LABEL} | Weight: {WEIGHT} | Session: {SESSION_NUM}")
    print(f"Auto-stops in {DURATION} seconds\n")
    threading.Timer(DURATION, stop_recording).start()

def on_message(c, userdata, msg):
    if csvfile.closed:  # ← add this check
        return
    data = json.loads(msg.payload)
    writer.writerow([
        data.get('ts'),
        data.get('ax'), data.get('ay'), data.get('az'),
        data.get('gx'), data.get('gy'), data.get('gz'),
        LABEL
    ])
    count[0] += 1
    if count[0] % 1000 == 0:
        elapsed = count[0] / 100
        remaining = max(0, DURATION - elapsed)
        print(f"Samples: {count[0]:,} | ~{int(remaining)}s remaining    ", end='\r')

client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(BROKER, 1883)
    client.loop_forever()
except KeyboardInterrupt:
    csvfile.close()
    print(f"\nStopped manually. Saved {count[0]:,} samples → {filename}")