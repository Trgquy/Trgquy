dfrom imutils.video import VideoStream
import imutils
import cv2
import RPi.GPIO as GPIO
import time
from datetime import datetime
import requests

# ========================
# CONFIG
# ========================
GPIO_PIN = 21

FIREBASE_URL = "https://test1-ef023-default-rtdb.asia-southeast1.firebasedatabase.app".rstrip("/")
FACE_LOGS_PATH = "/face_logs"
DOOR_PATH = "/door"

# ✅ NODE ACCOUNT (auto create)
ACCOUNT_PATH = "/account"
DEFAULT_ACCOUNT = {
    "username": "admin",
    "password": "123456"
}

# Nếu DB bạn bật auth token (Database secret) thì điền vào, còn không để ""
FIREBASE_AUTH = ""

# LBPH: confidence_raw càng nhỏ càng match tốt
CONF_THRESHOLD = 80

# chống spam log
LOG_COOLDOWN_SEC = 1.0

# ✅ heartbeat cập nhật /door dù trạng thái không đổi (để app thấy updatedAt nhảy)
DOOR_HEARTBEAT_SEC = 2.0

# ========================
# GLOBAL STATE
# ========================
_last_log_time = 0.0
_last_door_state = None
_last_door_push = 0.0


# ========================
# FIREBASE HELPERS
# ========================
def _firebase_params():
    params = {}
    if FIREBASE_AUTH:
        params["auth"] = FIREBASE_AUTH
    return params


def ensure_account_node():
    """Tạo node /account nếu chưa có username/password"""
    try:
        url = f"{FIREBASE_URL}{ACCOUNT_PATH}.json"
        r = requests.get(url, params=_firebase_params(), timeout=5)
        print("[GET account]", r.status_code)

        if r.status_code >= 300:
            print("❌ GET account failed:", r.text)
            return

        current = r.json()
        if not isinstance(current, dict) or "username" not in current or "password" not in current:
            w = requests.put(url, json=DEFAULT_ACCOUNT, params=_firebase_params(), timeout=5)
            print("[PUT account]", w.status_code, w.text)
            if w.status_code < 300:
                print("✅ Created /account:", DEFAULT_ACCOUNT["username"])
        else:
            print("✅ /account already exists:", current.get("username"))

    except Exception as e:
        print("❌ ensure_account_node error:", e)


def send_face_log(name, confidence_percent, recognized):
    """POST log vào /face_logs (tạo key ngẫu nhiên)"""
    global _last_log_time
    now_ts = time.time()
    if now_ts - _last_log_time < LOG_COOLDOWN_SEC:
        return
    _last_log_time = now_ts

    now = datetime.now()
    data = {
        "name": name,
        "confidence": int(confidence_percent),
        "recognized": bool(recognized),
        "timestamp": now.isoformat(),
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "year": now.year
    }

    try:
        url = f"{FIREBASE_URL}{FACE_LOGS_PATH}.json"
        r = requests.post(url, json=data, params=_firebase_params(), timeout=5)

        if r.status_code >= 300:
            print("❌ POST face_logs failed:", r.status_code, r.text)
        else:
            print("✅ Sent face log:", data["name"], data["confidence"], data["recognized"])

    except Exception as e:
        print("❌ Firebase face log error:", e)


def update_door_status(is_open):
    """
    PATCH /door:
      - isOpen: true/false
      - updatedAt: ISO time
    Có heartbeat để updatedAt luôn nhảy dù trạng thái không đổi.
    """
    global _last_door_state, _last_door_push

    now_ts = time.time()
    state_changed = (_last_door_state != is_open)
    heartbeat_due = (now_ts - _last_door_push) >= DOOR_HEARTBEAT_SEC

    # nếu không đổi trạng thái và chưa tới heartbeat -> không gửi
    if (not state_changed) and (not heartbeat_due):
        return

    _last_door_state = is_open
    _last_door_push = now_ts

    data = {
        "isOpen": bool(is_open),
        "updatedAt": datetime.now().isoformat()
    }

    try:
        url = f"{FIREBASE_URL}{DOOR_PATH}.json"
        r = requests.patch(url, json=data, params=_firebase_params(), timeout=5)

        if r.status_code >= 300:
            print("❌ PATCH door failed:", r.status_code, r.text)
        else:
            print("✅ Door:", "OPEN" if is_open else "CLOSED", "| updatedAt pushed")

    except Exception as e:
        print("❌ Firebase door error:", e)


# ========================
# DOOR CONTROL (GPIO)
# ========================
def door_open():
    GPIO.output(GPIO_PIN, 0)  # relay active-low
    update_door_status(True)


def door_close():
    GPIO.output(GPIO_PIN, 1)
    update_door_status(False)


# ========================
# INIT GPIO AND CAMERA
# ========================
GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIO_PIN, GPIO.OUT)

# đóng cửa ngay khi start (đồng thời update lên Firebase)
door_close()

# tạo account node nếu thiếu
ensure_account_node()

# Camera
vs = VideoStream(src=0).start()
time.sleep(1.0)

# LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

# Haar cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

names = [
    "0","ID1","ID2","ID3","ID4","ID5","ID6","ID7","ID8","ID9",
    "ID10","ID11","ID12","ID13","ID14","ID15","ID16"
]

print("System started. Press q to quit.")

# ========================
# MAIN LOOP
# ========================
try:
    while True:
        frame = vs.read()

        # ✅ Dù có camera hay không, vẫn heartbeat cập nhật door (để node khỏi đứng)
        # Nếu không có frame thì vẫn coi là cửa đóng
        if frame is None:
            door_close()
            time.sleep(0.2)
            continue

        frame = imutils.resize(frame, width=480)
        image = cv2.flip(frame, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        face_found = False

        for (x, y, w, h) in faces:
            face_found = True

            roi_gray = gray[y:y+h, x:x+w]
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

            label, confidence_raw = recognizer.predict(roi_gray)

            conf_percent = int(round(100 - confidence_raw))
            conf_percent = max(0, min(100, conf_percent))

            if confidence_raw < CONF_THRESHOLD:
                name = names[label] if 0 <= label < len(names) else f"ID_{label}"
                recognized = True
                door_open()
            else:
                name = "KHONG THE NHAN DIEN"
                recognized = False
                door_close()

            cv2.putText(image, name, (x+5, y-5), font, 0.9, (255, 255, 255), 2)
            cv2.putText(image, f"{conf_percent}%", (x+5, y+h-5), font, 0.8, (255, 255, 0), 2)

            print("Face:", x, y, w, h, "|", name, conf_percent, recognized)

            # log lên firebase
            send_face_log(name, conf_percent, recognized)

        if not face_found:
            door_close()  # không thấy mặt -> cửa đóng

        # Hiển thị camera
        show_img = imutils.resize(image, width=700)
        cv2.imshow("Frame", show_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cv2.destroyAllWindows()
    vs.stop()
    GPIO.cleanup()
    print("System stopped.")
