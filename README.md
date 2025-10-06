# ExtraTrees Model REST API - คู่มือการใช้งาน

## วิธีเริ่มต้น API Server

```bash
python api.py
```

API จะรันที่: `http://localhost:5000`

---

## API Endpoints

### 1. หน้าแรก
**GET** `/`

แสดงรายการ endpoints ทั้งหมด

**ตัวอย่างการใช้งาน:**
```bash
curl http://localhost:5000/
```

---

### 2. ตรวจสอบสถานะ API
**GET** `/api/health`

ตรวจสอบว่า API ทำงานปกติหรือไม่

**ตัวอย่างการใช้งาน:**
```bash
curl http://localhost:5000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "params_loaded": true
}
```

---

### 3. ดึงข้อมูลโมเดล
**GET** `/api/model/info`

ดึงข้อมูลพื้นฐานของโมเดล

**ตัวอย่างการใช้งาน:**
```bash
curl http://localhost:5000/api/model/info
```

**Response:**
```json
{
  "model_type": "ExtraTreesRegressor",
  "n_estimators": 272,
  "n_features": 9,
  "n_outputs": 1,
  "max_depth": 16,
  "min_samples_split": 2,
  "min_samples_leaf": 2
}
```

---

### 4. ดึง Parameters ทั้งหมด
**GET** `/api/model/params`

ดึง best parameters ของทุกโมเดลจากไฟล์ best_params.json

**ตัวอย่างการใช้งาน:**
```bash
curl http://localhost:5000/api/model/params
```

---

### 5. ดึง Parameters ของ ExtraTrees
**GET** `/api/model/params/extratrees`

ดึง best parameters ของ ExtraTrees เท่านั้น

**ตัวอย่างการใช้งาน:**
```bash
curl http://localhost:5000/api/model/params/extratrees
```

**Response:**
```json
{
  "n_estimators": 272,
  "max_depth": 16,
  "min_samples_split": 2,
  "min_samples_leaf": 2,
  "max_features": "sqrt"
}
```

---

### 6. ดึง Feature Importances
**GET** `/api/model/feature-importances`

ดึงค่าความสำคัญของ features ทั้งหมด

**ตัวอย่างการใช้งาน:**
```bash
curl http://localhost:5000/api/model/feature-importances
```

**Response:**
```json
{
  "raw_array": [0.024, 0.064, 0.156, 0.096, 0.076, 0.080, 0.075, 0.133, 0.295],
  "features": [
    {"feature_index": 0, "importance": 0.024, "percentage": 2.43},
    {"feature_index": 1, "importance": 0.064, "percentage": 6.38},
    ...
  ],
  "sorted_by_importance": [
    {"rank": 1, "feature_index": 8, "importance": 0.295, "percentage": 29.54},
    {"rank": 2, "feature_index": 2, "importance": 0.156, "percentage": 15.58},
    ...
  ],
  "total_sum": 1.0
}
```

---

### 7. ทำนายผล
**POST** `/api/predict`

ส่งข้อมูล features เพื่อทำนายผล

**Request Body:**
```json
{
  "features": [1.2, 3.4, 0.7, 2.1, 5.0, 1.8, 2.3, 4.5, 3.2]
}
```

หรือทำนายหลายรายการพร้อมกัน:
```json
{
  "features": [
    [1.2, 3.4, 0.7, 2.1, 5.0, 1.8, 2.3, 4.5, 3.2],
    [2.1, 4.5, 1.3, 3.2, 6.1, 2.4, 3.1, 5.2, 4.1]
  ]
}
```

**ตัวอย่างการใช้งาน (curl):**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\": [1.2, 3.4, 0.7, 2.1, 5.0, 1.8, 2.3, 4.5, 3.2]}"
```

**ตัวอย่างการใช้งาน (Python):**
```python
import requests

url = "http://localhost:5000/api/predict"
data = {
    "features": [1.2, 3.4, 0.7, 2.1, 5.0, 1.8, 2.3, 4.5, 3.2]
}

response = requests.post(url, json=data)
print(response.json())
```

**Response:**
```json
{
  "predictions": [12.345],
  "n_samples": 1,
  "features_used": 9
}
```

---

## ตัวอย่างการใช้งานด้วย Python

```python
import requests

# Base URL
BASE_URL = "http://localhost:5000"

# 1. ตรวจสอบสถานะ
response = requests.get(f"{BASE_URL}/api/health")
print("Health:", response.json())

# 2. ดึงข้อมูลโมเดล
response = requests.get(f"{BASE_URL}/api/model/info")
print("Model Info:", response.json())

# 3. ดึง Feature Importances
response = requests.get(f"{BASE_URL}/api/model/feature-importances")
print("Feature Importances:", response.json())

# 4. ทำนายผล
data = {
    "features": [1.2, 3.4, 0.7, 2.1, 5.0, 1.8, 2.3, 4.5, 3.2]
}
response = requests.post(f"{BASE_URL}/api/predict", json=data)
print("Prediction:", response.json())
```

---

## ตัวอย่างการใช้งานด้วย JavaScript (Fetch API)

```javascript
// Base URL
const BASE_URL = "http://localhost:5000";

// 1. ดึงข้อมูลโมเดล
fetch(`${BASE_URL}/api/model/info`)
  .then(response => response.json())
  .then(data => console.log("Model Info:", data));

// 2. ทำนายผล
fetch(`${BASE_URL}/api/predict`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    features: [1.2, 3.4, 0.7, 2.1, 5.0, 1.8, 2.3, 4.5, 3.2]
  })
})
  .then(response => response.json())
  .then(data => console.log("Prediction:", data));
```

---

## หมายเหตุ

- API รันบน port 5000 (เปลี่ยนได้ในไฟล์ api.py)
- โมเดลต้องการ input 9 features
- Response ทั้งหมดเป็นรูปแบบ JSON
- สำหรับ Production ควรปิด debug mode และใช้ WSGI server เช่น Gunicorn
