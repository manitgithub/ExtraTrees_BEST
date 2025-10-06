"""
ตัวอย่างการเรียกใช้ ExtraTrees Model REST API
รันไฟล์นี้หลังจาก start API server (python api.py) แล้ว
"""

import requests
import json

# Base URL ของ API
BASE_URL = "http://localhost:5000"

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_response(response):
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    else:
        print(f"Error {response.status_code}: {response.text}")

# 1. ตรวจสอบสถานะ API
print_section("1. ตรวจสอบสถานะ API")
response = requests.get(f"{BASE_URL}/api/health")
print_response(response)

# 2. ดึงข้อมูลโมเดล
print_section("2. ข้อมูลโมเดล")
response = requests.get(f"{BASE_URL}/api/model/info")
print_response(response)

# 3. ดึง Parameters ของ ExtraTrees
print_section("3. Parameters ของ ExtraTrees")
response = requests.get(f"{BASE_URL}/api/model/params/extratrees")
print_response(response)

# 4. ดึง Feature Importances
print_section("4. Feature Importances")
response = requests.get(f"{BASE_URL}/api/model/feature-importances")
data = response.json()
if response.status_code == 200:
    print("\nRaw Array:")
    print(data['raw_array'])
    
    print("\nTop 5 Important Features:")
    for item in data['sorted_by_importance'][:5]:
        print(f"  Rank {item['rank']}: Feature {item['feature_index']} "
              f"= {item['importance']:.6f} ({item['percentage']:.2f}%)")

# 5. ทำนายผล (1 ตัวอย่าง)
print_section("5. ทำนายผล (1 ตัวอย่าง)")
data = {
    "features": [1.2, 3.4, 0.7, 2.1, 5.0, 1.8, 2.3, 4.5, 3.2]
}
print("Input:")
print(json.dumps(data, indent=2))
response = requests.post(f"{BASE_URL}/api/predict", json=data)
print("\nOutput:")
print_response(response)

# 6. ทำนายผล (หลายตัวอย่าง)
print_section("6. ทำนายผล (หลายตัวอย่าง)")
data = {
    "features": [
        [1.2, 3.4, 0.7, 2.1, 5.0, 1.8, 2.3, 4.5, 3.2],
        [2.1, 4.5, 1.3, 3.2, 6.1, 2.4, 3.1, 5.2, 4.1],
        [0.5, 2.1, 0.3, 1.5, 3.2, 1.1, 1.8, 3.5, 2.4]
    ]
}
print("Input: 3 ตัวอย่าง")
response = requests.post(f"{BASE_URL}/api/predict", json=data)
print("\nOutput:")
print_response(response)

# 7. ทดสอบ Error Handling (ส่ง features ไม่ครบ)
print_section("7. ทดสอบ Error Handling (features ไม่ครบ)")
data = {
    "features": [1.2, 3.4, 0.7]  # ส่งแค่ 3 features (ต้องการ 9)
}
print("Input: features ไม่ครบ (ส่ง 3 แต่ต้องการ 9)")
response = requests.post(f"{BASE_URL}/api/predict", json=data)
print("\nOutput:")
if response.status_code != 200:
    print(f"Error {response.status_code}:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

print("\n" + "=" * 70)
print("✓ ทดสอบเสร็จสิ้น!")
print("=" * 70)
