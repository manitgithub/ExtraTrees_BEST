"""
ทดสอบ API อย่างง่าย - ใช้ได้แม้ API ยังไม่เปิด
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

print("=" * 70)
print("กำลังทดสอบ API...")
print("=" * 70)

# รอให้ API พร้อม
print("\nรอ API Server...")
max_retries = 10
for i in range(max_retries):
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=2)
        if response.status_code == 200:
            print("✓ API Server พร้อมใช้งาน!\n")
            break
    except:
        print(f"  กำลังรอ... ({i+1}/{max_retries})")
        time.sleep(1)
else:
    print("\n✗ ไม่สามารถเชื่อมต่อ API Server")
    print("  กรุณาเปิด API Server ก่อน: python api.py")
    exit(1)

# ทดสอบ API endpoints
print("=" * 70)
print("1. ข้อมูลโมเดล")
print("=" * 70)
response = requests.get(f"{BASE_URL}/api/model/info")
print(json.dumps(response.json(), indent=2, ensure_ascii=False))

print("\n" + "=" * 70)
print("2. Feature Importances (Top 3)")
print("=" * 70)
response = requests.get(f"{BASE_URL}/api/model/feature-importances")
data = response.json()
for item in data['sorted_by_importance'][:3]:
    print(f"  Rank {item['rank']}: Feature {item['feature_index']} = {item['importance']:.6f} ({item['percentage']:.2f}%)")

print("\n" + "=" * 70)
print("3. ทำนายผล")
print("=" * 70)
test_data = {
    "features": [1.2, 3.4, 0.7, 2.1, 5.0, 1.8, 2.3, 4.5, 3.2]
}
print("Input:", test_data['features'])
response = requests.post(f"{BASE_URL}/api/predict", json=test_data)
result = response.json()
print(f"Output: {result['predictions']}")

print("\n" + "=" * 70)
print("✓ ทดสอบเสร็จสิ้น!")
print("=" * 70)
print(f"\nดู API ทั้งหมดได้ที่: {BASE_URL}/")
