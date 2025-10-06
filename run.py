import pickle
import json
import warnings
try:
    import joblib
    has_joblib = True
except ImportError:
    has_joblib = False

# อ่านไฟล์ best_params.json
with open('best_params.json', 'r', encoding='utf-8') as f:
    best_params = json.load(f)

# แสดงทุก parameters ที่มีในไฟล์
print("=" * 70)
print("ALL BEST PARAMETERS FROM best_params.json")
print("=" * 70)
for model_name, params in best_params.items():
    print(f"\n{model_name}:")
    print("-" * 70)
    for param_name, param_value in params.items():
        print(f"  {param_name:25s}: {param_value}")

# โฟกัสที่ ExtraTrees parameters
print("\n" + "=" * 70)
print("EXTRATREES PARAMETERS (สำหรับโมเดล ExtraTrees_BEST.pkl)")
print("=" * 70)
extra_trees_params = best_params.get('ExtraTrees', {})
for param_name, param_value in extra_trees_params.items():
    print(f"  {param_name:25s}: {param_value}")
print("=" * 70)

# พยายามโหลดโมเดล ExtraTrees จากไฟล์
print("\nกำลังโหลดโมเดล ExtraTrees_BEST.pkl...")
model = None
error_msg = None

# ลองใช้ joblib ก่อน (แนะนำสำหรับ scikit-learn)
if has_joblib:
    try:
        print("  พยายามโหลดด้วย joblib...")
        model = joblib.load('ExtraTrees_BEST.pkl')
        print("  ✓ โหลดสำเร็จด้วย joblib!")
    except Exception as e:
        error_msg = f"joblib: {e}"
        print(f"  ✗ ไม่สำเร็จ: {e}")

# ถ้าไม่สำเร็จ ลองใช้ pickle
if model is None:
    try:
        print("  พยายามโหลดด้วย pickle...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            with open('ExtraTrees_BEST.pkl', 'rb') as f:
                model = pickle.load(f)
        print("  ✓ โหลดสำเร็จด้วย pickle!")
    except Exception as e:
        if error_msg:
            error_msg += f"\npickle: {e}"
        else:
            error_msg = f"pickle: {e}"
        print(f"  ✗ ไม่สำเร็จ: {e}")

if model is not None:
    # แสดงข้อมูลโมเดล
    print("\n" + "=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)
    print(f"Model Type: {type(model).__name__}")
    print(f"Number of Estimators: {model.n_estimators}")
    print(f"Number of Features: {model.n_features_in_}")
    if hasattr(model, 'n_outputs_'):
        print(f"Number of Outputs: {model.n_outputs_}")
    
    # แสดงข้อมูลดิบ Feature Importances
    if hasattr(model, 'feature_importances_'):
        print(f"\nFeature Importances (Raw Data):")
        importances = model.feature_importances_
        print(importances)
        
        print(f"\nFeature Importances (List):")
        for idx, importance in enumerate(importances):
            print(f"Feature_{idx}: {importance}")
    
    print("=" * 70)
    
    # ตัวอย่างการใช้โมเดลทำนาย (ถ้ามีข้อมูล)
    # สมมุติว่าคุณมีข้อมูลใหม่ (X_new)
    # X_new = [[1.2, 3.4, 0.7, 2.1, 5.0]]  # ต้องมีจำนวนฟีเจอร์เท่ากับโมเดล
    # y_pred = model.predict(X_new)
    # print("\nผลการทำนาย:", y_pred)
    
    print("\n✓ โหลดและแสดงผลสำเร็จ!")
else:
    print(f"\n⚠ เกิดข้อผิดพลาดในการโหลดโมเดล:")
    print(f"  {error_msg}")
    print("\nอาจเป็นเพราะ version ของ scikit-learn ไม่ตรงกัน หรือไฟล์เสียหาย")
    print("แต่ยังสามารถดู parameters ที่ดีที่สุดได้จาก best_params.json ข้างต้น")
    print("\n✓ แสดง parameters สำเร็จ!")
