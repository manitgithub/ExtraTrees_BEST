from flask import Flask, jsonify, request
import pickle
import json
import warnings
try:
    import joblib
    has_joblib = True
except ImportError:
    has_joblib = False
import numpy as np

app = Flask(__name__)

# โหลดโมเดลและ parameters ตอนเริ่มต้น
model = None
best_params = None

def load_model():
    global model
    if has_joblib:
        try:
            model = joblib.load('ExtraTrees_BEST.pkl')
            print("✓ โหลดโมเดลสำเร็จด้วย joblib!")
            return True
        except Exception as e:
            print(f"✗ joblib error: {e}")
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            with open('ExtraTrees_BEST.pkl', 'rb') as f:
                model = pickle.load(f)
        print("✓ โหลดโมเดลสำเร็จด้วย pickle!")
        return True
    except Exception as e:
        print(f"✗ pickle error: {e}")
        return False

def load_params():
    global best_params
    try:
        with open('best_params.json', 'r', encoding='utf-8') as f:
            best_params = json.load(f)
        print("✓ โหลด parameters สำเร็จ!")
        return True
    except Exception as e:
        print(f"✗ Error loading params: {e}")
        return False

# โหลดข้อมูลตอนเริ่มต้น
load_model()
load_params()

@app.route('/')
def home():
    """หน้าแรก - แสดงข้อมูล API endpoints"""
    return jsonify({
        "message": "ExtraTrees Model API",
        "endpoints": {
            "/api/model/info": "GET - ข้อมูลโมเดล",
            "/api/model/params": "GET - Parameters ทั้งหมด",
            "/api/model/params/extratrees": "GET - Parameters ของ ExtraTrees",
            "/api/model/feature-importances": "GET - Feature Importances",
            "/api/predict": "POST - ทำนายผล (ต้องส่ง features)"
        }
    })

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """ดึงข้อมูลโมเดล"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    info = {
        "model_type": type(model).__name__,
        "n_estimators": int(model.n_estimators),
        "n_features": int(model.n_features_in_),
        "n_outputs": int(model.n_outputs_) if hasattr(model, 'n_outputs_') else None,
        "max_depth": model.max_depth,
        "min_samples_split": model.min_samples_split,
        "min_samples_leaf": model.min_samples_leaf,
    }
    
    return jsonify(info)

@app.route('/api/model/params', methods=['GET'])
def get_all_params():
    """ดึง parameters ทั้งหมดจาก best_params.json"""
    if best_params is None:
        return jsonify({"error": "Parameters not loaded"}), 500
    
    return jsonify(best_params)

@app.route('/api/model/params/extratrees', methods=['GET'])
def get_extratrees_params():
    """ดึง parameters ของ ExtraTrees เท่านั้น"""
    if best_params is None:
        return jsonify({"error": "Parameters not loaded"}), 500
    
    return jsonify(best_params.get('ExtraTrees', {}))

@app.route('/api/model/feature-importances', methods=['GET'])
def get_feature_importances():
    """ดึง Feature Importances"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    if not hasattr(model, 'feature_importances_'):
        return jsonify({"error": "Model doesn't have feature importances"}), 400
    
    importances = model.feature_importances_
    
    # สร้างข้อมูลในรูปแบบต่างๆ
    result = {
        "raw_array": importances.tolist(),
        "features": [
            {
                "feature_index": i,
                "importance": float(imp),
                "percentage": float(imp * 100)
            }
            for i, imp in enumerate(importances)
        ],
        "sorted_by_importance": [
            {
                "rank": rank,
                "feature_index": idx,
                "importance": float(imp),
                "percentage": float(imp * 100)
            }
            for rank, (idx, imp) in enumerate(
                sorted(enumerate(importances), key=lambda x: x[1], reverse=True), 
                start=1
            )
        ],
        "total_sum": float(sum(importances))
    }
    
    return jsonify(result)

@app.route('/api/predict', methods=['POST'])
def predict():
    """ทำนายผลจากข้อมูลที่ส่งมา
    
    Request Body (JSON):
    {
        "features": [value1, value2, ..., value9]
    }
    หรือ
    {
        "features": [[value1, value2, ..., value9], [value1, value2, ..., value9]]
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({
                "error": "Missing 'features' in request body",
                "example": {
                    "features": [1.2, 3.4, 0.7, 2.1, 5.0, 1.8, 2.3, 4.5, 3.2]
                }
            }), 400
        
        features = np.array(data['features'])
        
        # ตรวจสอบว่าเป็น 1D หรือ 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # ตรวจสอบจำนวน features
        if features.shape[1] != model.n_features_in_:
            return jsonify({
                "error": f"Expected {model.n_features_in_} features, got {features.shape[1]}"
            }), 400
        
        # ทำนาย
        predictions = model.predict(features)
        
        result = {
            "predictions": predictions.tolist(),
            "n_samples": int(features.shape[0]),
            "features_used": int(features.shape[1])
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """ตรวจสอบสถานะของ API"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "params_loaded": best_params is not None
    })

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("ExtraTrees Model REST API")
    print("=" * 70)
    print("API endpoints:")
    print("  GET  /                              - หน้าแรก")
    print("  GET  /api/health                    - ตรวจสอบสถานะ")
    print("  GET  /api/model/info                - ข้อมูลโมเดล")
    print("  GET  /api/model/params              - Parameters ทั้งหมด")
    print("  GET  /api/model/params/extratrees   - Parameters ของ ExtraTrees")
    print("  GET  /api/model/feature-importances - Feature Importances")
    print("  POST /api/predict                   - ทำนายผล")
    print("=" * 70)
    print("\nเริ่มต้น API Server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
