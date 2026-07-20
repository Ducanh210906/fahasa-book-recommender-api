from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Fahasa Book Recommender API")

model = joblib.load('book_recommender_model.pkl')
le = joblib.load('label_encoder.pkl')

class KhachHang(BaseModel):
    tuoi: int
    gioi_tinh: int
    nghe_nghiep: int

@app.get("/")
def home():
    return {"message": "API gợi ý sách Fahasa đang chạy"}

@app.post("/predict")
def predict(khach: KhachHang):
    X_new = pd.DataFrame([[khach.tuoi, khach.gioi_tinh, khach.nghe_nghiep]],
                          columns=['Tuoi', 'Giới tính', 'Nghề nghiệp'])
    proba = model.predict_proba(X_new)[0]
    top_idx = proba.argsort()[::-1][:2]
    ket_qua = [{"the_loai": le.classes_[i], "do_tin_cay": round(float(proba[i])*100, 1)} for i in top_idx]
    return {"goi_y": ket_qua}
