from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from fastapi.staticfiles import StaticFiles
import tempfile
import os
from pydantic import BaseModel

app = FastAPI()

# Подключение статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/wb.html")


class DataRequest(BaseModel):
    data: list

def normalize_with_iqr(df):
    df = df.copy()
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].min() >= 0 and df[col].max() <= 1:
            print(f"Колонка '{col}' уже в диапазоне [0, 1], пропускаем")
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

@app.post("/predict/csv/")
async def predict_csv(file: UploadFile = File(...)):
    try:
        # Сохраняем загруженный файл во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Загрузка тестовых данных
        test_data = pd.read_csv(tmp_path)

        # Предобработка данных
        columns_to_drop = ['CreatedDate', 'user_id', 'nm_id', 'mean_number_of_ordered_items', 'min_number_of_ordered_items', 'max_number_of_ordered_items', 'IsPaid_bit_0']
        test_data = test_data.drop(columns=[col for col in columns_to_drop if col in test_data.columns])

        norm_test_data = normalize_with_iqr(test_data)

        # Разделение на признаки и целевую переменную
        X_final = norm_test_data.drop(columns=['target'])
        y_final = norm_test_data['target']

        # Загрузка модели из файла
        loaded_model = CatBoostClassifier()
        loaded_model.load_model('best_catboost_model.cbm')

        # Предсказание вероятностей
        y_proba = loaded_model.predict_proba(X_final)

        # Предсказание классов
        y_pred = loaded_model.predict(X_final)

        # Оценка модели
        precision = precision_score(y_final, y_pred)
        recall = recall_score(y_final, y_pred)
        f1 = f1_score(y_final, y_pred)

        # Полный отчёт по классам
        report = classification_report(y_final, y_pred, output_dict=True)

        # Формируем результаты с уверенностью
        results = []
        for idx, (pred, proba) in enumerate(zip(y_pred, y_proba)):
            results.append({
                "prediction": int(pred),
                "confidence": float(max(proba))
            })

        # Удаляем временный файл
        os.unlink(tmp_path)

        return JSONResponse(content={
            "results": results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/json/")
async def predict_json(request: DataRequest):
    try:
        # Преобразуем JSON-данные в DataFrame
        test_data = pd.DataFrame(request.data)

        # Предобработка данных
        columns_to_drop = ['CreatedDate', 'user_id', 'nm_id', 'mean_number_of_ordered_items', 'min_number_of_ordered_items', 'max_number_of_ordered_items', 'IsPaid_bit_0']
        test_data = test_data.drop(columns=[col for col in columns_to_drop if col in test_data.columns])

        norm_test_data = normalize_with_iqr(test_data)

        # Разделение на признаки и целевую переменную
        X_final = norm_test_data.drop(columns=['target'])
        y_final = norm_test_data['target']

        # Загрузка модели из файла
        loaded_model = CatBoostClassifier()
        loaded_model.load_model('best_catboost_model.cbm')

        # Предсказание вероятностей
        y_proba = loaded_model.predict_proba(X_final)

        # Предсказание классов
        y_pred = loaded_model.predict(X_final)

        # Оценка модели
        precision = precision_score(y_final, y_pred)
        recall = recall_score(y_final, y_pred)
        f1 = f1_score(y_final, y_pred)

        # Полный отчёт по классам
        report = classification_report(y_final, y_pred, output_dict=True)

        # Формируем результаты с уверенностью
        results = []
        for idx, (pred, proba) in enumerate(zip(y_pred, y_proba)):
            results.append({
                "prediction": int(pred),
                "confidence": float(max(proba))
            })

        return JSONResponse(content={
            "results": results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
