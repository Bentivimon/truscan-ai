import concurrent.futures
import datetime
import os
import pickle
import time
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import cv2
import tensorflow as tf
import uuid
from deepface import DeepFace

from fastapi import File, UploadFile
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import model_from_json
from tensorflow.keras.optimizers import Adam


class VideoProcessResponse(object):
    result: str
    video_path: str
    demographics: dict
    probability: float


def train_log_reg(x_train, y_train):
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    return "logistic_regression", log_reg


def train_svm(x_train, y_train):
    svm_model = SVC(probability=True)
    svm_model.fit(x_train, y_train)
    return "svm", svm_model


def train_rf(x_train, y_train):
    rf_model = RandomForestClassifier()
    rf_model.fit(x_train, y_train)
    return "random_forest", rf_model


def train_gb(x_train, y_train):
    gb_model = GradientBoostingClassifier()
    gb_model.fit(x_train, y_train)
    return "gradient_boosting", gb_model


def train_knn(x_train, y_train):
    knn_model = KNeighborsClassifier()
    knn_model.fit(x_train, y_train)
    return "knn", knn_model


def train_dt(x_train, y_train):
    dt_model = DecisionTreeClassifier()
    dt_model.fit(x_train, y_train)
    return "decision_tree", dt_model


def train_xgb(x_train, y_train):
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb_model.fit(x_train, y_train)
    return "xgboost", xgb_model


def train_ada(x_train, y_train):
    ada_model = AdaBoostClassifier()
    ada_model.fit(x_train, y_train)
    return "adaboost", ada_model


def train_models(x_train, y_train):
    models = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(train_log_reg, x_train, y_train),
            executor.submit(train_svm, x_train, y_train),
            executor.submit(train_rf, x_train, y_train),
            # executor.submit(train_gb, x_train, y_train),
            # executor.submit(train_knn, x_train, y_train),
            # executor.submit(train_dt, x_train, y_train),
            # executor.submit(train_xgb, x_train, y_train),
            # executor.submit(train_ada, x_train, y_train),
        ]

        for future in concurrent.futures.as_completed(futures):
            model_name, model = future.result()
            models[model_name] = model

    return models


def blend_models_predict(models, x_test_transformed):
    predictions = np.zeros((x_test_transformed.shape[0], len(models)))

    def multiple_predictions(i, model):
        predictions[:, i] = model.predict_proba(x_test_transformed)[:, 1]

    executor = concurrent.futures.ThreadPoolExecutor()
    futures = [
        executor.submit(multiple_predictions, i, model)
        for i, (model_name, model) in enumerate(models.items())
    ]
    concurrent.futures.wait(futures)

    blended_predictions = np.mean(
        predictions,
        axis=1,
    )

    return blended_predictions


def is_models_exist():
    return len(get_models_files()) > 0


def save_models(models, folder_path):
    files_array = []
    # then use on reading models
    for name, model in models.items():
        filename = f"{folder_path}/{name}_model.sav"
        joblib.dump(model, filename)
        files_array.append(f"{name};{filename}\n")

    with open(f"{folder_path}/models.txt", "w", encoding="utf-8") as file:
        for row in files_array:
            file.write(row)


def get_models_files():
    # TODO get models from 'models' file
    in_folder = os.listdir(os.getcwd() + "/models/models")
    return [file for file in in_folder if file.endswith((".sav"))]


def load_models() -> dict[str, DataFrame]:
    files = get_models_files()
    models: dict[str, DataFrame] = {}

    for file in files:
        # TODO set model name from file 'models'
        models[file] = joblib.load("models/models/" + file)
    return models


def process_single_record(string_for_testing: str) -> int:
    vectorizer_folder = "models/models/vectorizer"
    vectorizer = TfidfVectorizer()
    with open(f"{vectorizer_folder}/tfidf.pickle", "rb") as tfidf_file:
        vectorizer = pickle.load(tfidf_file)
    x_test_transformed = vectorizer.transform([string_for_testing])

    if not is_models_exist():
        raise FileNotFoundError("Trained models was not found!")
    else:
        models = load_models()
    blended_predictions = blend_models_predict(models, x_test_transformed)
    predicted = round(float(blended_predictions), 2)

    # to percent
    return predicted * 100


# def is_text_column_missed():
#     return False


def process_csv_file(file: UploadFile = File(...)):
    # TODO add validation
    # if is_text_column_missed():
    #     return

    test_df = pd.read_csv(file.file)
    test_df.dropna(subset=["Text"], inplace=True)

    x_test = test_df["Text"]
    y_test = test_df["Label"]

    vectorizer_folder = os.getcwd() + "/models/models/vectorizer"
    vectorizer = TfidfVectorizer()

    if not is_models_exist():

        train_csv_file = "models/fake/merged_dataset_ukr.csv"
        train_df = pd.read_csv(train_csv_file)
        train_df.dropna(subset=["Text"], inplace=True)

        x_train = train_df["Text"]
        y_train = train_df["Label"]
        tfidf = vectorizer.fit(x_train)
        x_train_transformed = vectorizer.transform(x_train)
        x_test_transformed = vectorizer.transform(x_test)

        pickle.dump(tfidf, open(f"{vectorizer_folder}/tfidf.pickle", "wb"))
        pickle.dump(
            x_train_transformed,
            open(f"{vectorizer_folder}/x_train_transformed.pickle", "wb"),
        )
        pickle.dump(
            x_test_transformed,
            open(f"{vectorizer_folder}/x_test_transformed.pickle", "wb"),
        )

        models = train_models(x_train_transformed, y_train)
        saving_folder_path = f"{os.getcwd()}/models/models"
        save_models(models, saving_folder_path)
    else:

        with open(f"{vectorizer_folder}/tfidf.pickle", "rb") as tfidf_file:
            vectorizer = pickle.load(tfidf_file)
        x_test_transformed = vectorizer.transform(x_test)

        models = load_models()

    blended_predictions = blend_models_predict(models, x_test_transformed)
    predictions: list[int] = (blended_predictions > 0.5).astype(int)

    # print(classification_report(y_test, predictions))

    test_df["Predicted Label"] = predictions

    unix_timestamp = datetime.datetime.timestamp(datetime.datetime.utcnow())
    result_file_name = f"models/result/test_results_{unix_timestamp}.csv"
    test_df.to_csv(result_file_name, index=False)

    # TODO remove created file

    return result_file_name


def save_corrected_answer(text: str, corrected_label: bool):
    df = pd.DataFrame({"Text": [text], "Label": [corrected_label]})

    file_name = "models/corrected_answers.csv"
    if os.path.exists(file_name):
        df.to_csv(file_name, mode="a", header=False, index=False)
    else:
        df.to_csv(file_name, mode="w", header=True, index=False)


def load_video_models():
    model_names = ["xception", "resnet", "efficientnet", "inception", "facenet"]
    models = {}

    for name in model_names:
        json_path = f"models/video-models/{name}_model_json.json"
        weights_path = f"models/video-models/{name}_model.weights.h5"
        if not os.path.exists(json_path) or not os.path.exists(weights_path):
            print(f"[WARN] Missing files for model '{name}'. Skipping...")
            continue

        with open(json_path, "r") as f:
            model_json = f.read().rstrip()
        model = model_from_json(model_json)
        model.load_weights(weights_path)
        models[name] = model

    return models

def analyze_demographics(frame):
    try:
        result = DeepFace.analyze(frame, actions=['age', 'gender', 'race'], enforce_detection=False)
        return {
            "age": result[0]["age"],
            "gender": result[0]["gender"],
            "race": result[0]["dominant_race"]
        }
    except Exception as e:
        print(f"Demographic analysis failed: {e}")
        return {
            "age": None,
            "gender": None,
            "race": None
        }

def predict_video_ensemble(models_dict, video_path, num_frames=10):
    frames = preprocess_single_video(video_path, num_frames)

    if frames is None:
        return "Error: Video has insufficient frames."

    middle_frame = frames[frames.shape[0] // 2].astype(np.uint8)
    demographic_info = analyze_demographics(middle_frame)

    frame_to_predict = np.expand_dims(middle_frame / 255.0, axis=0)
    predictions = [model.predict(frame_to_predict)[0][0] for model in models_dict.values()]
    avg_prediction = np.mean(predictions)

    result = "FAKE" if avg_prediction > 0.5 else "REAL"

    return {
        "prediction": result,
        "probability": round(avg_prediction, 4),
        "demographics": demographic_info
    }

async def process_video_file(file: UploadFile = File(...)):
    video_path = os.path.join('video-to-proccess', str(uuid.uuid4()) + '-' + file.filename)
    await chunked_copy(file, video_path)

    models_dict = load_video_models()
    if not models_dict:
        raise RuntimeError("No video models found for ensemble prediction!")

    predict_result = predict_video_ensemble(models_dict, video_path)

    response = VideoProcessResponse()
    response.result = predict_result["prediction"]
    response.video_path = video_path

    # Демографія + ймовірність
    response.demographics = predict_result["demographics"]
    response.probability = predict_result["probability"]

    return response


async def chunked_copy(src, dst):
    await src.seek(0)
    with open(dst, "wb") as buffer:
        while True:
            contents = await src.read(2 ** 20)
            if not contents:
                break
            buffer.write(contents)

def extract_golden_frames_from_video(video_path, num_frames=10, threshold=30.0):
    cap = cv2.VideoCapture(video_path)
    golden_frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0:
        print(f"Skipping video {video_path}: cannot read video file or it has no frames.")
        cap.release()
        return []

    ret, prev_frame = cap.read()
    if not ret:
        print(f"Skipping video {video_path}: cannot read the first frame.")
        cap.release()
        return []

    prev_frame = cv2.resize(prev_frame, (224, 224))
    golden_frames.append(prev_frame)

    step = max(1, frame_count // num_frames)
    for i in range(step, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            diff = cv2.absdiff(prev_frame, frame)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            score = np.sum(diff_gray) / diff_gray.size

            if score > threshold:
                golden_frames.append(frame)
                prev_frame = frame

            if len(golden_frames) >= num_frames:
                break

    while len(golden_frames) < num_frames:
        golden_frames.append(golden_frames[-1])

    cap.release()
    return golden_frames


def preprocess_single_video(video_path, num_frames=10):
    frames = extract_golden_frames_from_video(video_path, num_frames=num_frames)

    if len(frames) < num_frames:
        print(f"Video {video_path} has insufficient frames. Expected: {num_frames}, Got: {len(frames)}.")
        return None

    frames = np.array(frames, dtype=np.float32)
    return frames


def predict_video(model, video_path, num_frames=10):
    frames = preprocess_single_video(video_path, num_frames)

    if frames is None:
        return "Error: Video has insufficient frames."

    frame_to_predict = frames[frames.shape[0] // 2]  # Вибір середнього кадру
    frame_to_predict = np.expand_dims(frame_to_predict, axis=0)  # Додаємо batch dimension

    prediction = model.predict(frame_to_predict)
    result = 'FAKE' if prediction[0] > 0.5 else 'REAL'
    return result


def save_correct_video_answer(video_path, wrong_answer):
    label = 1 if wrong_answer == 'REAL' else 0

    df = pd.DataFrame({"VideoPath": [video_path], "Label": [label]})
    file_name = "models/corrected_video_answers.csv"
    if os.path.exists(file_name):
        df.to_csv(file_name, mode="a", header=False, index=False)
    else:
        df.to_csv(file_name, mode="w", header=True, index=False)