# Приклад шляху до завантажених даних
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, GlobalAveragePooling2D, \
    BatchNormalization, Lambda, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, EfficientNetB0, Xception, InceptionV3, InceptionResNetV2
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import time
from optimizers.CustomAdamV2 import CustomAdamV2, CustomAdamV3, CustomAdamV4, CustomAdamV5, CustomAdamV6, HybridOptimizer, AdamM_Hessian, HybridOptimizerSimple
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use('Agg')
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Step 1: List all files in the dataset directory
start_time = time.time()
dataset_path = '../../Celeb-DF-v2'
celeb_real_path = f'{dataset_path}/Celeb-real'
celeb_synthesis_path = f'{dataset_path}/Celeb-synthesis'
youtube_real_path = f'{dataset_path}/YouTube-real'
metadata_path = f'{dataset_path}/List_of_testing_videos.txt'

tf.config.run_functions_eagerly(True)

# Step 2: Load metadata from file
video_data = []
with open(metadata_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            label, video_path = parts
            label = 'REAL' if label == '1' else 'FAKE'
            video_data.append({'video': video_path, 'label': label})
        else:
            print(f"Skipping invalid line: {line.strip()}")

# Create DataFrame with initial video metadata
video_df = pd.DataFrame(video_data)

# Step 3: Add FAKE videos from Celeb-synthesis
synthesis_videos = os.listdir(celeb_synthesis_path)
fake_video_data = [{'video': f'Celeb-synthesis/{video}', 'label': 'FAKE'} for video in synthesis_videos]
all_video_data = video_data + fake_video_data
video_df = pd.DataFrame(all_video_data)

# Step 4: Create a balanced dataset by sampling FAKE videos
fake_videos = video_df[video_df['label'] == 'FAKE'].sample(n=178, random_state=42)
real_videos = video_df[video_df['label'] == 'REAL']
balanced_video_df = pd.concat([fake_videos, real_videos], ignore_index=True)


# Step 5: Define functions for frame extraction and preprocessing
def extract_golden_frames_from_video(video_path, num_frames=30, threshold=30.0):
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


def preprocess_video_data(video_df, video_dir, num_frames=10):
    X = []
    y = []
    video_filenames = []
    missing_videos = []

    for idx, row in video_df.iterrows():
        video_path = os.path.join(video_dir, row['video'])
        if os.path.exists(video_path):
            frames = extract_golden_frames_from_video(video_path, num_frames=num_frames)

            if len(frames) < num_frames:
                print(f"Video {row['video']} has insufficient frames. Expected: {num_frames}, Got: {len(frames)}.")
                missing_videos.append(row['video'])
                continue

            X.append(frames)
            y.append(1 if row['label'] == 'FAKE' else 0)
            video_filenames.append(row['video'])
        else:
            print(f"Video {row['video']} not found in directory {video_dir}.")
            missing_videos.append(row['video'])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print(f"Total processed videos: {len(X)}, Missing videos: {len(missing_videos)}")
    if missing_videos:
        print(f"Missing videos: {missing_videos[:5]} ... {len(missing_videos)} total")

    return X, y, video_filenames


def display_golden_frames_from_videos(video_df, video_dir, num_videos=4, num_frames=2):
    selected_videos = video_df.head(num_videos)
    plt.figure(figsize=(24, 24))

    frame_counter = 1
    for idx, row in selected_videos.iterrows():
        video_path = os.path.join(video_dir, row['video'])
        if not os.path.exists(video_path):
            print(f"Video {row['video']} not found in directory {video_dir}. Skipping.")
            continue

        video_class = row['label']
        golden_frames = extract_golden_frames_from_video(video_path, num_frames=num_frames)

        for i, frame in enumerate(golden_frames):
            plt.subplot(num_videos, num_frames, frame_counter)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title(f'File: {row["video"]}\nClass: {video_class}', fontsize=18)
            plt.axis('off')
            frame_counter += 1

    plt.tight_layout()
    plt.show()


# Step 6: Process videos only once
video_dir = dataset_path
X, y, video_filenames = preprocess_video_data(balanced_video_df, video_dir, num_frames=10)

# Step 7: Visualize frames
display_golden_frames_from_videos(balanced_video_df, video_dir, num_videos=4, num_frames=5)

# Step 8: Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")


# Визначення mish як функції активації
def mish(x):
    return x * K.tanh(K.softplus(x))


# Додаємо mish як активацію, яку можна використовувати в Keras
get_custom_objects().update({'mish': Activation(mish)})


# Функція для створення ResNet з Bidirectional LSTM та регуляризацією
def create_resnet_swish_bilstm_model(input_shape):
    input_tensor = Input(shape=input_shape)
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
    x = GlobalAveragePooling2D()(resnet_model.output)
    x = Dense(512, activation='swish')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Lambda(lambda x: tf.expand_dims(x, axis=1))(x)  # Підготовка для LSTM
    x = Bidirectional(LSTM(256, return_sequences=False))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='swish')(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=input_tensor, outputs=x)


# Функція для створення EfficientNet з додатковими шарами
def create_efficientnet_model(input_shape):
    input_tensor = Input(shape=input_shape)
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=input_tensor)
    x = GlobalAveragePooling2D()(efficientnet_model.output)
    x = Dense(512, activation=mish)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='swish')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='swish')(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=input_tensor, outputs=x)


# Функція для створення Xception з додатковими шарами
def create_xception_model(input_shape):
    input_tensor = Input(shape=input_shape)
    xception_model = Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)
    x = GlobalAveragePooling2D()(xception_model.output)
    x = Dense(512, activation=mish)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='swish')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='swish')(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=input_tensor, outputs=x)


# Функція для створення моделі InceptionV3 з додатковими шарами і регуляризацією
def create_inception_model(input_shape):
    input_tensor = Input(shape=input_shape)
    inception_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
    x = GlobalAveragePooling2D()(inception_model.output)
    x = Dense(512, activation=mish)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='swish')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='swish')(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=input_tensor, outputs=x)


# Функція для створення моделі Facenet (InceptionResNetV2) з додатковими шарами і регуляризацією
def create_facenet_model(input_shape):
    input_tensor = Input(shape=input_shape)
    facenet_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
    x = GlobalAveragePooling2D()(facenet_model.output)
    x = Dense(512, activation=mish)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='swish')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='swish')(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=input_tensor, outputs=x)

def log_results(epoch, logs):
    print(f"Epoch: {epoch}, Loss: {logs['loss']}, Accuracy: {logs['accuracy']}")

# Визначення вхідного розміру для всіх моделей
input_shape = (224, 224, 3)




# Створення моделей
resnet_model = create_resnet_swish_bilstm_model(input_shape)
efficientnet_model = create_efficientnet_model(input_shape)
xception_model = create_xception_model(input_shape)
inception_model = create_inception_model(input_shape)
facenet_model = create_facenet_model(input_shape)

# Вибір середнього кадру з кожної послідовності для тренування базових моделей
X_train_single_frame = X_train[:, X_train.shape[1] // 2]
X_val_single_frame = X_val[:, X_val.shape[1] // 2]

# Перетворення даних у TensorFlow Tensor
X_train_single_frame = tf.convert_to_tensor(X_train_single_frame, dtype=tf.float32)
X_val_single_frame = tf.convert_to_tensor(X_val_single_frame, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

# Ініціалізація оптимізаторів ПІСЛЯ створення моделей
resnet_optimizer = HybridOptimizerSimple(learning_rate=3e-4)
efficientnet_optimizer = HybridOptimizerSimple(learning_rate=3e-4)
xception_optimizer = HybridOptimizerSimple(learning_rate=3e-4)
inception_optimizer = HybridOptimizerSimple(learning_rate=3e-4)
facenet_optimizer = HybridOptimizerSimple(learning_rate=3e-4)

# Компіляція моделей після створення оптимізаторів
resnet_model.compile(optimizer=resnet_optimizer, loss="binary_crossentropy", metrics=["accuracy"])
efficientnet_model.compile(optimizer=efficientnet_optimizer, loss="binary_crossentropy", metrics=["accuracy"])
xception_model.compile(optimizer=xception_optimizer, loss="binary_crossentropy", metrics=["accuracy"])
inception_model.compile(optimizer=inception_optimizer, loss="binary_crossentropy", metrics=["accuracy"])
facenet_model.compile(optimizer=facenet_optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Використання колбеків
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Тренування моделей
resnet_model.fit(X_train_single_frame, y_train, epochs=50, batch_size=32, validation_data=(X_val_single_frame, y_val),
                 callbacks=[early_stopping, reduce_lr])

efficientnet_model.fit(X_train_single_frame, y_train, epochs=50, batch_size=32, validation_data=(X_val_single_frame, y_val),
                       callbacks=[early_stopping, reduce_lr])

xception_model.fit(X_train_single_frame, y_train, epochs=50, batch_size=32, validation_data=(X_val_single_frame, y_val),
                   callbacks=[early_stopping, reduce_lr])

inception_model.fit(X_train_single_frame, y_train, epochs=50, batch_size=32, validation_data=(X_val_single_frame, y_val),
                    callbacks=[early_stopping, reduce_lr])

facenet_model.fit(X_train_single_frame, y_train, epochs=50, batch_size=32, validation_data=(X_val_single_frame, y_val),
                  callbacks=[early_stopping, reduce_lr])

# Прогнозування для тренувальної вибірки
resnet_train_preds = resnet_model.predict(X_train_single_frame)
efficientnet_train_preds = efficientnet_model.predict(X_train_single_frame)
xception_train_preds = xception_model.predict(X_train_single_frame)
inception_train_preds = inception_model.predict(X_train_single_frame)
facenet_train_preds = facenet_model.predict(X_train_single_frame)

# Прогнозування для валідаційної вибірки
resnet_val_preds = resnet_model.predict(X_val_single_frame)
efficientnet_val_preds = efficientnet_model.predict(X_val_single_frame)
xception_val_preds = xception_model.predict(X_val_single_frame)
inception_val_preds = inception_model.predict(X_val_single_frame)
facenet_val_preds = facenet_model.predict(X_val_single_frame)

# Налаштування для вибору кількості кадрів
candidate_num_frames = [10]
best_accuracy = 0
best_num_frames = None
best_weights = None

# Перебір різної кількості кадрів
for num_frames in candidate_num_frames:
    print(f"\nTesting with {num_frames} frames per video...")

    train_preds_list = []
    val_preds_list = []

    # Генерація прогнозів базових моделей для кожного кадру
    for i in range(num_frames):
        resnet_train_preds = resnet_model.predict(X_train[:, i])
        efficientnet_train_preds = efficientnet_model.predict(X_train[:, i])
        xception_train_preds = xception_model.predict(X_train[:, i])
        inception_train_preds = inception_model.predict(X_train[:, i])
        facenet_train_preds = facenet_model.predict(X_train[:, i])

        train_meta_features = np.hstack((
            resnet_train_preds,
            efficientnet_train_preds,
            xception_train_preds,
            inception_train_preds,
            facenet_train_preds
        ))
        train_preds_list.append(train_meta_features)

        resnet_val_preds = resnet_model.predict(X_val[:, i])
        efficientnet_val_preds = efficientnet_model.predict(X_val[:, i])
        xception_val_preds = xception_model.predict(X_val[:, i])
        inception_val_preds = inception_model.predict(X_val[:, i])
        facenet_val_preds = facenet_model.predict(X_val[:, i])

        val_meta_features = np.hstack((
            resnet_val_preds,
            efficientnet_val_preds,
            xception_val_preds,
            inception_val_preds,
            facenet_val_preds
        ))
        val_preds_list.append(val_meta_features)

    # Обчислення середнього, медіани та стандартного відхилення для кожного відео
    train_meta_features_mean = np.mean(train_preds_list, axis=0)
    train_meta_features_median = np.median(train_preds_list, axis=0)
    train_meta_features_std = np.std(train_preds_list, axis=0)

    val_meta_features_mean = np.mean(val_preds_list, axis=0)
    val_meta_features_median = np.median(val_preds_list, axis=0)
    val_meta_features_std = np.std(val_preds_list, axis=0)

    print(f"train_meta_features.shape: {train_meta_features.shape}")
    print(f"val_meta_features.shape: {val_meta_features.shape}")
    print(f"train_preds_list.shape: {np.array(train_preds_list).shape}")
    print(f"val_preds_list.shape: {np.array(val_preds_list).shape}")

    # Формуємо остаточні набори ознак
    train_meta_features = np.hstack((train_meta_features_mean, train_meta_features_median, train_meta_features_std))
    val_meta_features = np.hstack((val_meta_features_mean, val_meta_features_median, val_meta_features_std))

    print(f"train_meta_features.shape after hstack: {train_meta_features.shape}")
    print(f"val_meta_features.shape after hstack: {val_meta_features.shape}")

    # Переконуємось, що розмірність збігається
    if train_meta_features.shape != val_meta_features.shape:
        print("⚠️ Увага! train_meta_features і val_meta_features мають різні розміри!")

    train_meta_features = train_meta_features.reshape(train_meta_features.shape[0], -1)
    val_meta_features = val_meta_features.reshape(val_meta_features.shape[0], -1)

    print("NaN у train_meta_features:", np.isnan(train_meta_features).sum())
    print("NaN у val_meta_features:", np.isnan(val_meta_features).sum())

    # Імп'ютація пропущених значень
    imputer = SimpleImputer(strategy="mean")
    train_meta_features = imputer.fit_transform(train_meta_features.copy())
    val_meta_features = imputer.transform(val_meta_features.copy())

    print("NaN у train_meta_features:", np.isnan(train_meta_features).sum())
    print("NaN у val_meta_features:", np.isnan(val_meta_features).sum())

    # Використовуємо крос-валідацію для підбору оптимальних ваг базових моделей
    def objective(weights):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_accuracy = []
        for train_idx, val_idx in kf.split(train_meta_features):
            train_fold, val_fold = train_meta_features[train_idx], train_meta_features[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            # Прогноз з поточними вагами
            ensemble_preds = np.dot(val_fold, weights)
            final_preds = (ensemble_preds > 0.5).astype(int)
            cv_accuracy.append(accuracy_score(y_val_fold, final_preds))
        return 1 - np.mean(cv_accuracy)


    # Початкові значення ваг
    num_features = train_meta_features.shape[1]  # Очікується 15
    initial_weights = np.full((num_features,), 1 / num_features)

    constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}  # Сума ваг = 1
    bounds = [(0, 1)] * num_features

    # Оптимізація ваг
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
    optimized_weights = result.x

    # Прогноз з оптимальними вагами для валідаційної вибірки
    ensemble_preds = np.dot(val_meta_features, optimized_weights)
    final_preds = (ensemble_preds > 0.5).astype(int)

    # Оцінка точності для валідаційної вибірки
    ensemble_accuracy = accuracy_score(y_val, final_preds)
    print(f"Accuracy with {num_frames} frames and optimized weights: {ensemble_accuracy}")

    # Зберігаємо найкращі результати
    if ensemble_accuracy > best_accuracy:
        best_accuracy = ensemble_accuracy
        best_num_frames = num_frames
        best_weights = optimized_weights

print(f"\nBest configuration: {best_num_frames} frames with accuracy {best_accuracy}")
print(f"Optimized weights: {best_weights}")

# Використання вдосконалених ансамблевих моделей для метамоделі
meta_model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
meta_model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
meta_model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Навчання метамоделей
meta_model_rf.fit(train_meta_features, y_train)
meta_model_gb.fit(train_meta_features, y_train)
meta_model_xgb.fit(train_meta_features, y_train)

# Прогнози метамоделей на валідаційній вибірці
rf_preds = meta_model_rf.predict(val_meta_features)
gb_preds = meta_model_gb.predict(val_meta_features)
xgb_preds = meta_model_xgb.predict(val_meta_features)

# Оцінка точності кожної метамоделі
rf_accuracy = accuracy_score(y_val, rf_preds)
gb_accuracy = accuracy_score(y_val, gb_preds)
xgb_accuracy = accuracy_score(y_val, xgb_preds)

print(f"Random Forest Meta Model Accuracy: {rf_accuracy}")
print(f"Gradient Boosting Meta Model Accuracy: {gb_accuracy}")
print(f"XGBoost Meta Model Accuracy: {xgb_accuracy}")

# Налаштування для вибору кількості кадрів
candidate_num_frames = [10]
best_accuracy = 0
best_num_frames = None

# Перебір різної кількості кадрів
for num_frames in candidate_num_frames:
    print(f"\nTesting with {num_frames} frames per video...")

    train_preds_list = []
    val_preds_list = []

    # Генерація прогнозів базових моделей для кожного кадру
    for i in range(num_frames):
        resnet_train_preds = resnet_model.predict(X_train[:, i])
        efficientnet_train_preds = efficientnet_model.predict(X_train[:, i])
        xception_train_preds = xception_model.predict(X_train[:, i])
        inception_train_preds = inception_model.predict(X_train[:, i])
        facenet_train_preds = facenet_model.predict(X_train[:, i])

        train_meta_features = np.hstack((
            resnet_train_preds,
            efficientnet_train_preds,
            xception_train_preds,
            inception_train_preds,
            facenet_train_preds
        ))
        train_preds_list.append(train_meta_features)

        resnet_val_preds = resnet_model.predict(X_val[:, i])
        efficientnet_val_preds = efficientnet_model.predict(X_val[:, i])
        xception_val_preds = xception_model.predict(X_val[:, i])
        inception_val_preds = inception_model.predict(X_val[:, i])
        facenet_val_preds = facenet_model.predict(X_val[:, i])

        val_meta_features = np.hstack((
            resnet_val_preds,
            efficientnet_val_preds,
            xception_val_preds,
            inception_val_preds,
            facenet_val_preds
        ))
        val_preds_list.append(val_meta_features)

    # Обчислення середнього, медіани та стандартного відхилення для кожного відео
    train_meta_features_mean = np.mean(train_preds_list, axis=0)
    train_meta_features_median = np.median(train_preds_list, axis=0)
    train_meta_features_std = np.std(train_preds_list, axis=0)
    val_meta_features_mean = np.mean(val_preds_list, axis=0)
    val_meta_features_median = np.median(val_preds_list, axis=0)
    val_meta_features_std = np.std(val_preds_list, axis=0)

    # Формуємо остаточні набори ознак для метамоделі
    train_meta_features = np.hstack((train_meta_features_mean, train_meta_features_median, train_meta_features_std))
    val_meta_features = np.hstack((val_meta_features_mean, val_meta_features_median, val_meta_features_std))

    # Використання логістичної регресії як метамоделі для автоматичної оптимізації ваг
    meta_model_log_reg = LogisticRegression(max_iter=500, solver='liblinear', random_state=42)
    meta_model_log_reg.fit(train_meta_features, y_train)

    # Оцінка точності метамоделі
    meta_val_preds = meta_model_log_reg.predict(val_meta_features)
    ensemble_accuracy = accuracy_score(y_val, meta_val_preds)
    print(f"Logistic Regression Meta Model Accuracy with {num_frames} frames: {ensemble_accuracy}")

    # Зберігаємо найкращі результати
    if ensemble_accuracy > best_accuracy:
        best_accuracy = ensemble_accuracy
        best_num_frames = num_frames
        best_weights = meta_model_log_reg.coef_.flatten()

print(f"\nBest configuration: {best_num_frames} frames with accuracy {best_accuracy}")
print(f"Optimized weights: {best_weights}")

# Звіт важливості ознак для оптимальної конфігурації
features = [f'{model}_{stat}' for model in ['ResNet', 'EfficientNet', 'Xception', 'Inception', 'Facenet']
            for stat in ['mean', 'median', 'std']]
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': best_weights
}).sort_values(by='Importance', ascending=False)

# Візуалізація важливості ознак
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance with Logistic Regression Weights')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Друк таблиці важливості ознак
print(importance_df)

# Використання вдосконалених ансамблевих моделей для метамоделі
meta_model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
meta_model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
meta_model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Навчання метамоделей
meta_model_rf.fit(train_meta_features, y_train)
meta_model_gb.fit(train_meta_features, y_train)
meta_model_xgb.fit(train_meta_features, y_train)

# Прогнози метамоделей на валідаційній вибірці
rf_preds = meta_model_rf.predict(val_meta_features)
gb_preds = meta_model_gb.predict(val_meta_features)
xgb_preds = meta_model_xgb.predict(val_meta_features)

# Оцінка точності кожної метамоделі
rf_accuracy = accuracy_score(y_val, rf_preds)
gb_accuracy = accuracy_score(y_val, gb_preds)
xgb_accuracy = accuracy_score(y_val, xgb_preds)

print(f"Random Forest Meta Model Accuracy: {rf_accuracy}")
print(f"Gradient Boosting Meta Model Accuracy: {gb_accuracy}")
print(f"XGBoost Meta Model Accuracy: {xgb_accuracy}")

print("--- %s seconds ---" % (time.time() - start_time))

print("finish")
