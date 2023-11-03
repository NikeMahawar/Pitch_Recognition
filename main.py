import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def extract_features(file_path):
    y, sr = librosa.load(file_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    avg_pitch = np.mean(pitches)
    return [avg_pitch]

high_pitch_directory = "path_to_high_pitch_audio_folder"
low_pitch_directory = "path_to_low_pitch_audio_folder"

high_pitch_files = [f"{high_pitch_directory}/{file}" for file in os.listdir(high_pitch_directory)]
low_pitch_files = [f"{low_pitch_directory}/{file}" for file in os.listdir(low_pitch_directory)]

data = []
labels = []

for file in high_pitch_files:
    features = extract_features(file)
    data.append(features)
    labels.append("high")

for file in low_pitch_files:
    features = extract_features(file)
    data.append(features)
    labels.append("low")

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

def classify_voice(file_path):
    features = extract_features(file_path)
    prediction = clf.predict([features])[0]
    return prediction

sample_audio = "path_to_audio_file_to_classify"
result = classify_voice(sample_audio)
print(f"The voice in {sample_audio} is classified as: {result}")
