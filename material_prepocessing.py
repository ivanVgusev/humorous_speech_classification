import tgt
import librosa
import os
import soundfile as sf
import opensmile
import pandas as pd
from progress_monitor import progress_bar

# Инициализация OpenSMILE
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals
)

directory = "/Users/ivanguseff/VScode/SEW/1_round"
data = []

for dirpath, _, elements in os.walk(directory):
    for file_index, element in enumerate(elements):
        if not element.endswith(".TextGrid"):
            continue

        element_strip = element.replace(".TextGrid", "")
        audio_name = element_strip + ".wav"

        ann_fpath = os.path.join(dirpath, element)
        audio_fpath = os.path.join(dirpath, audio_name)

        if not os.path.exists(audio_fpath):
            continue

        grid = tgt.io.read_textgrid(ann_fpath)
        syntagms_tier = grid.get_tier_by_name("syntagm (j/n)")

        audio, sr = librosa.load(audio_fpath, sr=None)

        progress_bar_total = len(syntagms_tier) + len(elements)
        for index, syntagm in enumerate(syntagms_tier):
            progress_bar(progress_bar_total, index)

            start_sample = int(sr * syntagm.start_time)
            end_sample = int(sr * syntagm.end_time)
            annotation = syntagm.text.strip().lower()

            if annotation not in ["j", "n"]:
                continue

            segment = audio[start_sample:end_sample]

            # Сохраняем во временный файл в директории
            tmp_path = "/tmp/temp.wav"
            sf.write(tmp_path, segment, sr)

            # Извлекаем признаки
            features = smile.process_file(tmp_path)
            features["label"] = 1 if annotation == "j" else 0
            features["file"] = element_strip
            features["start"] = syntagm.start_time
            features["end"] = syntagm.end_time

            data.append(features)
            os.remove(tmp_path)

# Объединяем в DataFrame
df = pd.concat(data, ignore_index=True)
df.to_csv("features.csv", index = False)
