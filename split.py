from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import numpy as np
np.random.seed(1)
from os.path import exists

def split_array_randomly(array):
    indices = np.arange(len(array))
    np.random.shuffle(indices)

    split_point = len(array) // 2

    first_half_indices = indices[:split_point]
    second_half_indices = indices[split_point:]

    first_half = array[first_half_indices]
    second_half = array[second_half_indices]

    return first_half, second_half

print('Loading Emo-Soundscapes...')
soundscapes_path = './data/raw/Emo-Soundscapes/Emo-Soundscapes-Audio/600_Sounds'
soundscapes = sorted(list(glob.glob(f"{soundscapes_path}/*/**.wav")))

print('Loading Environmental Sounds...')
envs_path = './data/raw/environmental-sound'
envs = sorted(list(glob.glob(f"{envs_path}/*/**.wav")))

print('Loading FSL10K loops...')
loops_path = './data/raw/FSL10K/audio'
loops = sorted(list(glob.glob(f"{loops_path}/*/**.wav")))

print('Loading CommonVoice...')
voices_path = './data/raw/CommonVoice'
voices = sorted(list(glob.glob(f"{voices_path}/*/*/**.mp3")))

def categorize(a):
    if 'Emo-Soundscapes' in a:
        return 'Emo-Soundscapes'
    elif 'environmental-sound' in a:
        return 'environmental-sound'
    elif 'FSL10K' in a:
        return 'FSL10K'
    else:
        return 'CommonVoice'

def get_other_file(x):
    folder = categorize(x)
    file_name = x.split('/')[-1]
    new_path = f"./data/reconstructed/{folder}/GL_" + file_name
    return new_path.replace('.mp3', '.wav')

X = soundscapes + envs + loops + voices
X = [x for x in X if exists(get_other_file(x))]
X = np.array(X)


print('Choosing reconstructed split...')
original, reconstructed = split_array_randomly(X)
reconstructed = np.array([get_other_file(x) for x in reconstructed])

print('Generating labels...')
original_y, reconstructed_y = np.zeros(original.shape), np.ones(reconstructed.shape)
X = original.tolist() + reconstructed.tolist()
y = original_y.tolist() + reconstructed_y.tolist()

print('Creating train/test split...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

print('Creating validation set...')
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.25, random_state=1, stratify=y_test)
    
print('Making DataFrames...')
    
df_train = pd.DataFrame(X_train, columns=['X'])
df_train['y'] = y_train
df_train['dataset'] = df_train.X.apply(categorize)

df_test = pd.DataFrame(X_test, columns=['X'])
df_test['y'] = y_test
df_test['dataset'] = df_test.X.apply(categorize)

df_val = pd.DataFrame(X_val, columns=['X'])
df_val['y'] = y_val
df_val['dataset'] = df_val.X.apply(categorize)

print(df_train.shape, df_train.dataset.value_counts())
print(df_test.shape, df_test.dataset.value_counts())
print(df_val.shape, df_val.dataset.value_counts())

df_train.to_csv('data/splits/train.csv', index=False)
df_test.to_csv('data/splits/test.csv', index=False)
df_val.to_csv('data/splits/val.csv', index=False)