import librosa
import soundfile
import glob,os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40),axis=1)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate),axis=1)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate),axis=1)
            result=np.hstack((result, mel))
    return result

emotions={
  '01':'neutral',
  '02':'calm', 
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
observed_emotions=['angry', 'happy', 'fearful', 'disgust','sad','surprised']

def load_data(test_size):
    x,y=[],[]
    count=0
    for file in glob.glob(r"C:\Users\Deepak\Desktop\speech-emotion-recognition-ravdess-data\Actor_*\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        if emotion in observed_emotions:
            count+=1
            print("file: ",count,"	","emotion: ",emotion)
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

x_train,x_test,y_train,y_test=load_data(test_size=0.1)
print((x_train.shape[0], x_test.shape[0]))
print(f'Features extracted: {x_train.shape[1]}')
model=MLPClassifier(alpha=0.01, batch_size=10, hidden_layer_sizes=(1500,),activation='logistic',learning_rate='adaptive',max_iter=15000)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(x_train)
print(x_test)
print(y_test)
print(y_pred)

accuracy=accuracy_score(y_pred,y_test)
print("Accuracy: {:.2f}%".format(accuracy*100))
cm=confusion_matrix(y_test,y_pred)

co=[]
wr=[]

k=0
u=0
for i in range(6):
  for j in range(6):
    if i==j:
      co.append(cm[i][j])
      u=cm[i][j]
    else:
      k=sum(cm[i])-u
  wr.append(k)
dig = np.arange(6)  
plt.title("emotion classification metrics")
sns.heatmap(cm,annot=True,fmt="d",cmap='coolwarm',linewidth=.9)
plt.xticks(dig,('angry', 'happy', 'fearful', 'disgust','sad','surprised'))
plt.yticks(dig,('angry', 'happy', 'fearful', 'disgust','sad','surprised'))
plt.xlabel("emotions")
plt.ylabel("correctly classified data")

ind = np.arange(6)  
width = 0.4 
fig = plt.subplots(figsize =(10, 7))
p1 = plt.bar(ind, co, width)
p2 = plt.bar(ind, wr, width,bottom = co)
plt.xlabel("emotions")
plt.ylabel("correctly classified data")
plt.title("emotion classification")
plt.xticks(ind, ('angry', 'happy', 'fearful', 'disgust','sad','surprised'))
plt.yticks(np.arange(0, 30, 7))
plt.legend((p1[0], p2[0]), ('Correctly classified data','Miss classified data'))
plt.show()



