import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import time
import azure.cognitiveservices.speech as speechsdk
import glob
import os
from pydub import AudioSegment
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import wave
from google.cloud import storage
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "intense-emblem-252302-6ee085971da5.json"
filepath = "C:/Users/gayat/PycharmProjects/untitled2/data/"     #Input audio file path
bucketname = "callsaudiofiles0620" #Name of the bucket created in the step before


def stereo_to_mono(audio_file_name):
    sound = AudioSegment.from_wav(audio_file_name)
    sound = sound.set_channels(1)
    sound.export(audio_file_name, format="wav")

def frame_rate_channel(audio_file_name):
    with wave.open(audio_file_name, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        channels = wave_file.getnchannels()
        return frame_rate,channels

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.delete()


def google_transcribe(audio_file_name):
    file_name = filepath + audio_file_name

    # The name of the audio file to transcribe

    frame_rate, channels = frame_rate_channel(file_name)

    if channels > 1:
        stereo_to_mono(file_name)

    bucket_name = bucketname
    source_file_name = filepath + audio_file_name
    destination_blob_name = audio_file_name

    upload_blob(bucket_name, source_file_name, destination_blob_name)

    gcs_uri = 'gs://' + bucketname + '/' + audio_file_name
    transcript = ''

    client = speech.SpeechClient()
    audio = types.RecognitionAudio(uri=gcs_uri)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=frame_rate,
        language_code='en-US')

    # Detects speech in the audio file
    operation = client.long_running_recognize(config, audio)
    response = operation.result(timeout=10000)

    for result in response.results:
        transcript += result.alternatives[0].transcript

    delete_blob(bucket_name, destination_blob_name)
    return transcript


transcript = []
for audio_file_name in os.listdir(filepath):
    transcript.append(google_transcribe(audio_file_name))


#Train Set
Train_data=pd.read_csv(r"C:\Users\gayat\PycharmProjects\untitled2\Corpus.csv")
Train_data.drop(["breakdown", "feedback", "vehicle_quality", "test_drive_request", "new_vehicle_purchase_enquiry"], 1,inplace=True)

Train_data['sentence'].dropna(inplace=True)
Train_data['sentence'] = [entry.lower() for entry in Train_data['sentence']]
Train_data['sentence'] = [word_tokenize(entry) for entry in Train_data['sentence']]
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(Train_data['sentence']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    Train_data.loc[index,'text_final'] = str(Final_words)

Train_X=Train_data["text_final"]
Train_Y=Train_data["Label"]
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Train_data['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)

#Test Set

Test_X=pd.DataFrame({"sentence":transcript})
Test_Y=[3,1,2,4,5]
Test_X['sentence'] = [entry.lower() for entry in Test_X['sentence']]
Test_X['sentence']= [word_tokenize(entry) for entry in Test_X['sentence']]
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(Test_X['sentence']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    Test_X.loc[index,'text_final'] = str(Final_words)


Test_X_Tfidf = Tfidf_vect.transform(Test_X["text_final"])
Test_Y = Encoder.fit_transform(Test_Y)



#--------------------------------------------------------------
# Predction Models
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
predictions_NB = Naive.predict(Test_X_Tfidf)
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

# SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

#-------------------------------------------
# 0-breakdown
# 1-feedback
# 2-new vehicle purchase
# 3-test_drive request
# 4-vehicle quality
#--------------------------------------------
from keras import Sequential
from keras.layers import Dense
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


One_hot_test=keras.utils.to_categorical(Test_Y , num_classes=5)
one_hot_train = keras.utils.to_categorical(Train_Y, num_classes=5)

# model = Sequential()
# model.add(Dense(350, activation='tanh',input_dim=650))
# model.add(Dense(250, activation='tanh'))
# model.add(Dense(200, activation='tanh'))
# model.add(Dense(100, activation='tanh'))
# model.add(Dense(5, activation='softmax'))
# model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#
# filepath="Final_model.hdf5"
# checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# model.fit(Train_X_Tfidf, batch_size=32, y=one_hot_train, verbose=1,shuffle=True, epochs=50, callbacks=[checkpointer])
# model.evaluate(Test_X_Tfidf,One_hot_test)
# print(model.summary())
# #model.save('my_mode60.h5')

model2 = load_model('my_mode60.h5')
model2.evaluate(Test_X_Tfidf,One_hot_test)
pred=model2.predict(Test_X_Tfidf)

index={0:"Breakdown", 1:"Feedback",2:"New Vehicle Purchase", 3:"Test Drive Request", 4:"Vehicle Quality"}
predictions=[]
Test_actual=[]
for a in pred:
    predictions.append(index[np.argmax(a)])

for b in Test_Y:
    Test_actual.append(index[b])


print("predictions ---> ",predictions)
print("Actual values ---> ",Test_actual)
#Final_Result=pd.DataFrame({"file":Test_file_names,"Class":predictions })
#Final_Result.to_csv("Not_so_bayesic_I-0SAJ4.csv",header=True)
