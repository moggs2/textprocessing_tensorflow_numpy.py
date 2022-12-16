import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import numpy

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "aclImdb/train",
    labels="inferred",
    label_mode="binary",
    class_names=['neg', 'pos'],
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=37,
    )

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "aclImdb/train",
    labels="inferred",
    label_mode="binary",
    class_names=['neg', 'pos'],
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=37,
)

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "aclImdb/test", batch_size=32
)

from tensorflow.keras.layers import TextVectorization
import string
import re

def transformdata(inputdata="",wordvector="none"):

 textview=tfds.as_numpy(inputdata)
 
 i=0
 j=0
 k=0
 newtextarray = []
 newlabelarray = []
 for text in textview:
   #text=np.array(text)
   textshape=text[0].shape
   elements=textshape[0]
   print(elements)
   j=0
  
   while j < elements:
    newtextarray.append(text[0][j])
    newlabelarray.append(text[1][j][0])
    j=j+1
    k=k+1
  
   #print(i)
   #print(text)
   i=i+1
   print(text[0][0])
   print(text[1][0][0])
   #if i == 2:
   #     break
       
 i = 0
 elements=len(newtextarray)
 print(elements)
 textmodarray=[]
 while i < elements:
   text2mod=str(newtextarray[i])
   text2mod=text2mod.lower()
   text2mod=text2mod.replace("<br />", " ")
   #for character in string.punctuation:
   #  test2mod = text2mod.replace(character, '')
   text2mod=text2mod.translate(str.maketrans('', '', string.punctuation))
   textmodarray.append(text2mod)
   i = i+1
 #print(textmodarray)
 print(newlabelarray)
 #textmodarray=np.expand_dims(np.array(textmodarray), axis=1)
 textmodarray=np.array(textmodarray)
 print(textmodarray.shape)
 #textmodarray=vectorize_layer(textmodarray)
 newdataset = tf.data.Dataset.from_tensor_slices((textmodarray,np.array(newlabelarray)))	
 if wordvector=="none":
    return newdataset
 else:
  def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    label = tf.expand_dims(label, -1)
    return vectorize_layer(text), label
  
  print(textmodarray[9:10])
  print(vectorize_layer(textmodarray[9:10]))
  
  int_ds = newdataset.map(vectorize_text)
  return int_ds

trainingds=transformdata(raw_train_ds)

vectorize_layer = TextVectorization(
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=250,
)

print(trainingds)

ds=trainingds.map(lambda data,label: data) 

vectorize_layer.adapt(ds)

trainingdsint=transformdata(raw_train_ds,vectorize_layer)
valdsint=transformdata(raw_val_ds,vectorize_layer)
print(vectorize_layer.get_vocabulary())

model = tf.keras.Sequential([
tf.keras.layers.Embedding(10001, 32),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(32),
tf.keras.layers.GlobalAveragePooling1D(),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(10),
tf.keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  #loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  loss=tf.losses.BinaryCrossentropy(from_logits=False),
  #metrics=['accuracy'])
  metrics=tf.metrics.BinaryAccuracy(threshold=0.5))

early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=1000)
best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("mnist_model.tf", save_best_only=True)
learningratecallbackchange=tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.01 * 1.02 ** epoch)

fittingdiagram=model.fit(
  trainingdsint,
  validation_data=valdsint,
  epochs=100,
  callbacks=[early_stopping_callback, learningratecallbackchange])
  #best_checkpoint_callback
