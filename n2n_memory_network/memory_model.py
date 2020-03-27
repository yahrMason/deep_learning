import pickle
import numpy as np

'''
----------------
DATA PREPARATION 
----------------
'''

# train test split
with open('data/train_qa.txt','rb') as f:
    train_data = pickle.load(f) 
with open('data/test_qa.txt','rb') as f:
    test_data = pickle.load(f)


# Create Vocabulary
all_data = test_data + train_data

# add answers
vocab = set(['no','yes']) 
# add story and query text
for story, question , answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))
   
# create vocab length variable
# need to add one for keras pad sequence (0 is a placeholder)
vocab_size = len(vocab) + 1

# Longest Story
max_story_len = max([len(data[0]) for data in all_data])
# Longest Question
max_question_len = max([len(data[1]) for data in all_data])

# Vectorize the Data
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from network_functions import vectorize_stories

# tokenize the words
tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)


'''
-----------------------
Saving Supporting Files
-----------------------
'''
with open('vocabulary.txt','w') as f:
   f.write(str(vocab))
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('parameters.txt','w') as f:
    f.write(str({'max_story_len':max_story_len, 'max_question_len':max_question_len}))

# Vectorize Training Data
inputs_train, queries_train, answers_train = vectorize_stories(train_data, 
                                                               word_index=tokenizer.word_index, 
                                                               max_story_len=max_story_len,
                                                               max_question_len=max_question_len)
# Vectorize Testing Data
inputs_test, queries_test, answers_test = vectorize_stories(test_data,
                                                            word_index=tokenizer.word_index, 
                                                            max_story_len=max_story_len,
                                                            max_question_len=max_question_len)

'''
------------------
Building the Model
------------------
'''
from keras.models import Sequential,Model
from keras.layers.embeddings import Embedding
from keras.layers import Input,Activation,Dense,Permute,Dropout,add,dot,concatenate,LSTM

# Placeholder for Inputs
input_sequence = Input((max_story_len,))
question = Input((max_question_len,))

# --------
# Encoders
# --------

### INPUT ENCODER m
# Input gets embedded to a sequence of vectors
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,output_dim=64))
input_encoder_m.add(Dropout(0.3))
# output: (samples, story_maxlen, embedding_dim)

### INPUT ENCODER c
# embed the input into a sequence of vectors of size query_maxlen
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3))
# output: (samples, story_maxlen, query_maxlen)

### QUESTION ENCODER
# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,output_dim=64,input_length=max_question_len))
question_encoder.add(Dropout(0.3))
# output: (samples, query_maxlen, embedding_dim)

# Encode the Sequences
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

## Use dot product to compute the match between first input vector seq and the query
# shape: `(samples, story_maxlen, query_maxlen)`
match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)

# add the match matrix with the second input vector sequence
response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)


# concatenate the match matrix with the question vector sequence
answer = concatenate([response, question_encoded])
# Reduce with RNN (LSTM)
answer = LSTM(32)(answer)  # (samples, 32)
# Regularization with Dropout
answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
# output a probability distribution over the vocabulary
answer = Activation('softmax')(answer)

# build the final model
model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# train
num_epoch = 150
history = model.fit([inputs_train, queries_train], answers_train,batch_size=16,epochs=num_epoch,validation_data=([inputs_test, queries_test], answers_test))

# plot accuracy history
import matplotlib.pyplot as plt
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


'''
----------------
Saving the Model
----------------
'''
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Saving the Model weights
weights_filename = f'chatbot_{num_epoch}epochs.h5'
model.save(weights_filename)
print("Saved model to disk")