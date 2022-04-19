# classification-T108360150


1.Kaggle比賽簡介：
    台語語音辨認 using Recurrent Neural Networks-based End-to-End Approach

規格：
    單人女聲聲音（高雄腔）
    輸入：台語語音音檔（格式：wav檔, 22 kHz, mono, 32 bits）
    輸出：台羅拼音（依教育部標準）

重點：
    先做對參考腳本中，最簡單的一層LSTM架構
    再用Tensorflow語法，修改類神經網路設定，看不同網路架構的效果如何

2.檔案說明：

"language.ipynb"是訓練測試的ipynb檔

"IMG"中是報告需要的所有圖片

"keras2.h5"是訓練後的模型

"predict2.csv"預測結果

訓練音檔：train目錄（請忽略._*.wav，這是mac電腦的隱藏暫存檔）

測試音檔：test-shuf目錄（請忽略._*.wav，這是mac電腦的隱藏暫存檔）

字典：lexicon.txt（教育部部定台羅拼音，子音，母音）

訓練資料列表：train-toneless-update.csv（id, text）

答案範例檔：sample.csv（id, text）


3.程式：

我使用 jupyter notebook 編譯程式
![image](https://github.com/108360150-Qiuchonghao/MachineLearning_Houseprice/blob/main/IMG/jupyter1.jpg)


# Data Preprocessing：

In [0]:
        #將train-toneless-update.csv，分割成三千多個txt檔案
        import csv
        with open("/home/qiu/study_college/machine_learning/language/machine-learningntut-2021-autumn-asr/ML@NTUT-2021-Autumn-ASR/train-toneless_update.csv") as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                a = row[0]
                b = row[1]
                f = open("/home/qiu/study_college/machine_learning/language/machine-learningntut-2021-autumn-asr/ML@NTUT-2021-Autumn-ASR/train/txt/"+a+'.txt','w')
                f.write(b)
                f.close()


In [1]:
      
        #import 所有所需要的API
        # -*- coding: utf-8 -*-
        from keras.models import Model
        from keras.layers import Input, Activation, Conv1D, Lambda, Add, Multiply, BatchNormalization
        from tensorflow.keras.optimizers import Adam, SGD
        from keras import backend as K
        from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        %matplotlib inline
        import random
        import pickle
        import glob
        from tqdm import tqdm
        import os

        from python_speech_features import mfcc
        import scipy.io.wavfile as wav
        import librosa
        from IPython.display import Audio


In [2]:        

        # 定義讀取音頻檔案，讀取文字檔案方法
            def get_wav_files(wav_path):
                wav_files = []
                for (dirpath, dirnames, filenames) in os.walk(wav_path):
                    for filename in filenames:
                        if filename.endswith(".wav") or filename.endswith(".WAV"):
                            filename_path = os.path.join(dirpath, filename)
                            wav_files.append(filename_path)
                return wav_files    

        def get_tran_texts(wav_files, tran_path):
            tran_texts = []
            for wav_file in wav_files:
                basename = os.path.basename(wav_file)
                x = os.path.splitext(basename)[0]
                tran_file = os.path.join(tran_path,x+ '.txt') 
                if os.path.exists(tran_file) is False:
                    return None
                fd = open(tran_file, 'r')
                text = fd.readline()
                tran_texts.append(text.split('\n')[0])
                fd.close()
            return tran_texts  



In [3]:     

        # 定義處理音頻方法
        mfcc_dim =13

        def load_and_trim(path):
            audio, sr =librosa.load(path)
            energy = librosa.feature.rms(audio)
            frames = np.nonzero(energy >= np.max(energy) /5)
            indices = librosa.core.frames_to_samples(frames)[1]
            audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]
            return audio,sr
Liborsa : 專門用來做語音訊號處理的函式。
• 首先讀取wav.mp3之類的檔案
• audio : 音頻的信號值
• sr : 採樣率
• energy : 計算每幀的均方根值
• 整個過程是在做把過小聲的音檔除掉。

In [4]:      

        features = []
        for i in tqdm(range(len(wav_files))):
            path = wav_files[i]
            audio, sr = load_and_trim(path)
            features.append(mfcc(audio, sr, numcep = mfcc_dim, nfft = 551))  
讀取完畢：            
![imge](https://github.com/108360150-Qiuchonghao/MachineLearning_Voice/blob/main/IMG/1.png)

In [5]:      
        texts= get_tran_texts(wav_files, tran_path)
        chars = {}

        for text in texts:
            for c in text:
                chars[c] = chars.get(c, 0) + 1

        chars = sorted(chars.items(), key=lambda x: x[1], reverse=True)
        chars = [char[0] for char in chars]
        print(len(chars), chars[:100])

        char2id = {c: i for i, c in enumerate(chars)}
        id2char = {i: c for i, c in enumerate(chars)}
![imge](https://github.com/108360150-Qiuchonghao/MachineLearning_Voice/blob/main/IMG/2.png)

# Training PART：
In [6]:      

        data_index = np.arange(total)
        np.random.shuffle(data_index)
        train_size = int(0.9 * total)
        test_size = total - train_size
        train_index = data_index[:train_size]
        test_index = data_index[train_size:]

        X_train = [features[i] for i in train_index]
        Y_train = [texts[i] for i in train_index]
        X_test = [features[i] for i in test_index]
        Y_test = [texts[i] for i in test_index]

        batch_size = 16
            
        def batch_generator(x, y, batch_size=batch_size):  
            offset = 0
            while True:
                offset += batch_size
                
                if offset == batch_size or offset >= len(x):
                    data_index = np.arange(len(x))
                    np.random.shuffle(data_index)
                    x = [x[i] for i in data_index]
                    y = [y[i] for i in data_index]
                    offset = batch_size
                    
                X_data = x[offset - batch_size: offset]
                Y_data = y[offset - batch_size: offset]
                
                X_maxlen = max([X_data[i].shape[0] for i in range(batch_size)])
                Y_maxlen = max([len(Y_data[i]) for i in range(batch_size)])
                
                X_batch = np.zeros([batch_size, X_maxlen, mfcc_dim])
                Y_batch = np.ones([batch_size, Y_maxlen]) * len(char2id)
                X_length = np.zeros([batch_size, 1], dtype='int32')
                Y_length = np.zeros([batch_size, 1], dtype='int32')
                
                for i in range(batch_size):
                    X_length[i, 0] = X_data[i].shape[0]
                    X_batch[i, :X_length[i, 0], :] = X_data[i]
                    
                    Y_length[i, 0] = len(Y_data[i])
                    Y_batch[i, :Y_length[i, 0]] = [char2id[c] for c in Y_data[i]]
                
                inputs = {'X': X_batch, 'Y': Y_batch, 'X_length': X_length, 'Y_length': Y_length}
                outputs = {'ctc': np.zeros([batch_size])}
                
                yield (inputs, outputs)

In [7]:    

        epochs = 40
        num_blocks = 3
        filters = 128

        X = Input(shape=(None, mfcc_dim,), dtype='float32', name='X')
        Y = Input(shape=(None,), dtype='float32', name='Y')
        X_length = Input(shape=(1,), dtype='int32', name='X_length')
        Y_length = Input(shape=(1,), dtype='int32', name='Y_length')

        def conv1d(inputs, filters, kernel_size, dilation_rate):
            return Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='causal', activation=None, dilation_rate=dilation_rate)(inputs)

        def batchnorm(inputs):
            return BatchNormalization()(inputs)

        def activation(inputs, activation):
            return Activation(activation)(inputs)

        def res_block(inputs, filters, kernel_size, dilation_rate):
            hf = activation(batchnorm(conv1d(inputs, filters, kernel_size, dilation_rate)), 'tanh')
            hg = activation(batchnorm(conv1d(inputs, filters, kernel_size, dilation_rate)), 'sigmoid')
            h0 = Multiply()([hf, hg])
            
            ha = activation(batchnorm(conv1d(h0, filters, 1, 1)), 'tanh')
            hs = activation(batchnorm(conv1d(h0, filters, 1, 1)), 'tanh')
            return Add()([ha, inputs]), hs

        h0 = activation(batchnorm(conv1d(X, filters, 1, 1)), 'tanh')
        shortcut = []
        for i in range(num_blocks):
            for r in [1, 2, 4, 8, 16]:
                h0, s = res_block(h0, filters, 7, r)
                shortcut.append(s)

        h1 = activation(Add()(shortcut), 'relu')
        h1 = activation(batchnorm(conv1d(h1, filters, 1, 1)), 'relu')
        Y_pred = activation(batchnorm(conv1d(h1, len(char2id) + 1, 1, 1)), 'softmax')
        sub_model = Model(inputs=X, outputs=Y_pred)

        def calc_ctc_loss(args):
            y, yp, ypl, yl = args
            return K.ctc_batch_cost(y, yp, ypl, yl)

        ctc_loss = Lambda(calc_ctc_loss, output_shape=(1,), name='ctc')([Y, Y_pred, X_length, Y_length])
        model = Model(inputs=[X, Y, X_length, Y_length], outputs=ctc_loss)
        optimizer = SGD(lr=0.02, momentum=0.9, nesterov=True, clipnorm=5)
        model.compile(loss={'ctc': lambda ctc_true, ctc_pred: ctc_pred}, optimizer=optimizer)

        checkpointer = ModelCheckpoint(filepath='keras3.h5', verbose=0)
        lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=1, min_lr=0.000)

        history = model.fit_generator(
            generator=batch_generator(X_train, Y_train), 
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs, 
            validation_data=batch_generator(X_test, Y_test), 
            validation_steps=len(X_test) // batch_size, 
            callbacks=[checkpointer, lr_decay])
        sub_model.save('allen40.h5')
ＣＴＣ：為損失函數的聲學模型訓練，是一種完全端對端的聲學模型訓練，不
需要預先對數據做對齊，只需要一格輸入序列和一個輸出序列可以訓練。這次
作業的音檔的長度不一，而此方法可以不取時間，只取特徵。

In [8]:         

        #透過趨勢圖來觀察訓練與驗證的走向 ，是否有overfitting。
        with open('dictionary.pkl', 'wb') as fw:
            pickle.dump([char2id, id2char, mfcc_mean, mfcc_std], fw)
        
        train_loss = history.history['loss']
        valid_loss = history.history['val_loss']
        plt.plot(np.linspace(1, epochs, epochs), train_loss, label='train')
        plt.plot(np.linspace(1, epochs, epochs), valid_loss, label='valid')
        plt.legend(loc='upper right')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

Epochs1-50的訓練結果如圖所示：
![imge](https://github.com/108360150-Qiuchonghao/MachineLearning_Voice/blob/main/IMG/6.png)
![imge](https://github.com/108360150-Qiuchonghao/MachineLearning_Voice/blob/main/IMG/3.png)

# Test Part：
In [9]:   

        # 預測與比對
        #將預測結果存到csv檔案中
        # -*- coding: utf-8 -*-

        from keras.models import load_model
        from keras import backend as K
        import numpy as np
        import librosa
        from python_speech_features import mfcc
        import pickle
        import glob

        wavs = glob.glob('/home/qiu/study_college/machine_learning/language/machine-learningntut-2021-autumn-asr/ML@NTUT-2021-Autumn-ASR/test-shuf/*.wav')
        with open('dictionary.pkl', 'rb') as fr:
            [char2id, id2char, mfcc_mean, mfcc_std] = pickle.load(fr)

        mfcc_dim = 13
        # compile=False為了避免報錯
        model = load_model('keras3.h5', compile=False)

        with open('predict3.csv','w') as f:
            f.write('id,text\n')
            for j in range(1,347):
                path1=['/home/qiu/study_college/machine_learning/language/machine-learningntut-2021-autumn-asr/ML@NTUT-2021-Autumn-ASR/test-shuf/',str(j),'.wav']
                path=''.join(path1)
                #print(path)
                audio, sr=load_and_trim(path)
                X_data = mfcc(audio, sr, numcep=mfcc_dim, nfft=551，highfreq=8000)
                X_data = (X_data - mfcc_mean) / (mfcc_std + 1e-14)
                #print(X_data.shape)
                #預測：
                pred = model.predict(np.expand_dims(X_data, axis=0))
                pred_ids = K.eval(K.ctc_decode(pred, [X_data.shape[0]], greedy=False, beam_width=10, top_paths=1)[0][0])
                pred_ids = pred_ids.flatten().tolist()
                #去處預測出的數字中為“-1”的值
                #有－１的原因：因為當讀取的音檔內容是２６個字母＋空格（共２７）以外的東西
                while -1 in pred_ids:
                    pred_ids.remove(-1)
                f.write(str(j) +  ',' + (''.join([id2char[i] for i in pred_ids]))+'\n')
![imge](https://github.com/108360150-Qiuchonghao/MachineLearning_Voice/blob/main/IMG/4.png)

4.心得：
這次語音識別用到了mfcc的知識，mfcc是語音特徵參數（Mel-scale Frequency Cepstral Coefficients）。
我把自己的預測結果與學長放在ppt中的部分做比較發現，常常有一些詞語只是在最後末尾多了n或m，應該是對音頻的處理沒有處理好，導致訓練出來效果不佳，可以在這方面做改進。
![imge](https://github.com/108360150-Qiuchonghao/MachineLearning_Voice/blob/main/IMG/5.png)