# word2vec

C02KT24MFFT1:word2vec abhkumar$ history | grep train_word2vec_model
 6616  [2015-12-13 08:20:29] vim train_word2vec_model.py
 6619  [2015-12-13 08:21:05] python train_word2vec_model.py wiki.en.text wiki.en.word2vec.model
 6621  [2015-12-14 05:45:17] git add train_word2vec_model.py 
 6628  [2015-12-19 16:09:51] vim train_word2vec_model.py
 7127  [2016-02-12 18:27:09] history | grep train_word2vec_model
C02KT24MFFT1:word2vec abhkumar$ wc -l wiki.en.text
 3933461 wiki.en.text
C02KT24MFFT1:word2vec abhkumar$ history | grep process_wiki
 6610  [2015-12-13 03:20:53] vim process_wiki.py
 6613  [2015-12-13 03:22:22] python process_wiki.py enwiki-20151102-pages-articles.xml.bz2 wiki.en.text
 6620  [2015-12-14 05:45:12] git add process_wiki.py 
 6933  [2015-12-20 06:29:13] git add process_wiki.py 
 7129  [2016-02-12 18:34:52] history | grep process_wiki
C02KT24MFFT1:word2vec abhkumar$ history | grep Word2VecUtility
 6914  [2015-12-20 03:54:52] cp ../crowdflower_word2vec/KaggleWord2VecUtility.py ./
 6915  [2015-12-20 03:54:55] vim KaggleWord2VecUtility.py 
 6916  [2015-12-20 05:06:20] vim KaggleWord2VecUtility.py 
 6918  [2015-12-20 06:22:24] vim KaggleWord2VecUtility.py 
 6919  [2015-12-20 06:22:41] git add KaggleWord2VecUtility.py
 6920  [2015-12-20 06:27:33] git remove KaggleWord2VecUtility.py
 6921  [2015-12-20 06:27:40] git rm KaggleWord2VecUtility.py
 6922  [2015-12-20 06:27:51] git rm --cached KaggleWord2VecUtility.py
 6924  [2015-12-20 06:28:08] mv KaggleWord2VecUtility.py Word2VecUtility.py
 6926  [2015-12-20 06:28:22] mv KaggleWord2VecUtility.pyc Word2VecUtility.pyc
 6928  [2015-12-20 06:28:38] git add Word2VecUtility.py
 6929  [2015-12-20 06:28:40] git add Word2VecUtility.pyc 
 7130  [2016-02-12 18:35:49] history | grep Word2VecUtility
C02KT24MFFT1:word2vec abhkumar$ history | grep QuerySimilarity
 6856  [2015-12-19 20:21:29] vim QuerySimilarity.py
 6931  [2015-12-20 06:28:56] git add QuerySimilarity.py
 6932  [2015-12-20 06:28:59] git add QuerySimilarity.pyc 
 7131  [2016-02-12 18:36:36] history | grep QuerySimilarity
C02KT24MFFT1:word2vec abhkumar$ history | grep ParseSOpostSmall.py
 6846  [2015-12-20 12:08:38] vim ParseSOpostSmall.py 
 6848  [2015-12-20 12:10:35] vim ParseSOpostSmall.py 
 6849  [2015-12-20 12:11:14] python ParseSOpostSmall.py SOposts100000.xml > SOposts100000.txt
 6851  [2015-12-20 12:12:31] vim ParseSOpostSmall.py 
 6852  [2015-12-20 12:13:03] python ParseSOpostSmall.py SOposts100000.xml > SOposts100000.txt
 6913  [2015-12-20 03:51:30] vim ParseSOpostSmall.py
 6934  [2015-12-20 06:29:21] git ParseSOpostSmall.py 
 6935  [2015-12-20 06:29:24] git add ParseSOpostSmall.py 
 7132  [2016-02-12 19:13:53] history | grep ParseSOpostSmall.py



To run in iPython just follow def findSimilarity(trainFile, testQ) : in QuerySimilarity
