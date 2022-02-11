# *Face Classification based on Python*

## *Background*

*This is the summer semester homework for THU EE freshman in 2021.8, the purpose of which, interpreted by the consoulors, is to get familiar with the basic grammar of Python.
Therefore, the accuracy of the gender classification is not what to be concerned with and all methods and algorithms, no matter how basic or advanced they might be, are encouraged.*  

## *Intension*

- *The very primary intension of creating this repository is to check out whether I've learnt the basic using skills of git and Github rather than others.* 

- *Meanwhile, the repository might be a help for the junior if Python lectures is to be remained during the coming summer semester.*

## *Branches*

*There are two branches separated base on different environments.*

### *master*

- *Files*

  - `cnn.py` (Runnable)*trains the Convolutional Neural Networks and print the accuracy of the face classification*;
  - `Common` *contains the basic classes used in* `cnn.py`, *including dataloader, callback, etc.* 
  - `Dataset` *contains the image data used for training.* `Image` *contains the original forms and* `Label` *the correct gender*. *To improve accuracy, they are pre-treated and are saved in folder* `Image-haired` *and* `Image-haired_colored`.

- *How to use*

  *Clone the branch to your PC and open the entire folder with VS code, and the py file* `cnn.py` *can be ran directly*.

- *Environment*

  `Tensorflow 2.0` 

- *Algorithm*

  CNN

### *base*

- *Files*

  - `KNN.py` (Runnable)*print the accuracy of the face classification based on KNN*;
  - `KNN-SKLearn.py` (Runnable)*print the accuracy of the face classification based on SKLearn KNN which is faster*;
  - `LogisticRegression-SKLearn.py` (Runnable)*print the accuracy of the face classification based on LR, which is more accurate than KNN*;
  - `Common` *contains the basic classes used in* `cnn.py`, *including dataloader, classifier, etc.* 
  - `Dataset` *contains the image data used for training.* `Image` *contains the original forms and* `Label` *the correct gender*. *To improve accuracy, they are pre-treated and are saved in folder* `Image-haired` *and* `Image-haired_colored`.

- *How to use*

  *Clone the branch to your PC and open the entire folder with VS code, and the py files* `KNN.py`, `KNN-SKLearn.py` *and* `LogisticRegression-SKLearn.py` *can be ran directly*.

- *Environment*

  `Python 3` 

- *Algorithm*

  - *KNN*;
  - *Logistic Regression*.

## *Results*

*Results are based entirely on the accuracy of classification*:

- *95% for CNN
- *80% for KNN
- *85~90% for LR.

