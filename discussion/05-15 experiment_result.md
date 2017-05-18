### 1. 输出 长度为64
改变cnn，去掉一层max pool，使得输出的长度变为64，希望长度变长之后能够得到更多的有效信息，比如空白，和每个数字

**结果**

效果不好，因为ctc的机制是穷举所有的可能概率，序列太长导致学习过程错误，虽然训练的loss降到很低，但是实质上是错误的预测，在训练过程就是错误的预测

### 2. 改变读入图片大小
之前的图片读入是128x32，现在改为256x32，这样也得到64维的输出长度，训练时间加倍

**结果**

效果同样不好，同样的原因

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/3623720-11f1404370062589.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 3.改变图片大小，同时改变网络
图片输入变成256，同时网络增加一层max pool使得输出序列长度为32

预测数字基本都能识别对，对于打印的数字反而有一些识别不对，可能是因为打印版本的数字不同

### 4.使用GRU
效果比LSTM更好，能够识别一些重复数字，因为GRU比LSTM更新，14年提出的结构，效果更好

**LSTM**  
![1](http://upload-images.jianshu.io/upload_images/3623720-052d62685cf93eb4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**GRU**  
![2](http://upload-images.jianshu.io/upload_images/3623720-c25c7163e1107c6e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### 5.输出概率结果
对于每个位置输出的概率结果，很多位置的概率都接近1，应该是因为训练次数太多

对于每个数字在每个位置的概率，可以看看如何区分多位重复数字

### 6.增加attention

todo
