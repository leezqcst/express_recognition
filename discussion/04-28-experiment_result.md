
#### inceptionV4:  
learning_rate=0.001  
epoch=50
 - validation loss: 0.132252  
 - validation accuracy: 0.960000  

#### inception-resnet2:  
learning_rate=0.001  
epoch=50
  - validation loss: 0.029718  
  - validation accuracy: 0.991667  

使用了两个最新的网络结构，极大地提高了模型准确性，取得了很好的效果，之前最好的结果是由151层的resnet得到的，验证准确率也只有91左右。inceptionV4和inception-resnet2都引入了残差层来增加收敛速度，并且由于残差层的引入可以再加深网络结构，所以可以得到更好的精度。具体可以参考[论文](https://arxiv.org/pdf/1602.07261.pdf)

ps:  
1.昨天跑了很久的程序今天起来看发现验证集根本就没有收敛，但是在训练集上却收敛了，花了大量的时间时候得到了一个废模型，然后今天我将训练的程序修改了一下，使得每一个epoch跑完之后都进行一次验证，这样可以检查模型是否是有问题的。  

2.关于网络在验证集上不收敛修改了两个地方，修改了代码中的batch_normalization这个部分，修改了参数的初始化，参考了torch里面对于inceptionv3的初始化方法，查阅了一些博客，发现初始化是很重要的，会严重影响模型结果。

3.不再使用SGD优化方法，换成了Adam，所以之前的模型在这个方法下可能也能收敛到一个比之前好的结果，这个之后会重新验证

#### feature extraction
对于这个部分，通过vgg，inceptionv3和resnet151分别对训练接和验证集都提取了特征，然后将特征concat在一起，最后通过两层的全连接网络进行预测。试验之后发现网络在训练集上就很难收敛，而之前参考的那篇文章做的猫狗分类，他在提取特征之后，在没有训练全连接层的时候已经实现了90%的准确率，而我们的却无法收敛。原因应该是网络预训练的权重实在imagenet上面得到的，而imagenet是包含猫和狗的分类，而我们的手写汉字并不在其中，所以得到的效果不好。

我们可以用vgg，inceptionv3和resnet151在我们自己的数据集上训练，然后将权重保存下来作为一个预训练的权重，然后用相同的办法提取特征，这相当于是集成三个网络对这个问题进行预测，准确率应该能够上升。

目前incepitonv4和inception-resnet2的效果都很好，但是这只是在60分类的情况下，按道理分类越多，效果就会降低，所以我们最后要做的300分类肯定会影响精度，所以这可以作为一种新的想法去增强我们的分类效果。

- - -
inceptionv4的代码在cnn_pytorch里面的model.py中，inception-resent的代码在inceptionresnet.py里面，特征提取的代码在feature文件夹里面。
