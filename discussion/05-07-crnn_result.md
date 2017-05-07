### 测试CRNN结果

具体所有验证集结果在model/crnn/visualize.ipynb里面

#### 1.多位相同数字检测
![1](http://upload-images.jianshu.io/upload_images/3623720-51ae1aeb3c2f16d0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从这个结果可以看出中间的多位2没有检测出来，被ctc认为是误检测从而将3个2识别为了1个2

也有极少数检测出来了相同数字

![2](http://upload-images.jianshu.io/upload_images/3623720-087f2aac36ef765d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

多位数字能够被检测出来的关键是数字之间的空白能否被检测出来，大多数图片两个数字之间的空白都未能被检测出来

#### 2.数字检测有误

![3](http://upload-images.jianshu.io/upload_images/3623720-9d39b84b78c09da0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

比如这张图片，就将5检测成了3

#### 3.边缘误检测

![4](http://upload-images.jianshu.io/upload_images/3623720-898ca6622e108a3f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这张图片最右边上截出来的的竖检测成了1，这个属于图片截取和预处理的问题

#### 问题来源
问题来源是因为sequence learning处理的序列多数为字母，字母相对于数字重复的概率更小，所以在重复数字上会存在这样的问题
### 初步改进方案

**1.** 增加cnn提取特征的长度，使得提取出来的特征足够多，这样才能够有效的分别空白

**2.** 图片预处理部分
