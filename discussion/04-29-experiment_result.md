
# 本次测试结果使用数据集情况（64个城市分类）

- 训练集：152*60+140*4=9680张
- 验证集：13*60+4+4+8+8=804张

### inceptionV3
epoch = 150  
batch size = 32  
learning rate = 0.001  
optimizer = Adam  

time = 85s / epoch
validation loss = 0.179210  
validation accuracy = 0.965147


### resnet152
epoch = 100  
batch size = 28  
learning rate = 0.001  
optimizer = Adam  

time = 264s / epoch  
validation loss = 0.131161  
validation accuracy = 0.962687  



### inceptionV4
epoch = 100  
batch size = 32  
learning rate = 0.001  
optimizer = Adam  

time = 192s / epoch
validation loss = 0.067062  
validation accuracy = 0.986318  



### inception-resnet  
epoch = 50  
batch size = 32  
learning rate = 0.001  
optimizer = Adam  

time = 180s / epoch  
validation loss = 0.006285  
validation accuracy = 0.997512  



# 手机号识别进展情况

- pytorch版本：借鉴论文中的公开代码，目前初步可以进行用来训练自己的数据，测试结果
- keras版本： 借鉴example中提供的cnn-rnn-ctc模型修改成适合我们数据的代码，本周将全部调整完毕，并进行训练


# 目前面临的问题

- 城市数据需要扩充到多个市区？收集？
- 快递单系统直接输入整张图片，给出省份城市区域的判别，我们如何精确定位省份小区域？







