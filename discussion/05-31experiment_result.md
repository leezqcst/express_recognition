
### DataSet(手写):
- Train： 3152 张
- Test：   288 张

### Experiment Result

1. baseline  
  old accuracy: 0.659649(188)  
  new accuracy: 0.670175(191)   

2. convert 0, 1  
  old accuracy: 0.610526(174)  
  new accuracy: 0.631579(180)  

3. 重复数字中间加'abcdefghij'  
  accuracy: 0.392982
  
  
### 低像素手写体省份识别
![1.jpg](http://chuantu.biz/t5/96/1496196826x1822613109.jpg)

#### 快递单上图片识别结果
![2.jpg](http://chuantu.biz/t5/96/1496196920x1822613109.jpg)
![3.jpg](http://chuantu.biz/t5/96/1496196965x1822613109.jpg)
![4.jpg](http://chuantu.biz/t5/96/1496197000x1822613109.jpg)
![5.jpg](http://chuantu.biz/t5/96/1496197034x1822613109.jpg)
![6.jpg](http://chuantu.biz/t5/96/1496197070x1822613109.jpg)




#### 下周计划

- 使用低像素图像测试省、市分类整体识别结果
- 考虑在CNN-RNN-CTC网络中加入attention思想
- 调研论文中类似序列识别的可行方法

