# Pytorch-ResNet
学习Pytorch以及神经网络过程中搭建的第一个网络
日期：2021年1月26日~1月30日

----
在本次学习中，我实现了ResNet18,ResNet34,ResNet50,ResNet101,ResNet152五种不同层数的ResNet(后三者考虑了Bottleneck）,并将其第一个卷积层的卷积核大小改为3x3,由此来适应CIFAR-10数据集。
在train.py中，我实现了对网络模型的调用以及训练，并在过程中记录了相关的训练和验证信息，并保存了训练过程中以及完成时的网络模型。同时，我使用tensorboardX来对于网络和训练过程进行可视化。
模型输出在model文件夹，记录文件和tensorboard记录则位于record文件夹。由于仓库大小限制，我没有上传训练过程中记录的模型。
我也将自己所训练的ResNet18, ResNet50的最终模型和训练记录上传上来，可供参考。

测试结果如下：
![](test/result/result1.png)

由于是初次尝试，有很多问题还没有解决，恳请大家指教！
