# handpose

## 基于tensorRT的手部空间姿态检测
项目介绍：在Jetson平台上使用C++基于tensorRT+Qt完成了手部识别模型的部署和可视化。 
	把mediepipe模型(tflite)转为onnx， 修改不兼容的op并简化结构，再转为TRT的engine。 
	以零拷贝方式部署engine，并使用QtDataVisual和openCV实现监测结果的可视化。

>个人收获：学习了网络模型的修改、转换和部署，熟悉了CUDA编程的基本步骤。


mediapipe的模型有很多都是tflite格式的，精度和模型尺寸都不错,但是想跑到jetson上需要配置成tensorrt的engine并使用qt可视化就需要做一些额外的工作
效果：
![h](https://user-images.githubusercontent.com/69743646/194582006-4141d0b6-8d3d-4e7a-a55a-235d2a763be7.gif)
