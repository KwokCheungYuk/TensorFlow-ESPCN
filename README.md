# 基于ESPCN的图像超分辨率重建

### 环境

Windows10  1909

Python 3.7

### 安装依赖库

```
pip install -r requirements.txt
```

### 运行程序

```
py main.py 
```

### PS

本项目目前包含了2倍、3倍、5倍、7倍的重建模型，其中当epoch为100，batch_size为50的时候效果比较好，如果想尝试其他参数，只需修改espcn.py的EPOCH和BATCH_SIZE的值，重新训练即可。

可以通过修改preprocessing.py文件的RATIO值来建立放大不同倍数的模型，修改FOLDER_PATH值来更改训练图片的来源。