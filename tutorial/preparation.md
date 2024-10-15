# 准备工作

## 硬件配置

在命令行输入 `dxdiag` 查看 CPU 及显卡信息

Windows 11

thinkbook 14p Gen2

* CPU: AMD Ryzen 7 5800H
* GPU: AMD Radeon(TM) Graphics 集成显卡

---

## 安装 Anaconda

安装成功后，应该有这些东西

* Anaconda Navigator，environments 中有 base 环境
* Anaconda Prompt & Anaconda Powershell Propmt

---

## 创建虚拟环境

1. 直接使用 base 环境
2. `conda create`
3. 添加镜像加速

* 查看当前有哪些虚拟环境

  `conda env list`
* 创建一个虚拟环境
  
  `conda create -n edwardpytorch python=3.12`
* 激活
  
  `conda activate edwardpytorch`
* 查看当前虚拟环境下有哪些包
  
  `conda list`
* 退出当前虚拟环境
  
  `conda deactivate`
* 删除虚拟环境
  
  `conda remove -n envname --all`

---

## 安装 Pytorch

* OS: Windows
* Pacakge: Conda
* Language: Python
* Compute Platform: CPU
* Command: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`

---

## 验证 Pytorch 是否安装成功

```cmd
python
import torch
torch.cuda.is_available()
>>>False
```

---

## 配置 Jupyter Notebook

在需要的虚拟环境中:

* `conda install jupyter`
* `conda install ipykernel`
* `python -m ipykernel install --user --name edward`
* `jupyter notebook --notebook-dir "D:\DIDL\jupyter_notebook"`

### 修改默认启动路径

* `jupyter notebook --generate-config`
* 找到 `C:\Users\Eureka is exploring\.jupyter\jupyter_notebook_config`
* 修改

  ```py
  ## The directory to use for notebooks and kernels.
  #  Default: ''
  c.ServerApp.root_dir = 'D:\DIDL\jupyter_notebook'
  ```

---

## 在 VS Code 中配置解释器路径

D:\Anaconda\envs\edwardpytorch\python.exe
