# evi 语音输入法
rust 开发的跨平台语音输入法
能学习用户的语音习惯，优化输入表现

V2.9.1下载
通过网盘分享的文件：evi release
链接: https://pan.baidu.com/s/1BAR7b9Kc-1KtXa1tRh13rA?pwd=1234 提取码: 1234

# 路线图
- 极速版，纯本地纯cpu推理，不需要网络连接，速度飞快
- pro版，增加可开关的网络llm润色，用户自己选配llm进行润色
- master版，本地润色版
- 终极版，自动进化版

# 开发进度
- [x] 极速版 -- windows
- [x] 极速版 -- mac intel芯片适配
- [x] pro版 -- 网络llm润色和自动纠错功能
- [x] Pro版 -- 模型自动下载
- [x] pro版 -- windows
- [x] pro版 -- mac intel芯片适配
- [x] pro版 -- mac 苹果芯片适配
- [x] pro版 -- 增加隐私说明，增加内容提交和审核界面
- [ ] master版 -- 全本地润色
- [ ] master版 -- windows
- [ ] master版 -- mac intel芯片适配
- [ ] master版 -- mac 苹果芯片适配
- [ ] 终极版 -- 自动进化， 持续本地学习用户个人语音和用词习惯
- [ ] 多端共享个人习惯数据

# 使用方法
- windows版
  - 双击zip文件，选择一个目录解压
  - 双击程序，会有一个绿色小e的图标在右下角，建议锁定到任务栏方便后续使用
  - 按下右ctrl键说话，释放右ctrl键进行语音识别
  - 语音识别结果会自动填入当前聚焦的文本框
  - 安装视频（待录制）：

- mac版
  - 双击打开安装包，拖到Applications
  - 后续在finder中启用（需要设置权限）
  - 因为app没有签名，需要解除隔离才能正常启动，解除命令如下:
    ```bash
    sudo xattr -rd com.apple.quarantine /Applications/EVI.app 
    ```
  - 启动后在右上角会有绿色小e图标
  - 按下右command键说话，释放右command键进行语音识别
  - 语音识别结果会自动填入当前聚焦的文本框
  - 安装视频（待录制）：

# 鸣谢
基于阿里funasr语音识别开发
