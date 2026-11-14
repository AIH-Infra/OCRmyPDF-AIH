# AIH-OCR 1.0.0

## PDF OCR 与文字层

- 支持 Tesseract、PaddleOCR、Calamari 和 Chrome ScreenAI，以及仅转换 PDF。
- Chrome ScreenAI 默认进行 PDF 对象净化，清除所选页的旧隐藏 OCR 和批注，保留可见正文及原始图片编码。净化副本超过输入大小时停止；后续 OCR 的文字层和字体会计入成品大小。
- ScreenAI 原生文字层写回避开已有可见正文覆盖区域，并去除同位置重复识别词，补充图片中的文字。
- ScreenAI 输出进入标准 PDF/PDF-A 转换流程；严格 PDF/A 转换失败会报告错误。
- 读取 ScreenAI 版本不再重复初始化 OCR，原生进程异常保留诊断信息。

## 使用与配置

- 提供文件队列、识别/文档/输出/性能配置，以及逐文件结果和失败重试入口。
- 性能档位为低、中、高及高级自定义，默认中档，按本机资源和后端参数推荐。
- ScreenAI 默认优先单文件内并发，可切换多文件进程池并行；两种策略分别分配资源。
- 点击“保存配置”持久保存用户配置，下次启动恢复。
- ScreenAI 支持自动发现和重新扫描；Paddle 支持独立环境安装、检测及自动调用。

## 安装与分发

- Python 3.11–3.13；提供 Windows、macOS 和 Linux 安装及启动脚本，主环境依赖通过约束文件固定。
- 首次启动按 ScreenAI 组件完整性选择初始后端；未安装组件时初始选择 Tesseract，已保存配置优先恢复。
- Python 包名为 aih-ocr，入口为 aih-ocr 和 aih-ocr-webui；保留 ocrmypdf 等兼容入口。
- 发布源码不包含虚拟环境、用户配置、模型权重、运行输出和开发验收数据。
- Calamari 服务由使用者部署；ScreenAI 库与模型由使用者从其提供方获取，源码包不分发该组件。
