# OCRmyPDF-AIH 0.7.0

## 主要变化

- 正式集成 Chrome ScreenAI `locro` 适配代码，并保留 MIT 许可证和第三方说明。
- 支持 Chrome ScreenAI 原生直通、单书页段并行和多书整本并行。
- Chrome 单书并行默认 8 worker，最高 16 worker。
- 增加 PaddleOCR 本地全链路后端，以及 PaddleOCR/Chrome ScreenAI 行检测接入。
- 增加轻量文本层清理和显式栅格化清理选项。
- Web UI 改为任务级磁盘缓存，减少批量上传和输出造成的内存占用。
- 改进 Windows Web UI 端口发现和全网口监听。

## 已知边界

- Chrome ScreenAI 依赖本机 Chrome 提供的 ScreenAI DLL 和模型。
- Calamari 需要用户单独部署 HTTP 服务。
- PaddleOCR 的 CPU/GPU 环境需要用户安装对应 PaddlePaddle 运行时。
- Web UI 面向交互式批处理，不包含持久化任务数据库和断点续跑。
- 上千文件的无人值守任务建议使用外部队列或 AIH OCR MCP。

## 许可证

项目主体遵循 MPL-2.0。继承的 OCRmyPDF 文件保留各自 SPDX 声明；vendored `locro` 代码遵循 MIT License，许可证位于 `LICENSES/locro-MIT-LICENSE.txt`。
