<!-- SPDX-FileCopyrightText: 2025 Baireinhold / AIH-Infra -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# AIH-OCRmyPDF 0.9.0 · 历史版本

AIH-OCR 的 0.9 系列源码整理版，基于 OCRmyPDF，提供 PDF OCR 和 Streamlit 批量处理界面。安装包名保留 `aih-ocrmypdf`，Python 模块名为 `ocrmypdf`。

本次整理保留 0.9 的界面、配置和识别逻辑，修正安装入口、包内资源及 ScreenAI 写入版本信息时重复初始化的问题。当前版本为同批发布的 AIH-OCR 1.0；两个版本分别安装到各自的虚拟环境。

## 功能

- Tesseract、PaddleOCR、Calamari、Chrome ScreenAI，以及 PDF 转换。
- 侧边栏配置、批量上传、文件与页段并行、下载及磁盘输出。
- 输出目录、文件名后缀及自动保存选项自动记忆。
- OCR 前移除符合条件的隐藏文字层，保留可见文字及原文件。

## 环境与后端

- Python 3.11–3.13；Windows 10/11、macOS 或 Linux。
- Tesseract 与 Ghostscript 为外部系统依赖，安装脚本检查并显示缺失项。Windows 自动安装使用 winget，macOS 使用 Homebrew，Ubuntu/Debian 使用 apt。
- CLI 默认使用 Tesseract；历史 Web UI 初始选择 Chrome ScreenAI。未准备 ScreenAI 组件时，在侧边栏选择 Tesseract。
- ScreenAI 库和模型需由使用者从其提供方获取；源码目录不分发这些文件。可在 Chrome 的 `chrome://components` 检查 Screen AI 组件。Linux 首次加载组件需要 `cc` 编译器。
- PaddleOCR 为可选后端。安装器按设备给出 CPU/GPU 方案，询问后安装到独立 `.venv-paddle`；主环境不默认安装 Paddle。也可运行 `python scripts/install_paddle.py`；`--dry-run` 显示方案，`--mode cpu|gpu|auto|skip` 指定操作。
- Calamari 需先在本机自行部署并启动服务，再在侧边栏配置其地址和模型。

## 安装与启动

解压源码后进入 `AIH-OCR-0.9`，运行目录内的脚本。

**Windows**：先双击 `install.bat`，完成后双击 `start.bat`（或 `启动.bat`）。

**macOS / Linux**：

```bash
chmod +x install.sh start.sh install.command start.command
./install.sh
./start.sh
```

安装器创建 `.venv`，按 `requirements-lock.txt` 安装主环境依赖，再检查系统依赖和可选 Paddle。手动安装可使用：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e . -c requirements-lock.txt
aih-ocrmypdf-webui
```

CLI 示例：

```bash
ocrmypdf --ocr-engine tesseract --output-type pdf -l eng input.pdf output.pdf
ocrmypdf --ocr-engine chrome_screenai --output-type pdf -l eng input.pdf output.pdf
```

Web UI 默认监听所有本机网口，起始端口 8501；端口不可用时递增。本机访问终端显示的 URL。环境变量 `OCRMYPDF_AIH_HOST`、`OCRMYPDF_AIH_BROWSER_HOST`、`OCRMYPDF_AIH_PORT` 可设置监听地址、浏览器地址和起始端口。

## 0.9 的处理边界

- 隐藏文字层清理采用旧规则。没有图片背景或删除会影响可见文字位置时，会保留相关文字并报告。`--hidden-text-layer keep` 保留旧层；`--skip-text` 跳过已有文字页。
- 0.9 不包含 1.0 的 PDF 对象净化、批注清除和原生文字写回重叠去重。对已有 OCR 层的文档做完整重处理，使用 1.0。
- 0.9 原生 ScreenAI 输出普通 PDF；请求 PDF/A 时该路径会改为普通 PDF。PDF/A 输出使用 Tesseract，或使用 1.0 的标准转换流程。
- 批处理结束会清理上传输入缓存；历史界面没有 1.0 的失败输入保留与重试闭环。原始文档应保留在用户自己的目录。
- PaddleOCR 不执行方向分类；旋转页面需先校正。

## 与 1.0 的版本区别

| 项目 | 0.9 历史版 | 1.0 |
|---|---|---|
| 安装包 / Web UI 命令 | aih-ocrmypdf / aih-ocrmypdf-webui | aih-ocr / aih-ocr-webui |
| 界面 | 历史侧边栏 | 工作区及四类处理配置 |
| 配置保存 | 输出位置与命名自动记忆 | 保存按钮持久保存配置 |
| 性能 | 手动并行参数 | 低/中/高及高级自定义，默认中档 |
| ScreenAI 净化与去重 | 旧去层规则 | 对象净化、已有正文覆盖检查、同位置重复词去重 |

## 许可证与第三方

项目主体采用 MPL-2.0，各文件以原 SPDX 标记为准，第三方文件保留其许可证。许可证文本位于 `LICENSES/`，元数据见 `REUSE.toml`，locro 声明见 [第三方声明](docs/third_party_locro.md)。ScreenAI 库及模型适用其提供方的独立条款。

基于 [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF)，并集成 Tesseract、PaddleOCR、Calamari 和 locro 适配代码。上游使用手册见 [OCRmyPDF 官方文档](https://ocrmypdf.readthedocs.io/en/latest/)。

## 测试

```bash
uv run --group test pytest
```

外部 OCR 引擎不可用时，相应端到端检查自动跳过。
