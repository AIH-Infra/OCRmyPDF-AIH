<!-- SPDX-FileCopyrightText: 2025 Baireinhold / AIH-Infra -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# OCRmyPDF-AIH

批量 PDF OCR 处理管线，支持 Tesseract / Calamari 双后端，并提供 Streamlit Web UI。

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL--2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

---

## 项目定位

OCRmyPDF-AIH 是 [AIH-Infra](https://github.com/AIH-Infra) 的 OCR 处理组件，基于 [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) 构建，用于为扫描版 PDF 添加可搜索文字层，并提供适合批量处理的 Web UI。

项目目标是以尽量低的使用门槛提供可用的批量 OCR 服务；仓库附带安装脚本和启动脚本，但不承诺所有系统环境下零配置完成系统依赖安装。

```text
┌─────────────────────────────────────────────────┐
│  应用层                                          │
│  RAG / 检索 / 结构化文献处理                     │
├─────────────────────────────────────────────────┤
│  OCR 层：OCRmyPDF-AIH  ← 本项目                 │
│  Tesseract · Calamari · 批量处理 · Web UI        │
├─────────────────────────────────────────────────┤
│  基础层：PDF 解析                                │
│  Ghostscript · pdfminer · pypdfium2             │
└─────────────────────────────────────────────────┘
```

## 相比上游 OCRmyPDF 的增强

| 特性 | OCRmyPDF | OCRmyPDF-AIH |
|------|----------|---------------|
| OCR 后端 | Tesseract | Tesseract + Calamari（可选） |
| 使用方式 | CLI | CLI + Streamlit Web UI |
| 批量处理 | 逐文件命令行 | Web UI 多文件批量处理 |
| 进度显示 | 文件级 | 页面级进度反馈 |
| 参数配置 | CLI 参数 | 可视化面板 |
| 输出模式 | CLI 手动指定 | UI 中可选普通 PDF / 尽量 PDF/A / 严格 PDF/A |

## 环境要求

- Python 3.11 / 3.12 / 3.13
- Windows 10/11、macOS 或 Linux
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Ghostscript](https://www.ghostscript.com/)

说明：
- Tesseract 是默认后端。
- Calamari 不是内置依赖，而是可选的外部 HTTP 服务。
- Web UI 默认从 `http://127.0.0.1:8501` 开始尝试启动；若端口被占用或在 Windows 上命中系统保留端口，会自动递增到下一个可用端口。
- Streamlit 通常会自动打开默认浏览器；若未自动打开，请以终端输出的最终 URL 为准。
- 仓库提供跨平台安装脚本，但系统依赖的自动安装属于 best effort：Windows 依赖 `winget`，macOS 依赖 Homebrew，Linux 安装脚本优先支持 `apt`（Ubuntu / Debian）。
- 安装目标是稳定安装 Tesseract 主程序、Ghostscript 和常用 Tesseract 语言包，并清楚提示缺失语言包；不承诺自动安装所有 Tesseract 语言包。
- 常用语言包包括：`eng`、`chi_sim`、`chi_tra`、`deu`、`fra`、`jpn`、`kor`、`lat`。
- 如果自动安装系统依赖失败，请按脚本提示手动安装 Tesseract / Ghostscript 后重新运行启动脚本。

## 使用方式一：克隆源码后直接运行

适合本地开发、调试和直接使用仓库内脚本。

### 1. 克隆仓库

```bash
git clone https://github.com/Baireinhold/OCRmyPDF-AIH.git
cd OCRmyPDF-AIH
```

### 2. 运行安装脚本

**Windows**

双击 `install.bat`

**macOS**

```bash
chmod +x install.command start.command install.sh start.sh
./install.sh
```

**Linux**

```bash
chmod +x install.sh start.sh
./install.sh
```

安装脚本会：
- 检查 Python 版本
- 创建 `.venv`
- 执行 `pip install -e .`
- 检测 Tesseract / Ghostscript
- 在支持的平台上尝试自动安装系统依赖
- 安装后再次验证当前终端是否可以调用 Tesseract / Ghostscript
- 检查常用 Tesseract 语言包是否可用，并提示缺失项

说明：
- Windows 自动安装依赖 `winget`；脚本会检测全局 `tesseract.exe`，并在发现常见安装目录时临时加入当前会话 PATH
- Windows 若缺少常用 Tesseract 语言包，脚本会尝试下载到项目本地 `.tessdata` 目录，启动脚本会自动设置 `TESSDATA_PREFIX`
- macOS 自动安装依赖 Homebrew
- Linux 安装脚本优先支持 `apt`（Ubuntu / Debian）
- 其他 Linux 发行版通常需要手动安装系统依赖
- 脚本不会默认永久修改系统 PATH；Windows 安装器可能会写入系统 PATH，但通常需要打开新终端后生效

### 3. 启动 Web UI

**Windows**

双击 `start.bat`

**macOS / Linux**

```bash
./start.sh
```

启动脚本会自动打开默认浏览器。

如果没有自动打开，请查看终端输出的最终 URL 并手动访问。

## 使用方式二：作为 Python 包安装

适合已有 Python 环境、希望不依赖仓库脚本直接启动的场景。

### 1. 安装项目

从源码目录安装：

```bash
pip install .
```

开发模式安装：

```bash
pip install -e .
```

### 2. 启动命令

OCRmyPDF CLI：

```bash
ocrmypdf input.pdf output.pdf
```

Web UI：

```bash
ocrmypdf-aih-webui
```

如需修改监听地址或起始尝试端口，可设置环境变量：

```bash
OCRMYPDF_AIH_HOST=127.0.0.1
OCRMYPDF_AIH_PORT=8501
ocrmypdf-aih-webui
```

其中：
- `OCRMYPDF_AIH_HOST`：监听地址，默认 `127.0.0.1`
- `OCRMYPDF_AIH_PORT`：起始尝试端口，默认 `8501`；如果该端口不可用，程序会自动继续尝试更高端口

## Web UI 说明

Web UI 提供：
- 多文件批量上传
- Tesseract / Calamari 后端切换
- OCR 模式、图像预处理等参数配置
- 输出模式选择：
  - 普通 PDF
  - 尽量 PDF/A（失败则回退普通 PDF）
  - 严格 PDF/A

默认推荐使用普通 PDF，以降低因特殊颜色空间导致的 PDF/A 转换失败概率。

## Calamari 后端

如果你部署了 Calamari HTTP 服务，可在 Web UI 中切换到 Calamari 后端并填写服务地址。

说明：
- Calamari 是可选增强路径，不配置时仍可正常使用 Tesseract。
- 项目不会在安装阶段自动部署 Calamari 服务。
- 如需高精度历史文献识别，可将 Calamari 作为远程 OCR 服务接入。

## CLI 示例

CLI 与上游 OCRmyPDF 基本兼容：

```bash
ocrmypdf -l chi_sim+eng --deskew input.pdf output.pdf
```

如果遇到特殊颜色空间 PDF 无法生成 PDF/A，可显式输出普通 PDF：

```bash
ocrmypdf --output-type pdf input.pdf output.pdf
```

## 项目结构说明

当前公开副本已将 Web UI 运行时代码纳入正式包结构，避免依赖仓库中的临时脚本路径：

- `src/ocrmypdf/webui.py`：Web UI 主程序
- `src/ocrmypdf/webui_launcher.py`：Web UI 启动入口
- `src/ocrmypdf/aih_calamari_plugin.py`：Calamari 插件
- `src/ocrmypdf/aih_progress_plugin.py`：进度插件

因此：
- `pip install .` 后可以直接运行 `ocrmypdf-aih-webui`
- Web UI 的启动、端口选择和浏览器打开逻辑都已收口到正式包入口

## 上游功能保留

OCRmyPDF-AIH 保留了 OCRmyPDF 的核心能力，包括：

- 可搜索 PDF 输出
- 旋转校正、倾斜校正、图像清理
- 多核并行处理
- OCR 层注入
- 插件系统

更多 CLI 文档可参考 [OCRmyPDF 官方文档](https://ocrmypdf.readthedocs.io/en/latest/)。

## 许可证

本项目基于 **Mozilla Public License 2.0 (MPL-2.0)** 发布，与上游 OCRmyPDF 保持一致。

## 致谢

- [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) — James R. Barlow 及贡献者
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Calamari OCR](https://github.com/Calamari-OCR/calamari)

## 作者

**Güriedrich & Baireinhold** — [AIH-Infra](https://github.com/AIH-Infra)
