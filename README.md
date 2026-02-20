<!-- SPDX-FileCopyrightText: 2025 Baireinhold / AIH-Infra -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# OCRmyPDF-AIH

**批量 PDF OCR 处理管线，支持 Tesseract / Calamari 双后端**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL--2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

---

## 项目定位

OCRmyPDF-AIH 是 [AIH-Infra（人文学科人工智能基础设施）](https://github.com/AIH-Infra) 的 OCR 处理组件，基于 [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) v17.2.0 构建。

```
┌─────────────────────────────────────────────────┐
│  应用层                                          │
│  朴素 RAG → Graph RAG → Agent RAG               │
├─────────────────────────────────────────────────┤
│  0.5 层：經緯·Contexture                         │
│  页码锚点 · 边注识别 · 行内注 · 模板 · 配置共享   │
├─────────────────────────────────────────────────┤
│  OCR 层：OCRmyPDF-AIH  ← 本项目                 │
│  Tesseract · Calamari · 批量处理 · Web UI        │
├─────────────────────────────────────────────────┤
│  基础层：PDF 解析                                │
│  Ghostscript · pdftext · pypdfium2              │
└─────────────────────────────────────────────────┘
```

OCRmyPDF-AIH 为扫描版 PDF 添加可搜索的 OCR 文字层，输出标准 PDF/A 文件。这些文件随后可被 [經緯·Contexture](https://github.com/AIH-Infra/aih-contexture) 进一步处理为带页码锚点的结构化 Markdown，进入学术 RAG 检索流程。

## 相比上游 OCRmyPDF 的增强

| 特性 | OCRmyPDF | OCRmyPDF-AIH |
|------|----------|---------------|
| OCR 后端 | Tesseract | Tesseract + Calamari GPU |
| 使用方式 | CLI | CLI + Streamlit Web UI |
| 批量处理 | 逐文件命令行 | Web UI 多文件批量 + ZIP 打包下载 |
| 进度显示 | 文件级 | 页面级实时进度 |
| 语言选择 | 手动指定 | 自动检测已安装语言包 |
| 参数调优 | CLI 参数 | 可视化面板（OCR 模式、预处理、输出格式） |

## 快速开始

### 环境要求

- Python 3.11 / 3.12 / 3.13
- Windows 10/11、macOS 或 Linux
- Ghostscript、Tesseract OCR

### 一键安装

**Windows：** 双击 `install.bat`

**macOS：**
```bash
chmod +x install.command start.command install.sh start.sh
# 然后双击 install.command
```

**Linux：**
```bash
chmod +x install.sh start.sh
./install.sh
```

安装脚本会自动创建虚拟环境、安装依赖，并尝试自动安装 Tesseract（失败时提示手动安装）。

### pip 安装

```bash
pip install ocrmypdf-aih-infra
```

### 启动 Web UI

**Windows：** 双击 `start.bat`

**macOS / Linux：** `./start.sh`

浏览器访问 `http://localhost:6106`

## Calamari GPU 后端

OCRmyPDF-AIH 支持通过远程 API 调用 [Calamari OCR](https://github.com/Calamari-OCR/calamari) GPU 服务，适用于需要高精度历史文献识别的场景。

在 Web UI 侧边栏选择 Calamari 后端，填入服务地址即可使用。

## 上游功能

OCRmyPDF-AIH 完整保留了 OCRmyPDF 的所有功能：

- 生成可搜索的 PDF/A 文件
- 支持 100+ 语言（Tesseract 语言包）
- 页面旋转校正、倾斜校正、图像清理
- 多核并行处理
- 无损 OCR 层注入
- PDF 图像优化
- 插件系统

CLI 用法与上游完全兼容：

```bash
ocrmypdf -l chi_sim+eng --deskew input.pdf output.pdf
```

详细文档参见 [OCRmyPDF 官方文档](https://ocrmypdf.readthedocs.io/en/latest/)。

## 关于 AIH-Infra

**AIH-Infra（人文学科人工智能基础设施）** 致力于为人文学科研究者提供可追溯、可验证、可传承的 AI 工具链。

- **材料线**：[經緯·Contexture](https://github.com/AIH-Infra/aih-contexture)（文献数字化与结构化）
- **OCR 层**：OCRmyPDF-AIH（本项目，PDF OCR 处理）
- **系统线**：学术 RAG 知识库与检索系统

核心原则：**每一条 AI 生成的回答，都必须能够返回原书的那一页。**

## 许可证

本项目基于 **Mozilla Public License 2.0 (MPL-2.0)** 发布，与上游 OCRmyPDF 保持一致。

## 致谢

- [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) — James R. Barlow 及贡献者
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Calamari OCR](https://github.com/Calamari-OCR/calamari)

## 作者

**Güriedrich & Baireinhold** — [AIH-Infra](https://github.com/AIH-Infra)
