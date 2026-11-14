<!-- SPDX-FileCopyrightText: 2025 Baireinhold / AIH-Infra -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# AIH-OCR 1.0.0

批量 PDF OCR 处理管线，支持 Tesseract、PaddleOCR、Calamari 和 Chrome ScreenAI 后端，并提供 Streamlit Web UI。

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL--2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

---

## 项目定位

AIH-OCR 是 [AIH-Infra](https://github.com/AIH-Infra) 的 OCR 处理组件，基于 [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) 构建，用于为扫描版 PDF 添加可搜索文字层，并提供适合批量处理的 Web UI。

安装脚本创建 Python 环境、安装项目并检查系统依赖；启动脚本打开本机 Web UI。

```text
┌─────────────────────────────────────────────────┐
│  应用层                                          │
│  RAG / 检索 / 结构化文献处理                     │
├─────────────────────────────────────────────────┤
│  OCR 层：AIH-OCR  ← 本项目                 │
│  Tesseract · Calamari · 批量处理 · Web UI        │
├─────────────────────────────────────────────────┤
│  基础层：PDF 解析                                │
│  Ghostscript · pdfminer · pypdfium2             │
└─────────────────────────────────────────────────┘
```

## 相比上游 OCRmyPDF 的增强

| 特性 | OCRmyPDF | AIH-OCR |
|------|----------|---------------|
| OCR 后端 | Tesseract | Tesseract + PaddleOCR + Calamari + Chrome ScreenAI（可选） |
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
- CLI 默认后端为 Tesseract。Web UI 首次启动发现完整 ScreenAI 组件时默认选择 ScreenAI，否则选择 Tesseract；已保存的后端选择优先恢复。
- Calamari 需在本机自行部署并启动服务，Web UI 通过本机服务地址调用。
- PaddleOCR 是可选引擎，主线仍是 Chrome ScreenAI。`install.bat` / `install.sh` 末尾会检测显卡与 CUDA 驱动、给出 CPU 或 GPU 版推荐并**询问**是否安装（默认不装），装进独立的 `.venv-paddle`，`--ocr-engine paddle_ocr` 会自动找到它。无人值守用 `AIH_INSTALL_PADDLE=auto|cpu|gpu|skip`；也可随时手动运行 `python scripts/install_paddle.py`（`--dry-run` 只看方案）。
- 也可以按 extra 安装进主环境（不推荐，会调整 numpy/opencv）：`pip install "aih-ocr[paddle]"`（PaddleOCR 3.7 + PaddlePaddle 3.3 CPU 版）。依赖较重（会引入 `opencv-contrib-python` 并可能调整 numpy），建议装在独立虚拟环境，再用 `--paddle-ocr-python` 或环境变量 `AIH_PADDLE_PYTHON` 指向它。GPU 版（CUDA 13.0，RTX 50 系列适用）：`uv sync --extra paddle-gpu`（`paddle` 与 `paddle-gpu` 互斥），或用 pip：`pip install "aih-ocr[paddle-gpu]" --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/`，运行时加 `--paddle-ocr-device gpu:0`；GPU 不可用时 Worker 会自动退回 CPU。用法：`ocrmypdf --ocr-engine paddle_ocr -l eng in.pdf out.pdf`。
- Chrome ScreenAI 已集成 `locro` 适配代码。库和模型组件需由使用者从其提供方获取，源码包不含这些文件。安装 Chrome 后，在 `chrome://components` 检查 Screen AI 组件，完成下载后在界面点击“重新扫描 ScreenAI”。支持的发现范围见扫描详情；Linux 首次加载组件还需要提供 `cc` 编译器。
- Web UI 上传内容和临时输出采用任务级磁盘缓存；任务完成后释放输入缓存，清除结果时释放下载缓存。
- Chrome ScreenAI 的 Web UI 并行数由性能档位按 CPU 和可用内存推荐；CLI 使用 `-j N` 设置，CLI 未指定时使用 8 个页段 worker。
- Web UI 默认监听 `0.0.0.0:8501`，也就是本机所有网口；本机浏览器默认打开 `http://localhost:8501`。若端口被占用或在 Windows 上命中系统保留端口，会自动递增到下一个可用端口。
- Streamlit 通常会自动打开默认浏览器；若未自动打开，请以终端输出的最终 URL 为准。
- 系统依赖自动安装入口：Windows 使用 `winget`，macOS 使用 Homebrew，Linux 使用 `apt`（Ubuntu / Debian）；缺失项会显示手动安装提示。
- 安装脚本检查 Tesseract 主程序、Ghostscript 和常用 Tesseract 语言包，并显示缺失项。
- 常用语言包包括：`eng`、`chi_sim`、`chi_tra`、`deu`、`fra`、`jpn`、`kor`、`lat`。
- 如果自动安装系统依赖失败，请按脚本提示手动安装 Tesseract / Ghostscript 后重新运行启动脚本。

## 批处理规模说明

- Web UI 适合交互式处理单本大书或几十个文件。
- 大批量任务应优先使用磁盘输出，避免在浏览器中生成大体积 ZIP。
- Chrome ScreenAI 的单书页内并行与多书并行是互斥策略，避免形成并发乘法。
- 长期无人值守批次由外部任务队列调度；Web UI 提供当前队列的状态与失败重试。

## 使用方式一：从源码目录运行

适合本地开发、调试和直接使用仓库内脚本。

### 1. 解压源码

将 1.0 源码包解压到本机目录，进入含有 `pyproject.toml` 和安装脚本的 `AIH-OCR-1.0` 文件夹。

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
- 按 `requirements-lock.txt` 约束执行 `pip install -e .`
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
pip install . -c requirements-lock.txt
```

开发模式安装：

```bash
pip install -e . -c requirements-lock.txt
```

### 2. 启动命令

OCRmyPDF CLI：

```bash
ocrmypdf input.pdf output.pdf
```

Web UI：

```bash
aih-ocr-webui
```

如需修改监听地址或起始尝试端口，可设置环境变量：

```bash
OCRMYPDF_AIH_HOST=0.0.0.0
OCRMYPDF_AIH_BROWSER_HOST=localhost
OCRMYPDF_AIH_PORT=8501
aih-ocr-webui
```

其中：
- `OCRMYPDF_AIH_HOST`：监听地址，默认 `0.0.0.0`
- `OCRMYPDF_AIH_BROWSER_HOST`：本机默认打开的浏览器地址，默认 `localhost`
- `OCRMYPDF_AIH_PORT`：起始尝试端口，默认 `8501`；如果该端口不可用，程序会自动继续尝试更高端口

## Web UI 说明

Web UI 提供：
- 多文件批量上传
- Tesseract、PaddleOCR、Calamari、Chrome ScreenAI 与仅转换 PDF
- OCR 模式、图像预处理等参数配置
- 性能支持低、中、高三档，默认中档；根据硬件、后端、模型配置与文件数量联动，也可选择高级自定义
- 点击“保存配置”持久保存识别、输出和自定义性能设置，下次启动恢复
- 输出模式选择：
  - 普通 PDF
  - 尽量 PDF/A（失败则回退普通 PDF）
  - 严格 PDF/A

默认推荐使用普通 PDF，以降低因特殊颜色空间导致的 PDF/A 转换失败概率。

Chrome ScreenAI 的并发策略独立于性能档位，默认“单文件内并发优先”：逐个文件处理，集中资源并行处理该文件的页段。选择“多文件进程池并行”时，同时处理多个文件，每个文件内部顺序处理。策略和自定义并行数可保存。

Chrome ScreenAI 默认先做 PDF 对象净化：保留可见正文、原始图片编码、页面版式和书签，清除选定页的旧隐藏 OCR、批注、高亮、链接和表单注释。净化不进行全页栅格化；原文件不变。原生文字层写回保留已有正文，OCR 仅补入未覆盖区域，同位置重复识别词只写一次。净化副本若仍大于输入文件，处理会明确停止。后续 OCR 新增的文字层和字体另计入最终文件大小。

PDF/A 输入可正常处理。选择严格 PDF/A 输出时，ScreenAI 结果进入标准格式转换与校验流程，失败会报告；选择普通 PDF 时输出普通 PDF。

## Calamari 后端

在本机部署并启动 Calamari 服务后，在 Web UI 中选择 Calamari 并填写本机服务地址（默认 http://localhost:11800）。

说明：
- Calamari 是可选增强路径，不配置时仍可正常使用 Tesseract。
- 项目不会在安装阶段自动部署 Calamari 服务。
- 历史文献识别可选用本机 Calamari 服务中已部署的相应模型。

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
- 安装项目后可以直接运行 `aih-ocr-webui`
- Web UI 的启动、端口选择和浏览器打开逻辑都已收口到正式包入口

## 上游功能保留

AIH-OCR 保留了 OCRmyPDF 的核心能力，包括：

- 可搜索 PDF 输出
- 旋转校正、倾斜校正、图像清理
- 多核并行处理
- OCR 层注入
- 插件系统

## Tesseract、PaddleOCR 与 Calamari 的隐藏文字层处理

以下为通用 OCR 管线的隐藏层策略；ScreenAI 默认对象净化与原生写回规则见上方 Web UI 说明。

很多 PDF 是“扫描图 + 上一次 OCR 留下的隐形文字层”。上游 OCRmyPDF 遇到这类文件会报 `page already has text`（退出码 6）。本项目默认改为：**OCR 前先去掉隐形文字层，再重新 OCR**（`--mode default` 与 `--redo-ocr`）。

- 只删“不可见”文字（渲染模式 3、透明度 0 等），可见文字不动；输入文件不会被改写。
- 先用 MuPDF 逐页判断：没有隐形文字的页面原样保留，不重写内容流。
- 页面上没有图片/表单垫底（删除会变成白页）的页，保留隐形文字并给出警告。
- 隐形文字之后紧跟同一文本对象里的可见文字、删除会使其错位的页，保留并警告。
- `'` / `"` 运算符会以 `T*` 保留换行；处理后再用 MuPDF 复核，可见文字有变化则放弃使用去层结果。
- `--hidden-text-layer keep` 恢复上游行为；`--skip-text` 会跳过已有文字的页，不去层。

```bash
ocrmypdf --ocr-engine tesseract in.pdf out.pdf                            # 默认：去层后重新 OCR
ocrmypdf --hidden-text-layer keep --skip-text in.pdf out.pdf               # 保留原文字层
```

更多 CLI 文档可参考 [OCRmyPDF 官方文档](https://ocrmypdf.readthedocs.io/en/latest/)。

## 许可证

项目主体采用 **Mozilla Public License 2.0 (MPL-2.0)**；各文件以 SPDX 标记为准。第三方源码及资源适用其原许可证，文本见 `LICENSES/` 和 `REUSE.toml`，locro 集成声明见 [第三方声明](docs/third_party_locro.md)。Chrome ScreenAI 库和模型适用其提供方的条款。

## 致谢

- [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) — James R. Barlow 及贡献者
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Calamari OCR](https://github.com/Calamari-OCR/calamari)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- locro 与 Chrome ScreenAI：见 [第三方声明](docs/third_party_locro.md)

## 作者

**Güriedrich & Baireinhold** — [AIH-Infra](https://github.com/AIH-Infra)

## 测试

```bash
uv run --group test pytest            # 全部（外部引擎不可用时自动跳过）
uv run --group test pytest -m "not slow and not paddle"
```

测试位于 `tests/`：`test_cli_options.py`（选项解析、插件自动加载）、`test_stdout_hygiene.py`（stdout 只输出 PDF 的约束）、`test_cli_run.py`（真实 CLI 端到端；按 `tesseract` / `chrome` / `paddle` 标记，对应引擎不可用时自动跳过）。


项目安装包名为 `aih-ocr`，命令行为 `aih-ocr`，Web UI 命令为 `aih-ocr-webui`。Python 模块名保留 `ocrmypdf`；旧命令 `ocrmypdf`、`aih-ocrmypdf-webui`、`ocrmypdf-aih-webui` 继续可用。现有 Web UI 配置文件继续使用。开发目录为 `AIH-OCR-develop`。
