# AIH-OCRmyPDF 0.9.0 · 历史源码整理版

项目标识统一为 **AIH-OCRmyPDF**（原 OCRmyPDF-AIH）。安装包名 `aih-ocrmypdf`，Web UI 命令 `aih-ocrmypdf-webui`（旧命令 `ocrmypdf-aih-webui` 保留为别名）。目录名可自行决定，与功能无关。

## 主要变化

- **默认先去隐形文字层再 OCR**：扫描图 + 旧 OCR 隐形文字层的 PDF 不再因“已有文字”中止（原退出码 6）。新增 `--hidden-text-layer {strip,keep}`，Web UI 同步增加开关。
  - 只删不可见文字；先用 MuPDF 逐页判断；背后没有图片的页、删除会让可见文字错位的页保留并警告；`'` / `"` 保留换行；处理后复核；不改源文件。
- **PaddleOCR 作为可选后端**（PaddleOCR 3.7 / PaddlePaddle 3.x，支持 GPU）：`install.bat` / `install.sh` 末尾按设备检测并询问，装入独立 `.venv-paddle`；也可 `python scripts/install_paddle.py`。主线仍是 Chrome ScreenAI。
- **CLI 修复**：stdout 只输出 PDF（插件诊断改走日志）、`--ocr-engine` 支持 paddle_ocr / chrome_screenai / calamari、AIH 插件按需自动加载、Calamari 不可达时明确报错。
- **Web UI**：输出后缀、输出目录、是否自动保存均自动持久化；已有文字页给出明确提示；Chrome 多本并行 Worker 失败时显示原因，并避免管道阻塞。
- **安装脚本**：`.bat` 统一 CRLF；项目 `.tessdata` 补齐 `configs/` 与 `osd`，修复 Tesseract 退出码 15。
- **测试**：新增 `tests/`（102 项，外部引擎缺失时自动跳过）。

## 已知边界

- Chrome ScreenAI 依赖本机 Chrome 提供的 ScreenAI DLL 和模型。
- Calamari 需要用户单独部署 HTTP 服务（在线服务尚未实测）。
- PaddleOCR 不做方向分类，旋转页面可能漏识别；Tesseract 对大号合成中文效果一般。
- Linux / macOS 的安装脚本尚未在真机验证。
- Web UI 面向交互式批处理，不含断点续跑；上千文件的无人值守任务建议使用外部队列或 AIH OCR MCP。

## 许可证

项目主体遵循 MPL-2.0。继承的 OCRmyPDF 文件保留各自 SPDX 声明；vendored `locro` 代码遵循 MIT License，许可证位于 `LICENSES/locro-MIT-LICENSE.txt`。

## 历史版公开整理

- 保留0.9.0版本号、aih-ocrmypdf包名、历史侧边栏和既有处理功能。
- 主环境依赖使用从原锁文件导出的requirements-lock.txt；安装脚本先补齐pip，再通过当前Python安装。
- 修正wheel内logo及locro许可证的位置，启动脚本统一从当前Python调用正式入口。
- 修正Windows安装器的重复标签、重复调用与提前退出，确保语言包报告、Ghostscript及可选Paddle步骤可达。
- 回移ScreenAI版本读取稳定性修复：写入PDF创建者信息时只查询库版本，不重复初始化OCR；保留原生错误诊断。
- 公开目录排除其他项目架构文档、上游站点模板、环境、模型、缓存、配置及运行数据。
- 0.9不含1.0的对象净化与重复文字写回增强；原生ScreenAI输出普通PDF。处理边界与版本区别见README。
