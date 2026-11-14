# AIH-OCR 1.0 源码发布清单

发布目录为 `AIH-OCR-1.0`，包版本为 `1.0.0`，从已提交且已打标签的开发源码导出。

## 包含

- `src/ocrmypdf/`：Python 源码与包内资源。
- `LICENSE`、`LICENSES/`、`REUSE.toml`、`docs/third_party_locro.md`：版权与第三方声明。
- `README.md`、`RELEASE_NOTES_1.0.0.md`、本清单。
- `pyproject.toml`、`uv.lock`、`requirements-lock.txt`：包元数据与依赖锁文件。
- `install.bat`、`start.bat`、`启动.bat`、`install.sh`、`start.sh`、`install.command`、`start.command`。
- `scripts/install_paddle.py`、`misc/logo.png`、`.streamlit/config.toml`。
- `tests/*.py`：公开回归检查。
- `.gitignore`、`.gitattributes`：源码管理规则。

## 排除

- `.git/`、所有虚拟环境、缓存、编译字节码、构建产物和编辑器配置。
- 用户配置、API 配置、日志、临时文件、输出、验收数据、备份、模型、训练数据和第三方运行库。
- 其他项目架构文档、历史发布说明、上游网站构建配置、捐赠模板、容器及发行打包模板。
- `docs/` 中除第三方声明以外的上游参考文档；上游使用手册通过 README 的官方链接访问。

安装脚本使用 `pip install -e . -c requirements-lock.txt` 安装锁定的主环境依赖。也可在源码目录运行 `uv sync --locked --no-dev`。Paddle 为可选独立环境，不随主环境默认安装。
