# AIH-OCR 0.9 历史源码发布清单

包版本为0.9.0，包名aih-ocrmypdf，发布目录AIH-OCR-0.9。以原始0.9快照为源，在历史分支固定公开整理内容；原目录保留。

包含src/ocrmypdf、LICENSE、LICENSES、REUSE.toml、docs/third_party_locro.md、README、0.9发布说明、本清单、pyproject.toml、uv.lock、requirements-lock.txt、全部根目录install/start脚本与启动.bat、scripts/install_paddle.py、misc/logo.png、.streamlit/config.toml、tests/*.py、.gitignore、.gitattributes。

排除.git、环境、模型和运行库、缓存和字节码、用户配置/凭据、日志/输出/验收数据、备份、其他项目架构文档、历史站点/容器/发行模板以及除第三方声明以外的上游参考手册。BAT导出使用CRLF，Unix脚本使用LF。

安装命令为python -m pip install -e . -c requirements-lock.txt；Paddle通过独立环境按需安装。历史功能与当前版区别见README。
