# uv 包管理指南

`uv` 是一个现代 Python 包管理工具，比传统的 `pip` 更快、更可靠。本指南介绍如何使用 `uv` 管理项目依赖。

## 添加包

### 添加最新版本
```bash
uv add package_name
```

### 添加指定版本
```bash
uv add package_name==1.2.3
uv add "package_name>=1.0.0,<2.0.0"
```

### 添加为开发依赖
```bash
uv add --dev package_name
# 或
uv add -D package_name
```

### 添加多个包
```bash
uv add package1 package2 package3
```

## 移除包

### 移除包
```bash
uv remove package_name
```

### 移除开发依赖
```bash
uv remove --dev package_name
```

### 移除多个包
```bash
uv remove package1 package2 package3
```

## 常用选项

- `--dev` 或 `-D`: 添加为开发依赖（测试工具、linter 等）
- `--optional`: 添加为可选依赖
- `--no-sync`: 只更新配置文件而不安装

## 查看依赖信息

### 列出所有已安装的包
```bash
uv pip list
```

### 查看项目依赖树
```bash
uv tree
```

### 查看特定包信息
```bash
uv show package_name
```

## 项目初始化

如果项目还没有使用 `uv`，可以这样初始化：

```bash
# 创建新项目
uv init project_name

# 在现有项目中初始化
uv init
```

## 同步依赖

安装 `pyproject.toml` 中定义的所有依赖：

```bash
uv sync
```

只安装生产依赖：

```bash
uv sync --no-dev
```

## 优点

- **速度**: 比 pip 快 10-100 倍
- **可靠性**: 更好的依赖解析
- **跨平台**: 在不同操作系统上表现一致
- **现代化**: 支持最新的 Python 包管理标准

## 常见使用场景

### 开发环境设置
```bash
# 添加开发工具
uv add --dev pytest black flake8 mypy

# 添加生产依赖
uv add fastapi uvicorn
```

### 依赖更新
```bash
# 更新所有依赖到最新兼容版本
uv lock --upgrade

# 更新特定包
uv add package_name@latest
```

## 配置文件

`uv` 使用标准的 `pyproject.toml` 文件来管理依赖，这使得项目配置更加标准化和可移植。