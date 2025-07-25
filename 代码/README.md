# MasterPi 项目

本项目包含了控制 MasterPi 机器人所需的所有代码。

## 依赖项

要运行本项目中的代码，您需要安装以下依赖项。

### 使用 pip 安装

您可以使用 pip 来安装所有必需的 Python 库。建议在虚拟环境中使用 pip。

```bash
pip3 install -r requirements.txt
```

或者，您也可以单独安装每个软件包：

```bash
pip3 install pyyaml opencv-python numpy werkzeug jsonrpc-py RPi.GPIO smbus2 rpi_ws281x matplotlib torch ultralytics pandas Pillow
```

**注意:** `RPi.GPIO` 和 `rpi_ws281x` 是树莓派专用的库，用于控制 GPIO 引脚和 RGB LED 等硬件组件。

### `requirements.txt` 文件

为了方便安装，项目根目录提供了 `requirements.txt` 文件。