import yaml
import os
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 加载配置
config_path = 'web_config.yml'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
else:
    # 默认配置
    config = {
        'port': 3000,
        'host': 'localhost',
        'debug': False
    }

@app.route('/')
def index():
    """提供空页面"""
    return '', 200

if __name__ == '__main__':
    # 启动服务器
    app.run(
        host=config.get('host', 'localhost'),
        port=config.get('port', 3000),
        debug=config.get('debug', False)
    )