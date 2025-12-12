import yaml
import os
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 设置允许的文件类型
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    """返回主页"""
    return send_from_directory('page', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    """提供静态文件服务"""
    return send_from_directory('page', path)

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    """处理数据集文件上传"""
    # 检查是否有文件在请求中
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    # 检查文件类型并保存
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
    else:
        return jsonify({'error': 'Allowed file types are csv only'}), 400

if __name__ == '__main__':
    # 启动服务器
    app.run(
        host=config.get('host', 'localhost'),
        port=config.get('port', 3000),
        debug=config.get('debug', False)
    )