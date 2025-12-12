import yaml
import os
import subprocess
import sys
from flask import Flask, send_from_directory, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import queue

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置上传文件夹
UPLOAD_FOLDER = 'data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 设置允许的文件类型
ALLOWED_EXTENSIONS = {'csv'}

# Store training processes
training_processes = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """获取数据集文件列表"""
    try:
        datasets = []
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath) and allowed_file(filename):
                file_stats = os.stat(filepath)
                datasets.append({
                    'name': filename,
                    'size': file_stats.st_size,
                    'modified': file_stats.st_mtime
                })
        return jsonify(datasets), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-training', methods=['POST'])
def start_training():
    """启动模型训练"""
    try:
        data = request.get_json()
        dataset_name = data.get('dataset')
        
        if not dataset_name:
            return jsonify({'error': 'No dataset specified'}), 400
            
        # Check if dataset exists
        dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name)
        if not os.path.exists(dataset_path):
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Generate a unique ID for this training process
        import uuid
        import time
        training_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        # Start training in a separate thread
        def run_training():
            try:
                # Modify the train.py CONFIG to use the selected dataset
                env = os.environ.copy()
                env['TRAINING_DATASET'] = dataset_path
                env['TRAINING_ID'] = training_id
                
                # Run train.py as a subprocess
                process = subprocess.Popen([
                    sys.executable, '-u', 'train.py'
                ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                   bufsize=1, universal_newlines=True, env=env)
                
                training_processes[training_id] = {
                    'process': process,
                    'output': [],
                    'status': 'running',
                    'start_time': timestamp
                }
                
                # Capture output
                for line in iter(process.stdout.readline, ''):
                    if training_id in training_processes:
                        training_processes[training_id]['output'].append(line)
                
                process.wait()
                if training_id in training_processes:
                    training_processes[training_id]['status'] = 'completed' if process.returncode == 0 else 'failed'
                    
            except Exception as e:
                if training_id in training_processes:
                    training_processes[training_id]['status'] = 'failed'
                    training_processes[training_id]['output'].append(f"Error: {str(e)}")
        
        thread = threading.Thread(target=run_training)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'training_id': training_id,
            'message': 'Training started',
            'start_time': timestamp
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training-status/<training_id>', methods=['GET'])
def training_status(training_id):
    """获取训练状态和输出"""
    if training_id not in training_processes:
        return jsonify({'error': 'Training process not found'}), 404
    
    process_info = training_processes[training_id]
    return jsonify({
        'status': process_info['status'],
        'output': process_info['output'],
        'start_time': process_info.get('start_time')
    }), 200

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