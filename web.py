import yaml
import os
import subprocess
import sys
from flask import Flask, send_from_directory, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import queue
import time
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置上传文件夹
UPLOAD_FOLDER = 'data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 创建figures文件夹用于存放生成的图表
FIGURES_FOLDER = 'figures'
if not os.path.exists(FIGURES_FOLDER):
    os.makedirs(FIGURES_FOLDER)
app.config['FIGURES_FOLDER'] = FIGURES_FOLDER

# 设置允许的文件类型
ALLOWED_EXTENSIONS = {'csv'}

# Store training processes
training_processes = {}
# Store inference processes
inference_processes = {}

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

@app.route('/api/models', methods=['GET'])
def list_models():
    """获取模型检查点文件列表"""
    try:
        models = []
        # 从 checkpoints 目录查找模型文件
        checkpoints_dir = 'checkpoints'
        if os.path.exists(checkpoints_dir):
            for filename in os.listdir(checkpoints_dir):
                if filename.endswith('.pt'):
                    filepath = os.path.join(checkpoints_dir, filename)
                    if os.path.isfile(filepath):
                        file_stats = os.stat(filepath)
                        models.append({
                            'name': filename,
                            'size': file_stats.st_size,
                            'modified': file_stats.st_mtime
                        })
        return jsonify(models), 200
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

@app.route('/api/start-inference', methods=['POST'])
def start_inference():
    """启动推理过程"""
    try:
        data = request.get_json()
        model_name = data.get('model')
        standard = data.get('standard')
        obj = data.get('obj')
        
        if not model_name or not standard or not obj:
            return jsonify({'error': 'Missing model, standard or object'}), 400
            
        # Check if model exists in checkpoints directory
        model_path = os.path.join('checkpoints', model_name)
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found'}), 404
        
        # Generate a unique ID for this inference process
        import uuid
        inference_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        # Start inference in a separate thread
        def run_inference():
            try:
                # Set environment variables for inference
                env = os.environ.copy()
                env['MODEL_PATH'] = model_path
                env['STANDARD'] = standard
                env['OBJECT'] = obj
                env['INFERENCE_ID'] = inference_id
                
                # Run inference.py as a subprocess
                process = subprocess.Popen([
                    sys.executable, '-u', 'inference.py'
                ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                   bufsize=1, universal_newlines=True, env=env)
                
                inference_processes[inference_id] = {
                    'process': process,
                    'output': [],
                    'status': 'running',
                    'start_time': timestamp
                }
                
                # Capture output
                for line in iter(process.stdout.readline, ''):
                    if inference_id in inference_processes:
                        inference_processes[inference_id]['output'].append(line)
                
                process.wait()
                if inference_id in inference_processes:
                    inference_processes[inference_id]['status'] = 'completed' if process.returncode == 0 else 'failed'
                    
                    # If successful, generate figure
                    if process.returncode == 0:
                        try:
                            generate_figure_with_timestamp(inference_id)
                        except Exception as fig_error:
                            inference_processes[inference_id]['output'].append(f"Figure generation error: {str(fig_error)}")
                            inference_processes[inference_id]['status'] = 'completed_with_errors'
                            
            except Exception as e:
                if inference_id in inference_processes:
                    inference_processes[inference_id]['status'] = 'failed'
                    inference_processes[inference_id]['output'].append(f"Error: {str(e)}")
        
        thread = threading.Thread(target=run_inference)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'inference_id': inference_id,
            'message': 'Inference started',
            'start_time': timestamp
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-figure', methods=['POST'])
def generate_figure_endpoint():
    """根据scoring_steps.json生成图表"""
    try:
        # 检查scoring_steps.json是否存在
        if not os.path.exists('scoring_steps.json'):
            return jsonify({'error': 'scoring_steps.json not found'}), 404
            
        # 生成带时间戳的图表文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        figure_filename = f"scoring_evolution_{timestamp}.png"
        figure_path = os.path.join(app.config['FIGURES_FOLDER'], figure_filename)
        
        # 导入并调用figure.py中的函数
        sys.path.append('.')  # 添加当前目录到Python路径
        import figure
        figure.generate_figure('scoring_steps.json', figure_path)
        
        return jsonify({
            'message': 'Figure generated successfully',
            'figure': figure_filename
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_figure_with_timestamp(inference_id):
    """根据推理结果生成带时间戳的图表"""
    # Read scoring_steps.json
    with open('scoring_steps.json', 'r') as f:
        data = json.load(f)
    
    # Create timestamp for the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_filename = f"scoring_evolution_{timestamp}.png"
    figure_path = os.path.join(app.config['FIGURES_FOLDER'], figure_filename)
    
    # Generate the figure using the code from figure.py
    import matplotlib.pyplot as plt
    
    steps = [step['step'] for step in data['steps']]
    scores = [step['cumulative_score'] for step in data['steps']]
    deltas = [step['delta'] for step in data['steps']]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(steps, scores, 'b-', linewidth=2)
    plt.title('Cumulative Score Evolution')
    plt.xlabel('Step')
    plt.ylabel('Score')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(steps, deltas, 'r-', linewidth=2)
    plt.title('Delta per Step')
    plt.xlabel('Step')
    plt.ylabel('Delta')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    
    # Save figure info in inference process
    if inference_id in inference_processes:
        inference_processes[inference_id]['figure'] = figure_filename

@app.route('/api/inference-status/<inference_id>', methods=['GET'])
def inference_status(inference_id):
    """获取推理状态和输出"""
    if inference_id not in inference_processes:
        return jsonify({'error': 'Inference process not found'}), 404
    
    process_info = inference_processes[inference_id]
    response_data = {
        'status': process_info['status'],
        'output': process_info['output'],
        'start_time': process_info.get('start_time')
    }
    
    # Include figure info if available
    if 'figure' in process_info:
        response_data['figure'] = process_info['figure']
    
    return jsonify(response_data), 200

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

@app.route('/figures/<path:filename>')
def serve_figure(filename):
    """提供图表文件服务"""
    return send_from_directory(app.config['FIGURES_FOLDER'], filename)

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