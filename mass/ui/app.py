#!/usr/bin/env python3
"""
Web server for MASS (Metric Analytic Super System) UI
Provides web interface for config building, running analytics, and viewing reports
"""

import os
import sys
import json
import subprocess
import glob
from datetime import datetime
from pathlib import Path
from flask import Flask, send_file, send_from_directory, request, jsonify, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
# Check in multiple locations: ui/, project root
env_paths = [
    Path(__file__).parent / '.env',
    Path(__file__).parent.parent.parent / '.env',
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
        break

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Get paths
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent
CONFIGS_DIR = PROJECT_ROOT / 'configs'
DRY_RUN_OUTPUT_DIR = PROJECT_ROOT / 'dry_run_output'
MASS_CORE_DIR = PROJECT_ROOT / 'mass' / 'core'


@app.route('/')
def index():
    """Serve the main UI page"""
    return send_file('index.html')


@app.route('/favicon.svg')
def favicon_svg():
    """Serve SVG favicon"""
    svg_content = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="100" height="100" rx="20" fill="url(#grad)"/>
  <text x="50" y="70" font-family="Arial, sans-serif" font-size="60" font-weight="bold" fill="white" text-anchor="middle">M</text>
</svg>'''
    return svg_content, 200, {'Content-Type': 'image/svg+xml'}


@app.route('/favicon.ico')
def favicon_ico():
    """Serve ICO favicon (returns SVG for compatibility)"""
    svg_content = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="100" height="100" rx="20" fill="url(#grad)"/>
  <text x="50" y="70" font-family="Arial, sans-serif" font-size="60" font-weight="bold" fill="white" text-anchor="middle">M</text>
</svg>'''
    return svg_content, 200, {'Content-Type': 'image/svg+xml'}


@app.route('/api/configs', methods=['GET'])
def list_configs():
    """List all available config files"""
    configs = []
    if CONFIGS_DIR.exists():
        # Get both .yaml and .yml files
        for pattern in ['*.yaml', '*.yml']:
            for config_file in CONFIGS_DIR.glob(pattern):
                configs.append({
                    'name': config_file.name,
                    'path': str(config_file.relative_to(PROJECT_ROOT))
                })
    # Sort by name
    configs.sort(key=lambda x: x['name'])
    return jsonify({'configs': configs})


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get config file content by path"""
    config_path = request.args.get('path')
    if not config_path:
        return jsonify({'error': 'path parameter is required'}), 400
    
    # Resolve config path
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    
    # Security: ensure path is within project root
    try:
        config_path = os.path.abspath(config_path)
        project_root_abs = os.path.abspath(PROJECT_ROOT)
        if not config_path.startswith(project_root_abs):
            return jsonify({'error': 'Invalid path'}), 400
    except Exception:
        return jsonify({'error': 'Invalid path'}), 400
    
    if not os.path.exists(config_path):
        return jsonify({'error': 'Config file not found'}), 404
    
    # Only allow YAML files
    if not (config_path.endswith('.yaml') or config_path.endswith('.yml')):
        return jsonify({'error': 'Only YAML files are allowed'}), 400
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500


@app.route('/api/run', methods=['POST'])
def run_analytics():
    """Run analytics job with given config"""
    try:
        data = request.json
        config_path = data.get('config_path')
        dry_run = data.get('dry_run', True)
        event_deepness = data.get('event_deepness')
        
        if not config_path:
            return jsonify({'error': 'config_path is required'}), 400
        
        # Resolve config path
        if not os.path.isabs(config_path):
            config_path = os.path.join(PROJECT_ROOT, config_path)
        
        if not os.path.exists(config_path):
            return jsonify({'error': f'Config file not found: {config_path}'}), 404
        
        # Build command - use mass.core.analytics_job module
        cmd = [
            sys.executable,
            '-m', 'mass.core.analytics_job',
            config_path
        ]
        
        if dry_run:
            cmd.append('--dry-run')
        
        if event_deepness:
            cmd.extend(['--event-deepness', event_deepness])
        
        # Prepare environment variables for subprocess
        # Copy current environment and ensure YDB credentials are passed
        env = os.environ.copy()
        
        # Ensure CI_YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS is available
        # Priority: 1) Already in env, 2) CI_YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS_FILE points to file, 3) Try common locations
        if 'CI_YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS' not in env or not env['CI_YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS']:
            # Try to read from file if specified
            creds_file = env.get('CI_YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS_FILE')
            if creds_file and os.path.exists(creds_file):
                with open(creds_file, 'r') as f:
                    env['CI_YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS'] = f.read().strip()
            # If still not found, try to read from common credential file locations
            elif not env.get('CI_YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS'):
                common_cred_paths = [
                    os.path.expanduser('~/.ydb/credentials.json'),
                    os.path.expanduser('~/ydb_credentials.json'),
                    '/etc/ydb/credentials.json',
                ]
                for cred_path in common_cred_paths:
                    if os.path.exists(cred_path):
                        env['CI_YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS'] = cred_path
                        break
        
        # Check if credentials are still missing and provide helpful error
        if 'CI_YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS' not in env or not env['CI_YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS']:
            error_msg = (
                "YDB credentials not found. Please set CI_YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS environment variable.\n"
                "Options:\n"
                "1. Create .env file in mass/ui/ directory with: CI_YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS=/path/to/credentials.json\n"
                "2. Export before running: export CI_YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS=/path/to/credentials.json\n"
                "3. Set CI_YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS_FILE=/path/to/credentials.json to auto-load from file"
            )
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Run analytics job with environment variables
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,  # Pass environment variables
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            return jsonify({
                'success': False,
                'error': result.stderr,
                'output': result.stdout
            }), 500
        
        # Find generated reports
        reports = find_reports()
        
        return jsonify({
            'success': True,
            'output': result.stdout,
            'reports': reports
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Analytics job timed out'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports', methods=['GET'])
def list_reports():
    """List all available reports"""
    reports = find_reports()
    return jsonify({'reports': reports})


def find_reports():
    """Find summary HTML reports in dry_run_output directory"""
    reports = []
    if DRY_RUN_OUTPUT_DIR.exists():
        # Find only summary reports (pattern: *_summary_*.html)
        # Summary reports have "_summary_" in the name
        # Detail reports have "_event_" (singular) and are excluded
        for report_file in DRY_RUN_OUTPUT_DIR.glob('*_summary_*.html'):
            stat = report_file.stat()
            reports.append({
                'name': report_file.name,
                'type': 'summary',
                'path': f'/api/report/{report_file.name}',
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    
    # Sort by modified time (newest first)
    reports.sort(key=lambda x: x['modified'], reverse=True)
    return reports


@app.route('/api/report/<filename>')
def get_report(filename):
    """Serve a report file"""
    # Security: only allow HTML files
    if not filename.endswith('.html'):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Security: prevent directory traversal
    if '..' in filename or '/' in filename:
        return jsonify({'error': 'Invalid filename'}), 400
    
    # Try multiple possible locations for report files
    possible_paths = [
        DRY_RUN_OUTPUT_DIR / filename,  # Standard location
        PROJECT_ROOT / 'mass' / 'ui' / 'dry_run_output' / filename,  # UI location
        Path.cwd() / 'dry_run_output' / filename,  # Current working directory
        Path.cwd() / 'mass' / 'ui' / 'dry_run_output' / filename,  # UI from cwd
    ]
    
    report_path = None
    for path in possible_paths:
        if path.exists() and path.is_file():
            report_path = path
            break
    
    if report_path is None:
        return jsonify({'error': 'Report not found'}), 404
    
    return send_file(str(report_path))


@app.route('/api/viz-data/<filename>')
def get_viz_data(filename):
    """Download visualization data (JSON) for debugging and test creation"""
    # Security: only allow JSON files with _data.json suffix
    if not filename.endswith('_data.json'):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Security: prevent directory traversal
    if '..' in filename or '/' in filename:
        return jsonify({'error': 'Invalid filename'}), 400
    
    # Try multiple possible locations for data files
    possible_paths = [
        DRY_RUN_OUTPUT_DIR / filename,  # Standard location
        PROJECT_ROOT / 'mass' / 'ui' / 'dry_run_output' / filename,  # UI location
        Path.cwd() / 'dry_run_output' / filename,  # Current working directory
        Path.cwd() / 'mass' / 'ui' / 'dry_run_output' / filename,  # UI from cwd
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists() and path.is_file():
            data_path = path
            break
    
    if data_path is None:
        return jsonify({'error': 'Data file not found'}), 404
    
    return send_file(
        str(data_path),
        mimetype='application/json',
        as_attachment=True,
        download_name=filename
    )


@app.route('/api/save-config', methods=['POST'])
def save_config():
    """Save a config file"""
    try:
        data = request.json
        filename = data.get('filename')
        content = data.get('content')
        config = data.get('config')  # Support saving from config dict
        
        # If config dict is provided, convert to YAML
        if config and not content:
            import yaml
            content = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        if not filename or not content:
            return jsonify({'error': 'filename and content (or config) are required'}), 400
        
        # Security: only allow YAML files
        if not filename.endswith('.yaml') and not filename.endswith('.yml'):
            return jsonify({'error': 'Only YAML files are allowed'}), 400
        
        # Security: prevent directory traversal
        if '..' in filename or '/' in filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        # Save to configs directory
        config_path = CONFIGS_DIR / filename
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': str(config_path.relative_to(PROJECT_ROOT))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/saved-data', methods=['GET'])
def list_saved_data():
    """List all saved data files"""
    try:
        saved_data_dir = PROJECT_ROOT / 'saved_data'
        saved_files = []
        
        if saved_data_dir.exists():
            # Find all data files (parquet, csv, pkl)
            for pattern in ['*.parquet', '*.csv', '*.pkl', '*.pickle']:
                for data_file in saved_data_dir.glob(pattern):
                    stat = data_file.stat()
                    saved_files.append({
                        'name': data_file.name,
                        'path': str(data_file.relative_to(PROJECT_ROOT)),
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        # Sort by modified time (newest first)
        saved_files.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            'success': True,
            'files': saved_files
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load and save data from data source"""
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400
        
        data = request.get_json()
        base_config_path = data.get('base_config_path')
        
        if not base_config_path:
            return jsonify({
                'success': False,
                'error': 'base_config_path is required'
            }), 400
        
        # Resolve config path
        if not os.path.isabs(base_config_path):
            base_config_path = os.path.join(PROJECT_ROOT, base_config_path)
        
        # Security: ensure path is within project root
        try:
            base_config_path = os.path.abspath(base_config_path)
            project_root_abs = os.path.abspath(PROJECT_ROOT)
            if not base_config_path.startswith(project_root_abs):
                return jsonify({
                    'success': False,
                    'error': 'Invalid path'
                }), 400
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid path: {str(e)}'
            }), 400
        
        if not os.path.exists(base_config_path):
            return jsonify({
                'success': False,
                'error': f'Config file not found: {base_config_path}'
            }), 404
        
        # Only allow YAML files
        if not (base_config_path.endswith('.yaml') or base_config_path.endswith('.yml')):
            return jsonify({
                'success': False,
                'error': 'Only YAML files are allowed'
            }), 400
        
        # Import exploration module
        from mass.core.exploration import ExplorationRunner
        
        # Create exploration runner
        runner = ExplorationRunner(base_config_path)
        
        # Load and save data
        saved_path = runner.load_and_save_data()
        
        # Return relative path from project root
        saved_path_rel = os.path.relpath(saved_path, PROJECT_ROOT)
        
        return jsonify({
            'success': True,
            'data_file': saved_path_rel,
            'message': f'Data saved to {saved_path_rel}'
        })
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


@app.route('/api/explore', methods=['POST'])
def run_exploration():
    """Run analytics exploration with multiple configuration variants"""
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Empty request body'
            }), 400
        
        base_config_path = data.get('base_config_path')
        variants = data.get('variants', {})
        
        if not base_config_path:
            return jsonify({
                'success': False,
                'error': 'base_config_path is required'
            }), 400
        
        # Resolve config path
        if not os.path.isabs(base_config_path):
            base_config_path = os.path.join(PROJECT_ROOT, base_config_path)
        
        # Security: ensure path is within project root
        try:
            base_config_path = os.path.abspath(base_config_path)
            project_root_abs = os.path.abspath(PROJECT_ROOT)
            if not base_config_path.startswith(project_root_abs):
                return jsonify({
                    'success': False,
                    'error': 'Invalid path'
                }), 400
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid path: {str(e)}'
            }), 400
        
        if not os.path.exists(base_config_path):
            return jsonify({
                'success': False,
                'error': f'Config file not found: {base_config_path}'
            }), 404
        
        # Only allow YAML files
        if not (base_config_path.endswith('.yaml') or base_config_path.endswith('.yml')):
            return jsonify({
                'success': False,
                'error': 'Only YAML files are allowed'
            }), 400
        
        # Import exploration module
        from mass.core.exploration import ExplorationRunner
        
        # Get optional parameters
        load_data_first = data.get('load_data_first', False)
        data_file = data.get('data_file')  # Optional path to existing saved data file
        
        # Resolve data_file path if provided
        if data_file:
            if not os.path.isabs(data_file):
                data_file = os.path.join(PROJECT_ROOT, data_file)
            # Security: ensure path is within project root
            try:
                data_file = os.path.abspath(data_file)
                project_root_abs = os.path.abspath(PROJECT_ROOT)
                if not data_file.startswith(project_root_abs):
                    return jsonify({
                        'success': False,
                        'error': 'Invalid data_file path'
                    }), 400
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Invalid data_file path: {str(e)}'
                }), 400
        
        # Create exploration runner
        runner = ExplorationRunner(base_config_path, data_file=data_file)
        
        # Run exploration
        results = runner.run_exploration(
            variant_settings=variants,
            dry_run=True,  # Always use dry_run for exploration
            max_workers=4,  # Limit parallel workers
            load_data_first=load_data_first,
            data_file=data_file
        )
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


@app.route('/api/explore/auto_tune', methods=['POST'])
def run_auto_tune():
    """Автоматический подбор параметров и обнаружение событий"""
    try:
        data = request.get_json()
        
        # Получаем конфигурацию
        config_content = data.get('config')
        if not config_content:
            return jsonify({
                'success': False,
                'error': 'Configuration is required'
            }), 400
        
        # Сохраняем временный конфиг
        import tempfile
        import yaml
        
        temp_config_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False, dir=os.path.join(PROJECT_ROOT, 'configs')
        )
        yaml.dump(yaml.safe_load(config_content), temp_config_file, default_flow_style=False, allow_unicode=True)
        temp_config_path = temp_config_file.name
        temp_config_file.close()
        
        try:
            # Получаем опциональные параметры
            data_file = data.get('data_file')
            tuning_methods = data.get('tuning_methods', ['adaptive'])  # По умолчанию adaptive
            compare_methods = data.get('compare_methods', False)  # Сравнение методов
            
            # Resolve data_file path if provided
            if data_file:
                if not os.path.isabs(data_file):
                    data_file = os.path.join(PROJECT_ROOT, data_file)
                # Security: ensure path is within project root
                try:
                    data_file = os.path.abspath(data_file)
                    project_root_abs = os.path.abspath(PROJECT_ROOT)
                    if not data_file.startswith(project_root_abs):
                        return jsonify({
                            'success': False,
                            'error': 'Invalid data_file path'
                        }), 400
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': f'Invalid data_file path: {str(e)}'
                    }), 400
            
            # Создаем exploration runner
            from mass.core.exploration import ExplorationRunner
            runner = ExplorationRunner(temp_config_path, data_file=data_file)
            
            # Запускаем auto-tune
            results = runner.run_auto_tune(
                dry_run=True,
                tuning_methods=tuning_methods,
                compare_methods=compare_methods
            )
            
            return jsonify({
                'success': True,
                'results': results
            })
        finally:
            # Удаляем временный конфиг
            try:
                os.unlink(temp_config_path)
            except:
                pass
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MASS (Metric Analytic Super System) UI Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print(f"Starting MASS (Metric Analytic Super System) UI server on http://{args.host}:{args.port}")
    print(f"Open http://{args.host}:{args.port} in your browser")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
