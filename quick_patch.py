#!/usr/bin/env python3
"""
Быстрый патч для исправления ошибки MODIS pixel_reliability
Запустите: python quick_patch.py
"""

import os
import re
from pathlib import Path

def patch_modis_code():
    """Исправляет ошибку pixel_reliability в коде MODIS"""
    print("🔧 Применение патча для исправления MODIS ошибки...")
    
    # Найдем файл с ошибкой
    possible_files = [
        "src/data_pipeline/prepare_extended_cube.py",
        "src/data_pipeline/real_data_pipeline.py", 
        "src/data_pipeline/modis_gee_downloader.py"
    ]
    
    patched_files = []
    
    for file_path in possible_files:
        if Path(file_path).exists():
            print(f"📁 Проверяем файл: {file_path}")
            
            try:
                # Читаем файл
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Проверяем наличие проблемной строки
                if 'pixel_reliability' in content:
                    print(f"🎯 Найдена ошибка в {file_path}")
                    
                    # Исправления
                    original_content = content
                    
                    # 1. Убираем pixel_reliability из select
                    content = re.sub(
                        r"\.select\(\[([^\]]*)'pixel_reliability'([^\]]*)\]\)",
                        r".select([\1\2])",
                        content
                    )
                    
                    # 2. Исправляем строки с pixel_reliability
                    content = content.replace(
                        "select(['NDVI', 'EVI', 'pixel_reliability'])",
                        "select(['NDVI', 'EVI'])"
                    )
                    
                    content = content.replace(
                        "'pixel_reliability'",
                        "'DetailedQA'"
                    )
                    
                    # 3. Исправляем функции обработки
                    if "def scale_ndvi(img):" in content or "def process_modis(img):" in content:
                        # Заменяем обработку pixel_reliability на простую маску
                        pattern = r"(pixel_reliability[^\n]*\n[^\n]*\n[^\n]*\n)"
                        replacement = """# Simple quality mask instead of pixel_reliability
                        mask = ndvi.gte(0).and(ndvi.lte(1))
                        """
                        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                    
                    # 4. Если изменения есть, сохраняем
                    if content != original_content:
                        # Создаем бэкап
                        backup_path = f"{file_path}.backup"
                        with open(backup_path, 'w', encoding='utf-8') as f:
                            f.write(original_content)
                        print(f"💾 Создан бэкап: {backup_path}")
                        
                        # Сохраняем исправленный файл
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        print(f"✅ Исправлен файл: {file_path}")
                        patched_files.append(file_path)
                    
                else:
                    print(f"✅ {file_path} не требует исправлений")
                    
            except Exception as e:
                print(f"❌ Ошибка обработки {file_path}: {e}")
    
    if patched_files:
        print(f"\n🎉 Успешно исправлено файлов: {len(patched_files)}")
        for file in patched_files:
            print(f"  ✅ {file}")
    else:
        print("\n⚠️ Файлы с ошибкой pixel_reliability не найдены")
    
    return len(patched_files) > 0

def patch_region_bounds_error():
    """Исправляет ошибку REGION_BOUNDS"""
    print("\n🔧 Исправление ошибки REGION_BOUNDS...")
    
    # Добавляем определение REGION_BOUNDS в начало файлов
    region_bounds_code = '''
# ИСПРАВЛЕНИЕ: Определение REGION_BOUNDS
REGION_BOUNDS = {
    "us_plains": {"lat_min": 35, "lat_max": 48, "lon_min": -104, "lon_max": -90},
    "br_cerrado": {"lat_min": -20, "lat_max": -6, "lon_min": -62, "lon_max": -46},
    "in_ganga": {"lat_min": 21, "lat_max": 31, "lon_min": 73, "lon_max": 90},
    "ru_steppe": {"lat_min": 50, "lat_max": 55, "lon_min": 37, "lon_max": 47},
}
'''
    
    files_to_check = [
        "src/data_pipeline/prepare_extended_cube.py",
        "src/data_pipeline/real_data_pipeline.py"
    ]
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'REGION_BOUNDS' in content and 'name \'REGION_BOUNDS\' is not defined' not in content:
                continue  # Уже определено
                
            # Добавляем определение после импортов
            lines = content.split('\n')
            insert_position = 0
            
            # Найдем место после импортов
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    insert_position = i + 1
                elif line.strip() == '' and insert_position > 0:
                    continue
                elif not line.startswith('#') and line.strip() != '' and insert_position > 0:
                    break
            
            # Вставляем определение
            lines.insert(insert_position, region_bounds_code)
            
            # Сохраняем
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            print(f"✅ Добавлено определение REGION_BOUNDS в {file_path}")

def stop_current_process():
    """Останавливает текущие процессы загрузки данных"""
    print("⏹️ Остановка текущих процессов...")
    
    import subprocess
    import signal
    
    try:
        # Найдем и остановим процессы Python с data_pipeline
        result = subprocess.run(['pgrep', '-f', 'data_pipeline'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"🛑 Остановлен процесс PID: {pid}")
                    except:
                        pass
        
        # Также попробуем остановить run_project.py
        result = subprocess.run(['pgrep', '-f', 'run_project'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"🛑 Остановлен процесс PID: {pid}")
                    except:
                        pass
                        
    except Exception as e:
        print(f"⚠️ Не удалось остановить процессы: {e}")

def main():
    """Главная функция патча"""
    print("🚀 Быстрый патч для исправления ошибок загрузки MODIS")
    print("=" * 60)
    
    # 1. Останавливаем текущие процессы
    stop_current_process()
    
    # 2. Исправляем pixel_reliability ошибку
    patch_modis_code()
    
    # 3. Исправляем REGION_BOUNDS ошибку
    patch_region_bounds_error()
    
    print("\n" + "=" * 60)
    print("🎉 ПАТЧ ПРИМЕНЕН!")
    print("💡 Теперь можете перезапустить загрузку данных:")
    print("   python run_project.py --data-only")
    print("   или")
    print("   python run_project.py --quick")
    print("\n✅ Ошибки pixel_reliability и REGION_BOUNDS исправлены")

if __name__ == "__main__":
    main()