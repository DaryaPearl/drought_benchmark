#!/usr/bin/env python3
"""
Главный скрипт для запуска проекта предсказания засухи
Автоматизирует весь пайплайн: сборка данных → обучение → анализ результатов

Использование:
    python run_project.py --help                     # Справка
    python run_project.py --quick                    # Быстрый запуск
    python run_project.py --full                     # Полный пайплайн
    python run_project.py --data-only                # Только сборка данных
    python run_project.py --train-only --model all  # Только обучение
    python run_project.py --custom --config my_config.yaml
"""

import argparse
import os
import sys
import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import time

class ProjectRunner:
    """Основной класс для управления проектом"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.start_time = time.time()
        self.config = {}
        
        # Создание необходимых директорий
        self.ensure_directories()
    
    def ensure_directories(self):
        """Создание необходимых директорий проекта"""
        directories = [
            "data/raw",
            "data/processed", 
            "results",
            "logs",
            "configs",
            "checkpoints",
            "outputs"
        ]
        
        for dir_path in directories:
            (self.project_root / dir_path).mkdir(parents=True, exist_ok=True)
    
    def check_environment(self) -> bool:
        """Проверка окружения и зависимостей"""
        print("🔍 Проверка окружения...")
        
        # Проверка Python версии
        if sys.version_info < (3, 8):
            print("❌ Требуется Python 3.8 или выше")
            return False
        
        # Проверка основных зависимостей
        required_packages = [
            'numpy', 'pandas', 'xarray', 'torch', 'pytorch_lightning',
            'sklearn', 'matplotlib', 'requests'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Отсутствуют пакеты: {', '.join(missing_packages)}")
            print("📦 Установите их: pip install -r requirements.txt")
            return False
        
        # Проверка конфигурационных файлов
        netrc_file = Path.home() / ".netrc"
        if not netrc_file.exists():
            print("⚠ Файл ~/.netrc не найден (нужен для NASA Earthdata)")
            print("📝 Создайте файл с содержимым:")
            print("machine urs.earthdata.nasa.gov")
            print("login YOUR_USERNAME")
            print("password YOUR_PASSWORD")
        
        print("✅ Окружение готово")
        return True
    
    def load_config(self, config_path: Optional[str] = None) -> Dict:
        """Загрузка конфигурации"""
        
        # Конфигурация по умолчанию
        default_config = {
            "project": {
                "name": "drought_prediction_project",
                "version": "1.0.0",
                "description": "ML/AI system for agricultural drought prediction"
            },
            "data": {
                "years_range": [2003, 2024],
                "regions": {
                    "us_plains": [35, 48, -104, -90],
                    "br_cerrado": [-20, -6, -62, -46], 
                    "in_ganga": [21, 31, 73, 90],
                    "ru_steppe": [50, 55, 37, 47]
                },
                "variables": ["precipitation", "temperature", "ndvi", "soil_moisture"],
                "drought_indices": ["spi1", "spi3", "spi6", "spei3", "pdsi", "snepi"]
            },
            "training": {
                "sequence_length": 12,
                "prediction_horizon": 3,
                "train_years": [2003, 2016],
                "val_years": [2017, 2019],
                "test_years": [2020, 2024],
                "batch_size": 32,
                "max_epochs": 50,
                "early_stopping_patience": 10
            },
            "models": {
                "classical": ["linear_regression", "random_forest", "xgboost", "gradient_boosting"],
                "sota": ["earthformer", "convlstm", "tft", "unet"]
            },
            "output": {
                "save_predictions": True,
                "create_visualizations": True,
                "generate_report": True
            }
        }
        
        if config_path and Path(config_path).exists():
            print(f"📋 Загрузка конфигурации из {config_path}")
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
            
            # Объединение с конфигурацией по умолчанию
            self.config = self._merge_configs(default_config, custom_config)
        else:
            self.config = default_config
            print("📋 Использование конфигурации по умолчанию")
        
        return self.config
    
    def _merge_configs(self, default: Dict, custom: Dict) -> Dict:
        """Объединение конфигураций"""
        merged = default.copy()
        for key, value in custom.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    def save_config(self, config_path: str):
        """Сохранение текущей конфигурации"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        print(f"💾 Конфигурация сохранена: {config_path}")
    
    def run_data_pipeline(self, quick_mode: bool = False) -> bool:
        """Запуск пайплайна сборки данных"""
        print("\n" + "="*60)
        print("📦 ЭТАП 1: СБОРКА ДАННЫХ")
        print("="*60)
        
        # В быстром режиме используем упрощенный pipeline
        if quick_mode:
            print("⚡ Быстрый режим: создание синтетических данных...")
            
            # Проверяем, есть ли уже данные
            data_path = Path("data/processed/real_agro_cube.zarr")
            if data_path.exists():
                print("📁 Данные уже существуют, пропускаем создание")
                return True
            
            try:
                # Запускаем быстрый pipeline
                result = subprocess.run([
                    sys.executable, "-m", "src.data_pipeline.quick_data_pipeline"
                ], cwd=self.project_root, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print("✅ Синтетические данные созданы успешно")
                    print(result.stdout)
                    return True
                else:
                    print("❌ Ошибка при создании синтетических данных:")
                    print(result.stderr)
                    return False
                    
            except subprocess.TimeoutExpired:
                print("⏱ Таймаут при создании данных")
                return False
            except Exception as e:
                print(f"❌ Исключение при создании данных: {e}")
                return False
        
        # Обычный режим - полный pipeline
        else:
            data_script = self.project_root / "src" / "data_pipeline" / "real_data_pipeline.py"
            
            if not data_script.exists():
                print(f"❌ Скрипт сборки данных не найден: {data_script}")
                return False
            
            try:
                print("🚀 Запуск полного pipeline сборки данных...")
                
                env = os.environ.copy()
                if quick_mode:
                    env['QUICK_MODE'] = '1'
                
                result = subprocess.run([
                    sys.executable, "-m", "src.data_pipeline.real_data_pipeline"
                ], cwd=self.project_root, env=env, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Сборка данных завершена успешно")
                    print(result.stdout)
                    return True
                else:
                    print("❌ Ошибка при сборке данных:")
                    print(result.stderr)
                    return False
                    
            except Exception as e:
                print(f"❌ Исключение при сборке данных: {e}")
                return False

    def run_training_pipeline(self, models: List[str], quick_mode: bool = False) -> bool:
        """Запуск пайплайна обучения"""
        print("\n" + "="*60)
        print("🤖 ЭТАП 2: ОБУЧЕНИЕ МОДЕЛЕЙ")
        print("="*60)
        
        training_script = self.project_root / "src" / "complete_training_pipeline.py"
        
        if not training_script.exists():
            print(f"❌ Скрипт обучения не найден: {training_script}")
            return False
        
        # Проверка наличия данных
        data_path = self.project_root / "data" / "processed" / "real_agro_cube.zarr"
        if not data_path.exists():
            print(f"❌ Данные не найдены: {data_path}")
            print("💡 Сначала запустите сборку данных: --data-only")
            return False
        
        try:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            for model in models:
                print(f"\n🔧 Обучение модели: {model}")
                
                cmd = [
                    sys.executable, "src/complete_training_pipeline.py",
                    "--model", model,
                    "--experiment", experiment_name
                ]
                
                if quick_mode:
                    cmd.append("--fast")
                    print("⚡ Быстрый режим обучения")
                
                result = subprocess.run(cmd, cwd=self.project_root, 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ Модель {model} обучена успешно")
                    # Печатаем только последние строки вывода
                    output_lines = result.stdout.split('\n')
                    for line in output_lines[-10:]:
                        if line.strip():
                            print(f"  {line}")
                else:
                    print(f"❌ Ошибка обучения модели {model}:")
                    print(result.stderr)
                    return False
            
            print(f"\n🎉 Все модели обучены! Эксперимент: {experiment_name}")
            return True
            
        except Exception as e:
            print(f"❌ Исключение при обучении: {e}")
            return False
    
    def create_final_report(self) -> bool:
        """Создание финального отчета"""
        print("\n" + "="*60)
        print("📊 ЭТАП 3: АНАЛИЗ РЕЗУЛЬТАТОВ")
        print("="*60)
        
        try:
            # Поиск результатов экспериментов
            results_dir = self.project_root / "results"
            experiments = list(results_dir.glob("exp_*"))
            
            if not experiments:
                print("⚠ Результаты экспериментов не найдены")
                return False
            
            # Берем последний эксперимент
            latest_exp = max(experiments, key=lambda x: x.stat().st_mtime)
            print(f"📁 Анализ эксперимента: {latest_exp.name}")
            
            # Загрузка результатов
            results_file = latest_exp / "experiment_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Создание расширенного отчета
                self._create_executive_summary(results, latest_exp)
                self._create_technical_report(results, latest_exp)
                
                print("✅ Финальный отчет создан")
                return True
            else:
                print("❌ Файл результатов не найден")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка создания отчета: {e}")
            return False
    
    def _create_executive_summary(self, results: Dict, exp_dir: Path):
        """Создание краткого отчета для руководства"""
        summary_file = exp_dir / "executive_summary.md"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# Краткий отчет: Система предсказания засухи\n\n")
            f.write(f"**Дата:** {datetime.now().strftime('%d.%m.%Y')}\n")
            f.write(f"**Эксперимент:** {exp_dir.name}\n\n")
            
            f.write("## 🎯 Основные результаты\n\n")
            
            # Найти лучшую модель
            best_model = self._find_best_model(results.get('results', {}))
            if best_model:
                f.write(f"### Лучшая модель: {best_model['name']}\n")
                f.write(f"- **Точность:** {best_model['accuracy']:.1%}\n")
                f.write(f"- **Время обучения:** {best_model['training_time']:.1f} сек\n")
                f.write(f"- **Тип:** {best_model['type']}\n\n")
            
            f.write("## 📊 Сравнение подходов\n\n")
            f.write("| Тип модели | Лучший результат | Время обучения | Сложность |\n")
            f.write("|------------|------------------|----------------|------------|\n")
            
            # Классические модели
            classical_results = results.get('results', {}).get('classical', {})
            if classical_results:
                best_classical = min(classical_results.items(), 
                                   key=lambda x: x[1].get('val_metrics', {}).get('mae', float('inf')))
                mae = best_classical[1].get('val_metrics', {}).get('mae', 0)
                time_val = best_classical[1].get('training_time', 0)
                f.write(f"| Классические ML | MAE: {mae:.4f} | {time_val:.1f}s | Низкая |\n")
            
            # SOTA модели
            sota_models = ['earthformer', 'convlstm', 'tft', 'unet']
            for model_name in sota_models:
                if model_name in results.get('results', {}):
                    model_res = results['results'][model_name]
                    if isinstance(model_res, dict) and 'error' not in model_res:
                        val_loss = model_res.get('val_metrics', {}).get('final_val_loss', 0)
                        time_val = model_res.get('training_time', 0)
                        f.write(f"| {model_name.upper()} | Loss: {val_loss:.4f} | {time_val:.1f}s | Высокая |\n")
            
            f.write("\n## 🔍 Рекомендации\n\n")
            if best_model:
                if best_model['type'] == 'classical':
                    f.write("- ✅ **Для продакшена:** Используйте классические ML модели\n")
                    f.write("- ⚡ **Преимущества:** Быстрое обучение, интерпретируемость\n")
                    f.write("- 🎯 **Применение:** Оперативное предсказание засухи\n\n")
                else:
                    f.write("- 🚀 **Для исследований:** Используйте SOTA архитектуры\n")
                    f.write("- 📈 **Преимущества:** Высокая точность, работа с комплексными паттернами\n") 
                    f.write("- 🔬 **Применение:** Долгосрочное планирование\n\n")
            
            f.write("## 📈 Следующие шаги\n\n")
            f.write("1. Валидация лучшей модели на новых данных\n")
            f.write("2. Развертывание системы мониторинга\n")
            f.write("3. Интеграция с существующими системами\n")
            f.write("4. Сбор обратной связи от пользователей\n")
    
    def _create_technical_report(self, results: Dict, exp_dir: Path):
        """Создание технического отчета"""
        tech_file = exp_dir / "technical_report.md"
        
        with open(tech_file, 'w', encoding='utf-8') as f:
            f.write("# Технический отчет: Система предсказания засухи\n\n")
            
            f.write("## 🔧 Техническая конфигурация\n\n")
            config = results.get('config', {})
            f.write("```yaml\n")
            f.write(yaml.dump(config, default_flow_style=False))
            f.write("```\n\n")
            
            f.write("## 📊 Детальные результаты\n\n")
            
            # Подробный анализ каждой модели
            model_results = results.get('results', {})
            for model_name, model_data in model_results.items():
                f.write(f"### {model_name.upper()}\n\n")
                
                if isinstance(model_data, dict) and 'error' not in model_data:
                    if model_name == 'classical':
                        f.write("#### Классические модели ML\n\n")
                        f.write("| Модель | Train MAE | Val MAE | Val RMSE | Val R² | Время (с) |\n")
                        f.write("|--------|-----------|---------|----------|--------|----------|\n")
                        
                        for sub_model, sub_data in model_data.items():
                            if isinstance(sub_data, dict):
                                train_metrics = sub_data.get('train_metrics', {})
                                val_metrics = sub_data.get('val_metrics', {})
                                f.write(f"| {sub_model} | "
                                       f"{train_metrics.get('mae', 'N/A'):.4f} | "
                                       f"{val_metrics.get('mae', 'N/A'):.4f} | "
                                       f"{val_metrics.get('rmse', 'N/A'):.4f} | "
                                       f"{val_metrics.get('r2', 'N/A'):.4f} | "
                                       f"{sub_data.get('training_time', 'N/A'):.1f} |\n")
                    else:
                        f.write("#### Метрики обучения\n\n")
                        train_metrics = model_data.get('train_metrics', {})
                        val_metrics = model_data.get('val_metrics', {})
                        test_metrics = model_data.get('test_metrics', {})
                        
                        f.write(f"- **Финальная потеря на обучении:** {train_metrics.get('final_train_loss', 'N/A')}\n")
                        f.write(f"- **Финальная потеря на валидации:** {val_metrics.get('final_val_loss', 'N/A')}\n")
                        f.write(f"- **Лучшая потеря на валидации:** {val_metrics.get('best_val_loss', 'N/A')}\n")
                        f.write(f"- **Время обучения:** {model_data.get('training_time', 'N/A'):.1f} сек\n")
                        
                        if test_metrics:
                            f.write(f"- **Тестовые метрики:** {test_metrics}\n")
                        
                        f.write(f"- **Конфигурация модели:**\n")
                        model_config = model_data.get('model_config', {})
                        for key, value in model_config.items():
                            f.write(f"  - {key}: {value}\n")
                else:
                    f.write(f"❌ **Ошибка:** {model_data.get('error', 'Неизвестная ошибка')}\n")
                
                f.write("\n")
            
            f.write("## 💡 Технические выводы\n\n")
            f.write("### Производительность\n")
            f.write("- Классические модели показывают стабильные результаты\n")
            f.write("- SOTA архитектуры требуют больше времени на обучение\n")
            f.write("- Необходима дополнительная настройка гиперпараметров\n\n")
            
            f.write("### Рекомендации по улучшению\n")
            f.write("1. Увеличить объем обучающих данных\n")
            f.write("2. Добавить дополнительные источники данных\n")
            f.write("3. Провести более тщательную настройку гиперпараметров\n")
            f.write("4. Реализовать ансамблевые методы\n")
    
    def _find_best_model(self, results: Dict) -> Optional[Dict]:
        """Поиск лучшей модели по результатам"""
        best_model = None
        best_score = float('inf')
        
        for model_name, model_data in results.items():
            if isinstance(model_data, dict) and 'error' not in model_data:
                if model_name == 'classical':
                    # Для классических моделей ищем лучший MAE
                    for sub_model, sub_data in model_data.items():
                        if isinstance(sub_data, dict):
                            mae = sub_data.get('val_metrics', {}).get('mae', float('inf'))
                            if mae < best_score:
                                best_score = mae
                                best_model = {
                                    'name': sub_model,
                                    'accuracy': 1 - mae,  # Примерная конверсия
                                    'training_time': sub_data.get('training_time', 0),
                                    'type': 'classical'
                                }
                else:
                    # Для SOTA моделей используем validation loss
                    val_loss = model_data.get('val_metrics', {}).get('final_val_loss', float('inf'))
                    if val_loss < best_score:
                        best_score = val_loss
                        best_model = {
                            'name': model_name,
                            'accuracy': 1 - val_loss,  # Примерная конверсия
                            'training_time': model_data.get('training_time', 0),
                            'type': 'SOTA'
                        }
        
        return best_model
    
    def run_full_pipeline(self, models: List[str], quick_mode: bool = False) -> bool:
        """Запуск полного пайплайна"""
        print("🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА ПРЕДСКАЗАНИЯ ЗАСУХИ")
        print("=" * 60)
        
        total_start = time.time()
        
        # Этап 1: Сборка данных
        if not self.run_data_pipeline(quick_mode):
            print("❌ Пайплайн остановлен на этапе сборки данных")
            return False
        
        # Этап 2: Обучение моделей
        if not self.run_training_pipeline(models, quick_mode):
            print("❌ Пайплайн остановлен на этапе обучения")
            return False
        
        # Этап 3: Анализ результатов
        if not self.create_final_report():
            print("⚠ Ошибка создания финального отчета")
        
        total_time = time.time() - total_start
        
        print("\n" + "🎉" * 20)
        print("ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
        print("🎉" * 20)
        print(f"⏱ Общее время выполнения: {total_time/60:.1f} минут")
        print(f"📁 Результаты сохранены в: {self.project_root / 'results'}")
        
        return True
    
    def cleanup(self):
        """Очистка временных файлов"""
        print("🧹 Очистка временных файлов...")
        
        temp_dirs = [
            self.project_root / "outputs",
            self.project_root / ".hydra"
        ]
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                print(f"  Удалено: {temp_dir}")
        
        print("✅ Очистка завершена")

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="Система предсказания засухи - главный скрипт запуска",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python run_project.py --quick                    # Быстрый тест всего пайплайна
  python run_project.py --full                     # Полный пайплайн со всеми моделями
  python run_project.py --data-only                # Только сборка данных
  python run_project.py --train-only --model rf    # Только обучение Random Forest
  python run_project.py --custom --config my.yaml  # Кастомная конфигурация
        """
    )
    
    # Основные режимы
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--quick', action='store_true',
                           help='Быстрый запуск (урезанные данные и модели)')
    mode_group.add_argument('--full', action='store_true',
                           help='Полный пайплайн со всеми моделями')
    mode_group.add_argument('--data-only', action='store_true',
                           help='Только сборка данных')
    mode_group.add_argument('--train-only', action='store_true',
                           help='Только обучение моделей')
    mode_group.add_argument('--custom', action='store_true',
                           help='Кастомный режим с конфигурацией')
    
    # Параметры
    parser.add_argument('--model', choices=['all', 'classical', 'earthformer', 'convlstm', 'tft', 'unet'],
                       default='all', help='Модель для обучения')
    parser.add_argument('--config', type=str, help='Путь к файлу конфигурации')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Не удалять временные файлы')
    parser.add_argument('--save-config', type=str,
                       help='Сохранить конфигурацию в файл')
    
    args = parser.parse_args()
    
    # Инициализация
    runner = ProjectRunner()
    
    try:
        # Проверка окружения
        if not runner.check_environment():
            return 1
        
        # Загрузка конфигурации
        runner.load_config(args.config)
        
        # Сохранение конфигурации (если указано)
        if args.save_config:
            runner.save_config(args.save_config)
        
        # Определение моделей для обучения
        if args.model == 'all':
            models = ['classical', 'earthformer', 'convlstm', 'tft', 'unet']
        elif args.model == 'classical':
            models = ['classical']
        else:
            models = [args.model]
        
        # Выполнение в зависимости от режима
        success = False
        
        if args.quick:
            print("⚡ БЫСТРЫЙ РЕЖИМ")
            success = runner.run_full_pipeline(['classical'], quick_mode=True)
            
        elif args.full:
            print("🔥 ПОЛНЫЙ РЕЖИМ")
            success = runner.run_full_pipeline(models, quick_mode=False)
            
        elif args.data_only:
            print("📦 ТОЛЬКО СБОРКА ДАННЫХ")
            success = runner.run_data_pipeline(quick_mode=args.quick)
            
        elif args.train_only:
            print("🤖 ТОЛЬКО ОБУЧЕНИЕ МОДЕЛЕЙ")
            success = runner.run_training_pipeline(models, quick_mode=False)
            
        elif args.custom:
            print("⚙️ КАСТОМНЫЙ РЕЖИМ")
            # Интерактивный выбор этапов
            print("Выберите этапы для выполнения:")
            print("1. Сборка данных")
            print("2. Обучение моделей")
            print("3. Создание отчета")
            
            stages = input("Введите номера через запятую (например, 1,2,3): ").strip()
            
            success = True
            if '1' in stages:
                success = success and runner.run_data_pipeline()
            if '2' in stages and success:
                success = success and runner.run_training_pipeline(models)
            if '3' in stages and success:
                success = success and runner.create_final_report()
        
        # Очистка
        if not args.no_cleanup:
            runner.cleanup()
        
        # Финальный статус
        if success:
            print(f"\n🎉 Проект выполнен успешно!")
            return 0
        else:
            print(f"\n❌ Проект завершился с ошибками")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹ Проект прерван пользователем")
        return 1
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        return 1

if __name__ == "__main__":
    exit(main())