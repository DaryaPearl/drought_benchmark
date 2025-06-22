#!/usr/bin/env python3
"""
Утилита для управления конфигурациями проекта предсказания засухи

Использование:
    python configs/config_manager.py list                    # Список доступных конфигураций
    python configs/config_manager.py validate quick_test     # Валидация конфигурации
    python configs/config_manager.py create my_config        # Создание новой конфигурации
    python configs/config_manager.py merge base quick_test   # Объединение конфигураций
    python configs/config_manager.py compare config1 config2 # Сравнение конфигураций
"""

import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
from datetime import datetime

class ConfigManager:
    """Менеджер конфигураций проекта"""
    
    def __init__(self, configs_dir: str = "configs"):
        self.configs_dir = Path(configs_dir)
        self.configs_dir.mkdir(exist_ok=True)
        
        # Доступные конфигурации
        self.available_configs = {
            'config.yaml': 'Базовая конфигурация проекта',
            'quick_test.yaml': 'Быстрый тест системы (5-10 минут)',
            'earthformer.yaml': 'Специализированная конфигурация для EarthFormer',
            'classical_only.yaml': 'Только классические ML модели',
            'research.yaml': 'Полное исследование с максимальным качеством',
            'production.yaml': 'Продакшн-готовая конфигурация',
            'gpu_optimized.yaml': 'Оптимизация для GPU',
            'multi_gpu.yaml': 'Мульти-GPU обучение',
            'cpu_only.yaml': 'CPU-only конфигурация'
        }
    
    def list_configs(self) -> None:
        """Список доступных конфигураций"""
        print("📋 Доступные конфигурации:")
        print("=" * 50)
        
        for config_name, description in self.available_configs.items():
            config_path = self.configs_dir / config_name
            status = "✅" if config_path.exists() else "❌"
            print(f"{status} {config_name:<20} - {description}")
        
        print("\n💡 Создайте отсутствующие конфигурации с помощью:")
        print("   python configs/config_manager.py create <config_name>")
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        config_path = self.configs_dir / f"{config_name}.yaml"
        if not config_path.exists():
            config_path = self.configs_dir / config_name
            if not config_path.exists():
                raise FileNotFoundError(f"Конфигурация не найдена: {config_name}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def save_config(self, config: Dict[str, Any], config_name: str) -> None:
        """Сохранение конфигурации"""
        config_path = self.configs_dir / f"{config_name}.yaml"
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)
        
        print(f"💾 Конфигурация сохранена: {config_path}")
    
    def validate_config(self, config_name: str) -> bool:
        """Валидация конфигурации"""
        print(f"🔍 Валидация конфигурации: {config_name}")
        
        try:
            config = self.load_config(config_name)
            
            # Базовые проверки
            errors = []
            warnings = []
            
            # Проверка обязательных секций
            required_sections = ['project', 'data', 'training', 'models']
            for section in required_sections:
                if section not in config:
                    errors.append(f"Отсутствует обязательная секция: {section}")
            
            # Проверка проекта
            if 'project' in config:
                if 'name' not in config['project']:
                    errors.append("Отсутствует project.name")
            
            # Проверка данных
            if 'data' in config:
                data_config = config['data']
                
                if 'years_range' in data_config:
                    years = data_config['years_range']
                    if len(years) != 2 or years[0] >= years[1]:
                        errors.append("Некорректный years_range")
                
                if 'target_variable' not in data_config:
                    warnings.append("Не указана target_variable")
            
            # Проверка обучения
            if 'training' in config:
                train_config = config['training']
                
                if 'batch_size' in train_config and train_config['batch_size'] <= 0:
                    errors.append("batch_size должен быть положительным")
                
                if 'max_epochs' in train_config and train_config['max_epochs'] <= 0:
                    errors.append("max_epochs должен быть положительным")
                
                if 'learning_rate' in train_config:
                    lr = train_config['learning_rate']
                    if lr <= 0 or lr > 1:
                        warnings.append(f"Подозрительный learning_rate: {lr}")
            
            # Проверка моделей
            if 'models' in config:
                models_config = config['models']
                
                # Проверяем что хотя бы одна модель включена
                any_enabled = False
                
                if 'classical' in models_config:
                    for model_name, model_config in models_config['classical'].items():
                        if isinstance(model_config, dict) and model_config.get('enabled', False):
                            any_enabled = True
                
                if 'sota' in models_config:
                    for model_name, model_config in models_config['sota'].items():
                        if isinstance(model_config, dict) and model_config.get('enabled', False):
                            any_enabled = True
                
                if not any_enabled:
                    errors.append("Ни одна модель не включена")
            
            # Проверка compute
            if 'compute' in config:
                compute_config = config['compute']
                
                if 'accelerator' in compute_config:
                    accelerator = compute_config['accelerator']
                    if accelerator not in ['auto', 'cpu', 'gpu', 'tpu']:
                        warnings.append(f"Неизвестный accelerator: {accelerator}")
                
                if 'precision' in compute_config:
                    precision = compute_config['precision']
                    if precision not in [16, 32, 64]:
                        warnings.append(f"Подозрительная precision: {precision}")
            
            # Вывод результатов
            if errors:
                print("❌ Найдены ошибки:")
                for error in errors:
                    print(f"  • {error}")
            
            if warnings:
                print("⚠️  Предупреждения:")
                for warning in warnings:
                    print(f"  • {warning}")
            
            if not errors and not warnings:
                print("✅ Конфигурация валидна!")
            elif not errors:
                print("✅ Конфигурация валидна (есть предупреждения)")
            
            return len(errors) == 0
            
        except Exception as e:
            print(f"❌ Ошибка при валидации: {e}")
            return False
    
    def create_config(self, config_name: str, template: str = "config") -> None:
        """Создание новой конфигурации на основе шаблона"""
        print(f"📝 Создание конфигурации: {config_name}")
        
        try:
            # Загружаем базовый шаблон
            base_config = self.load_config(template)
            
            # Модифицируем для новой конфигурации
            if 'project' in base_config:
                base_config['project']['name'] = config_name
                base_config['project']['created'] = datetime.now().isoformat()
                base_config['project']['based_on'] = template
            
            # Сохраняем
            self.save_config(base_config, config_name)
            
            print(f"✅ Конфигурация создана на основе {template}")
            print(f"💡 Отредактируйте файл: {self.configs_dir / f'{config_name}.yaml'}")
            
        except Exception as e:
            print(f"❌ Ошибка создания конфигурации: {e}")
    
    def merge_configs(self, base_config: str, overlay_config: str, 
                     output_name: str) -> None:
        """Объединение двух конфигураций"""
        print(f"🔀 Объединение {base_config} + {overlay_config} → {output_name}")
        
        try:
            base = self.load_config(base_config)
            overlay = self.load_config(overlay_config)
            
            # Рекурсивное объединение
            merged = self._deep_merge(base, overlay)
            
            # Добавляем метаданные
            if 'project' not in merged:
                merged['project'] = {}
            
            merged['project']['name'] = output_name
            merged['project']['merged_from'] = [base_config, overlay_config]
            merged['project']['created'] = datetime.now().isoformat()
            
            # Сохраняем
            self.save_config(merged, output_name)
            
            print("✅ Конфигурации объединены успешно")
            
        except Exception as e:
            print(f"❌ Ошибка объединения: {e}")
    
    def _deep_merge(self, base: Dict, overlay: Dict) -> Dict:
        """Глубокое объединение словарей"""
        result = base.copy()
        
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def compare_configs(self, config1: str, config2: str) -> None:
        """Сравнение двух конфигураций"""
        print(f"🔍 Сравнение {config1} ↔ {config2}")
        print("=" * 50)
        
        try:
            cfg1 = self.load_config(config1)
            cfg2 = self.load_config(config2)
            
            # Находим различия
            differences = self._find_differences(cfg1, cfg2, "")
            
            if not differences:
                print("✅ Конфигурации идентичны")
                return
            
            print(f"Найдено различий: {len(differences)}")
            print()
            
            for diff in differences:
                print(f"📍 {diff['path']}")
                print(f"  {config1}: {diff['value1']}")
                print(f"  {config2}: {diff['value2']}")
                print()
                
        except Exception as e:
            print(f"❌ Ошибка сравнения: {e}")
    
    def _find_differences(self, dict1: Any, dict2: Any, path: str) -> List[Dict]:
        """Поиск различий между структурами данных"""
        differences = []
        
        if type(dict1) != type(dict2):
            differences.append({
                'path': path,
                'value1': f"{type(dict1).__name__}: {dict1}",
                'value2': f"{type(dict2).__name__}: {dict2}"
            })
            return differences
        
        if isinstance(dict1, dict):
            all_keys = set(dict1.keys()) | set(dict2.keys())
            
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                
                if key not in dict1:
                    differences.append({
                        'path': new_path,
                        'value1': '<отсутствует>',
                        'value2': dict2[key]
                    })
                elif key not in dict2:
                    differences.append({
                        'path': new_path,
                        'value1': dict1[key],
                        'value2': '<отсутствует>'
                    })
                else:
                    differences.extend(
                        self._find_differences(dict1[key], dict2[key], new_path)
                    )
        
        elif isinstance(dict1, list):
            if len(dict1) != len(dict2):
                differences.append({
                    'path': f"{path}[length]",
                    'value1': len(dict1),
                    'value2': len(dict2)
                })
            
            for i in range(min(len(dict1), len(dict2))):
                differences.extend(
                    self._find_differences(dict1[i], dict2[i], f"{path}[{i}]")
                )
        
        elif dict1 != dict2:
            differences.append({
                'path': path,
                'value1': dict1,
                'value2': dict2
            })
        
        return differences
    
    def optimize_config(self, config_name: str, target: str = "balanced") -> None:
        """Оптимизация конфигурации для определенной цели"""
        print(f"⚡ Оптимизация {config_name} для {target}")
        
        try:
            config = self.load_config(config_name)
            
            if target == "speed":
                # Оптимизация для скорости
                if 'training' in config:
                    config['training']['batch_size'] = min(config['training'].get('batch_size', 32) * 2, 128)
                    config['training']['max_epochs'] = min(config['training'].get('max_epochs', 100), 50)
                    config['training']['num_workers'] = 8
                
                if 'models' in config and 'sota' in config['models']:
                    # Отключаем тяжелые модели
                    if 'earthformer' in config['models']['sota']:
                        config['models']['sota']['earthformer']['enabled'] = False
                    if 'tft' in config['models']['sota']:
                        config['models']['sota']['tft']['enabled'] = False
                
                if 'compute' in config:
                    config['compute']['precision'] = 16
                    config['compute']['compile_models'] = True
            
            elif target == "quality":
                # Оптимизация для качества
                if 'training' in config:
                    config['training']['max_epochs'] = max(config['training'].get('max_epochs', 100), 200)
                    config['training']['early_stopping_patience'] = 30
                
                if 'evaluation' in config:
                    config['evaluation']['cross_validation']['enabled'] = True
                    config['evaluation']['cross_validation']['n_splits'] = 10
            
            elif target == "memory":
                # Оптимизация для памяти
                if 'training' in config:
                    config['training']['batch_size'] = max(config['training'].get('batch_size', 32) // 2, 4)
                    config['training']['accumulate_grad_batches'] = 4
                
                if 'compute' in config:
                    config['compute']['precision'] = 16
                    config['compute']['max_memory_usage'] = 0.7
            
            # Сохраняем оптимизированную версию
            optimized_name = f"{config_name}_{target}_optimized"
            self.save_config(config, optimized_name)
            
            print(f"✅ Оптимизированная конфигурация: {optimized_name}.yaml")
            
        except Exception as e:
            print(f"❌ Ошибка оптимизации: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Менеджер конфигураций проекта предсказания засухи",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python config_manager.py list
  python config_manager.py validate quick_test
  python config_manager.py create my_experiment --template research
  python config_manager.py merge config quick_test my_quick_config
  python config_manager.py compare earthformer research
  python config_manager.py optimize config --target speed
        """
    )
    
    parser.add_argument('command', choices=[
        'list', 'validate', 'create', 'merge', 'compare', 'optimize'
    ], help='Команда для выполнения')
    
    parser.add_argument('args', nargs='*', help='Аргументы команды')
    parser.add_argument('--template', default='config', help='Шаблон для создания')
    parser.add_argument('--target', choices=['speed', 'quality', 'memory'], 
                       default='balanced', help='Цель оптимизации')
    parser.add_argument('--configs-dir', default='configs', help='Директория конфигураций')
    
    args = parser.parse_args()
    
    manager = ConfigManager(args.configs_dir)
    
    try:
        if args.command == 'list':
            manager.list_configs()
        
        elif args.command == 'validate':
            if not args.args:
                print("❌ Укажите имя конфигурации для валидации")
                return 1
            manager.validate_config(args.args[0])
        
        elif args.command == 'create':
            if not args.args:
                print("❌ Укажите имя новой конфигурации")
                return 1
            manager.create_config(args.args[0], args.template)
        
        elif args.command == 'merge':
            if len(args.args) < 3:
                print("❌ Укажите: base_config overlay_config output_name")
                return 1
            manager.merge_configs(args.args[0], args.args[1], args.args[2])
        
        elif args.command == 'compare':
            if len(args.args) < 2:
                print("❌ Укажите два имени конфигураций для сравнения")
                return 1
            manager.compare_configs(args.args[0], args.args[1])
        
        elif args.command == 'optimize':
            if not args.args:
                print("❌ Укажите имя конфигурации для оптимизации")
                return 1
            manager.optimize_config(args.args[0], args.target)
        
        return 0
        
    except Exception as e:
        print(f"❌ Ошибка выполнения команды: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())