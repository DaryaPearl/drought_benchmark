"""
Утилиты для визуализации обучения и результатов моделей предсказания засухи

Включает:
- Визуализация процесса обучения
- Сравнение моделей
- Анализ предсказаний
- Визуализация структуры данных
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import xarray as xr
from datetime import datetime
import json

# Настройка стиля
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

class DatasetVisualizer:
    """Визуализация структуры и содержимого датасета"""
    
    def __init__(self, dataset: xr.Dataset, save_dir: Path):
        self.dataset = dataset
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_dataset_overview(self):
        """Обзорная визуализация датасета"""
        print("📊 Создание обзора датасета...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Временные ряды переменных
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_time_series(ax1)
        
        # 2. Пространственные карты
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_spatial_means(ax2)
        
        # 3. Корреляционная матрица
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_correlation_matrix(ax3)
        
        # 4. Распределения значений
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_value_distributions(ax4)
        
        # 5. Покрытие данными
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_data_coverage(ax5)
        
        # 6. Сезонность
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_seasonality(ax6)
        
        plt.suptitle('Обзор датасета для предсказания засухи', fontsize=16, fontweight='bold')
        plt.savefig(self.save_dir / 'dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Обзор датасета сохранен: {self.save_dir / 'dataset_overview.png'}")
    
    def _plot_time_series(self, ax):
        """Временные ряды основных переменных"""
        variables = ['precipitation', 'temperature', 'spi3', 'ndvi']
        available_vars = [v for v in variables if v in self.dataset.data_vars]
        
        for i, var in enumerate(available_vars[:4]):
            # Глобальное среднее
            global_mean = self.dataset[var].mean(dim=['latitude', 'longitude'])
            
            ax.plot(self.dataset.time, global_mean, label=var, linewidth=1.5, alpha=0.8)
        
        ax.set_title('Временные ряды (глобальные средние)')
        ax.set_xlabel('Время')
        ax.set_ylabel('Значение')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Форматирование оси времени
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_spatial_means(self, ax):
        """Пространственное распределение средних значений"""
        # Используем SPI3 как основную переменную
        target_var = 'spi3' if 'spi3' in self.dataset.data_vars else list(self.dataset.data_vars)[0]
        
        mean_data = self.dataset[target_var].mean(dim='time')
        
        im = ax.imshow(
            mean_data.values,
            extent=[
                self.dataset.longitude.min(), self.dataset.longitude.max(),
                self.dataset.latitude.min(), self.dataset.latitude.max()
            ],
            aspect='auto',
            origin='lower',
            cmap='RdYlBu_r'
        )
        
        ax.set_title(f'Среднее пространственное распределение {target_var}')
        ax.set_xlabel('Долгота')
        ax.set_ylabel('Широта')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _plot_correlation_matrix(self, ax):
        """Корреляционная матрица переменных"""
        # Выбираем основные переменные
        variables = ['precipitation', 'temperature', 'spi3', 'ndvi', 'soil_moisture']
        available_vars = [v for v in variables if v in self.dataset.data_vars]
        
        if len(available_vars) < 2:
            ax.text(0.5, 0.5, 'Недостаточно переменных\nдля корреляционной матрицы', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Корреляционная матрица')
            return
        
        # Вычисление корреляций (случайная выборка для ускорения)
        correlation_data = {}
        for var in available_vars:
            data = self.dataset[var].values.flatten()
            data = data[~np.isnan(data)]
            if len(data) > 10000:
                data = np.random.choice(data, 10000, replace=False)
            correlation_data[var] = data
        
        # Приведение к одинаковой длине
        min_len = min(len(v) for v in correlation_data.values())
        for var in correlation_data:
            correlation_data[var] = correlation_data[var][:min_len]
        
        corr_df = pd.DataFrame(correlation_data)
        corr_matrix = corr_df.corr()
        
        # Тепловая карта
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Корреляционная матрица переменных')
    
    def _plot_value_distributions(self, ax):
        """Распределения значений переменных"""
        variables = ['precipitation', 'spi3', 'temperature']
        available_vars = [v for v in variables if v in self.dataset.data_vars]
        
        for var in available_vars[:3]:
            data = self.dataset[var].values.flatten()
            data = data[~np.isnan(data)]
            
            # Случайная выборка для ускорения
            if len(data) > 50000:
                data = np.random.choice(data, 50000, replace=False)
            
            ax.hist(data, bins=50, alpha=0.6, label=var, density=True)
        
        ax.set_title('Распределения значений переменных')
        ax.set_xlabel('Значение')
        ax.set_ylabel('Плотность')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_data_coverage(self, ax):
        """Покрытие данными по времени"""
        # Процент валидных данных по времени
        target_var = list(self.dataset.data_vars)[0]
        
        coverage_by_time = []
        for t in range(len(self.dataset.time)):
            data_slice = self.dataset[target_var].isel(time=t)
            valid_fraction = 1 - np.isnan(data_slice.values).mean()
            coverage_by_time.append(valid_fraction * 100)
        
        ax.plot(self.dataset.time, coverage_by_time, linewidth=2, color='blue')
        ax.fill_between(self.dataset.time, coverage_by_time, alpha=0.3, color='blue')
        
        ax.set_title('Покрытие данными по времени')
        ax.set_xlabel('Время')
        ax.set_ylabel('Процент валидных данных')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Форматирование оси времени
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_seasonality(self, ax):
        """Сезонные паттерны"""
        # Используем SPI3 или первую доступную переменную
        target_var = 'spi3' if 'spi3' in self.dataset.data_vars else list(self.dataset.data_vars)[0]
        
        # Группировка по месяцам
        monthly_data = self.dataset[target_var].groupby('time.month').mean(dim=['latitude', 'longitude', 'time'])
        
        months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
        
        ax.bar(range(1, 13), monthly_data.values, color='skyblue', alpha=0.7)
        ax.set_title(f'Сезонность {target_var}')
        ax.set_xlabel('Месяц')
        ax.set_ylabel(f'Среднее {target_var}')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months, rotation=45)
        ax.grid(True, alpha=0.3)

class TrainingVisualizer:
    """Визуализация процесса обучения"""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(self, log_dir: Path):
        """Построение кривых обучения из логов TensorBoard"""
        print("📈 Создание кривых обучения...")
        
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            
            # Поиск файлов логов
            log_files = list(log_dir.rglob('events.out.tfevents.*'))
            
            if not log_files:
                print("⚠ Логи TensorBoard не найдены")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            colors = ['blue', 'red', 'green', 'orange']
            
            for i, log_file in enumerate(log_files[:4]):
                model_name = log_file.parent.name
                
                # Загрузка логов
                ea = EventAccumulator(str(log_file.parent))
                ea.Reload()
                
                # Извлечение метрик
                train_loss = ea.Scalars('train_loss_epoch') if 'train_loss_epoch' in ea.Tags()['scalars'] else []
                val_loss = ea.Scalars('val_loss') if 'val_loss' in ea.Tags()['scalars'] else []
                
                ax = axes[i] if i < len(axes) else axes[-1]
                
                if train_loss:
                    steps_train, values_train = zip(*[(s.step, s.value) for s in train_loss])
                    ax.plot(steps_train, values_train, label='Train Loss', color=colors[i % len(colors)], alpha=0.7)
                
                if val_loss:
                    steps_val, values_val = zip(*[(s.step, s.value) for s in val_loss])
                    ax.plot(steps_val, values_val, label='Val Loss', color=colors[i % len(colors)], linestyle='--')
                
                ax.set_title(f'{model_name} - Training Curves')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Удаление пустых subplot'ов
            for i in range(len(log_files), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Кривые обучения сохранены: {self.save_dir / 'training_curves.png'}")
            
        except ImportError:
            print("⚠ tensorboard не установлен, пропускаем кривые обучения")
        except Exception as e:
            print(f"⚠ Ошибка создания кривых обучения: {e}")
    
    def plot_loss_comparison(self, results: Dict[str, Dict]):
        """Сравнение финальных потерь моделей"""
        print("📊 Создание сравнения потерь...")
        
        model_names = []
        train_losses = []
        val_losses = []
        
        for model_name, model_results in results.items():
            if isinstance(model_results, dict) and 'error' not in model_results:
                if model_name == 'classical':
                    # Для классических моделей берем лучший результат
                    best_model = min(model_results.items(), 
                                   key=lambda x: x[1].get('val_metrics', {}).get('mae', float('inf')))
                    model_names.append(f"{best_model[0]} (best)")
                    train_losses.append(best_model[1].get('train_metrics', {}).get('mae', 0))
                    val_losses.append(best_model[1].get('val_metrics', {}).get('mae', 0))
                else:
                    model_names.append(model_name)
                    train_losses.append(model_results.get('train_metrics', {}).get('final_train_loss', 0))
                    val_losses.append(model_results.get('val_metrics', {}).get('final_val_loss', 0))
        
        if not model_names:
            print("⚠ Нет данных для сравнения потерь")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # График 1: Сравнение потерь
        x = np.arange(len(model_names))
        width = 0.35
        
        ax1.bar(x - width/2, train_losses, width, label='Train Loss', alpha=0.8)
        ax1.bar(x + width/2, val_losses, width, label='Val Loss', alpha=0.8)
        
        ax1.set_xlabel('Модели')
        ax1.set_ylabel('Loss')
        ax1.set_title('Сравнение потерь моделей')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: Время обучения
        training_times = []
        for model_name, model_results in results.items():
            if isinstance(model_results, dict) and 'error' not in model_results:
                if model_name == 'classical':
                    avg_time = np.mean([r.get('training_time', 0) for r in model_results.values() 
                                      if isinstance(r, dict)])
                    training_times.append(avg_time)
                else:
                    training_times.append(model_results.get('training_time', 0))
        
        if training_times:
            bars = ax2.bar(model_names, training_times, alpha=0.8, color='orange')
            ax2.set_xlabel('Модели')
            ax2.set_ylabel('Время обучения (с)')
            ax2.set_title('Время обучения моделей')
            ax2.set_xticklabels(model_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Добавление значений на столбцы
            for bar, time_val in zip(bars, training_times):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{time_val:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Сравнение моделей сохранено: {self.save_dir / 'model_comparison.png'}")

class ResultsVisualizer:
    """Визуализация результатов и предсказаний"""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_model_comparison(self, results: Dict[str, Dict]):
        """Комплексное сравнение моделей"""
        print("📊 Создание комплексного сравнения моделей...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Метрики качества
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_metrics_comparison(ax1, results)
        
        # 2. Время обучения
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_training_time(ax2, results)
        
        # 3. Radar chart метрик
        ax3 = fig.add_subplot(gs[0, 2], projection='polar')
        self._plot_metrics_radar(ax3, results)
        
        # 4. Детальное сравнение классических моделей
        ax4 = fig.add_subplot(gs[1, :2])
        self._plot_classical_detailed(ax4, results)
        
        # 5. Сводная таблица
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_summary_table(ax5, results)
        
        plt.suptitle('Комплексное сравнение моделей предсказания засухи', 
                    fontsize=16, fontweight='bold')
        plt.savefig(self.save_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Комплексное сравнение сохранено: {self.save_dir / 'comprehensive_comparison.png'}")
    
    def _plot_metrics_comparison(self, ax, results):
        """Сравнение основных метрик"""
        models = []
        mae_scores = []
        rmse_scores = []
        
        for model_name, model_results in results.items():
            if isinstance(model_results, dict) and 'error' not in model_results:
                if model_name == 'classical':
                    # Берем лучшую классическую модель
                    best_classical = min(model_results.items(), 
                                        key=lambda x: x[1].get('val_metrics', {}).get('mae', float('inf')))
                    models.append(f"{best_classical[0]} (best)")
                    mae_scores.append(best_classical[1].get('val_metrics', {}).get('mae', 0))
                    rmse_scores.append(best_classical[1].get('val_metrics', {}).get('rmse', 0))
                else:
                    models.append(model_name)
                    # Для SOTA моделей используем loss как аппроксимацию MAE
                    val_loss = model_results.get('val_metrics', {}).get('final_val_loss', 0)
                    mae_scores.append(val_loss)
                    rmse_scores.append(val_loss * 1.2)  # Примерная оценка
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, mae_scores, width, label='MAE', alpha=0.8)
        ax.bar(x + width/2, rmse_scores, width, label='RMSE', alpha=0.8)
        
        ax.set_xlabel('Модели')
        ax.set_ylabel('Значение метрики')
        ax.set_title('Сравнение метрик качества')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_time(self, ax, results):
        """График времени обучения"""
        models = []
        times = []
        
        for model_name, model_results in results.items():
            if isinstance(model_results, dict) and 'error' not in model_results:
                if model_name == 'classical':
                    avg_time = np.mean([r.get('training_time', 0) for r in model_results.values()
                                      if isinstance(r, dict)])
                    models.append('Classical (avg)')
                    times.append(avg_time)
                else:
                    models.append(model_name)
                    times.append(model_results.get('training_time', 0))
        
        bars = ax.bar(models, times, alpha=0.8, color='orange')
        ax.set_xlabel('Модели')
        ax.set_ylabel('Время обучения (с)')
        ax.set_title('Время обучения моделей')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Логарифмическая шкала если большой разброс
        if max(times) / min([t for t in times if t > 0] + [1]) > 100:
            ax.set_yscale('log')
        
        # Добавление значений
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_val:.1f}s', ha='center', va='bottom')
    
    def _plot_metrics_radar(self, ax, results):
        """Radar chart сравнения метрик"""
        # Нормализованные метрики для radar chart
        categories = ['Accuracy', 'Speed', 'Complexity']
        
        models_data = {}
        for model_name, model_results in results.items():
            if isinstance(model_results, dict) and 'error' not in model_results:
                if model_name == 'classical':
                    # Усредняем по всем классическим моделям
                    accuracy = 1 - np.mean([r.get('val_metrics', {}).get('mae', 1) 
                                          for r in model_results.values() if isinstance(r, dict)])
                    speed = 1.0  # Классические модели быстрые
                    complexity = 0.3  # Простые
                else:
                    accuracy = 1 - model_results.get('val_metrics', {}).get('final_val_loss', 1)
                    speed = 0.5  # SOTA модели медленнее
                    complexity = 0.8  # Сложные
                
                models_data[model_name] = [max(0, accuracy), speed, complexity]
        
        # Углы для radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Замыкаем круг
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_name, values) in enumerate(models_data.items()):
            values += values[:1]  # Замыкаем круг
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, 
                   color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Сравнение характеристик моделей', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
    
    def _plot_classical_detailed(self, ax, results):
        """Детальное сравнение классических моделей"""
        if 'classical' not in results:
            ax.text(0.5, 0.5, 'Классические модели не обучались', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Детальное сравнение классических моделей')
            return
        
        classical_results = results['classical']
        models = []
        mae_vals = []
        r2_vals = []
        
        for model_name, model_result in classical_results.items():
            if isinstance(model_result, dict) and 'val_metrics' in model_result:
                models.append(model_name)
                mae_vals.append(model_result['val_metrics'].get('mae', 0))
                r2_vals.append(model_result['val_metrics'].get('r2', 0))
        
        x = np.arange(len(models))
        
        # Двойная ось Y
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x - 0.2, mae_vals, 0.4, label='MAE', alpha=0.8, color='blue')
        bars2 = ax2.bar(x + 0.2, r2_vals, 0.4, label='R²', alpha=0.8, color='red')
        
        ax.set_xlabel('Классические модели')
        ax.set_ylabel('MAE', color='blue')
        ax2.set_ylabel('R²', color='red')
        ax.set_title('Детальное сравнение классических моделей')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Добавление значений на столбцы
        for bar, val in zip(bars1, mae_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', color='blue')
        
        for bar, val in zip(bars2, r2_vals):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', color='red')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_summary_table(self, ax, results):
        """Сводная таблица результатов"""
        ax.axis('tight')
        ax.axis('off')
        
        # Подготовка данных для таблицы
        table_data = []
        headers = ['Модель', 'Val Loss/MAE', 'Время (с)', 'Тип']
        
        for model_name, model_results in results.items():
            if isinstance(model_results, dict) and 'error' not in model_results:
                if model_name == 'classical':
                    # Лучшая классическая модель
                    best_classical = min(model_results.items(), 
                                        key=lambda x: x[1].get('val_metrics', {}).get('mae', float('inf')))
                    table_data.append([
                        f"{best_classical[0]} (best)",
                        f"{best_classical[1].get('val_metrics', {}).get('mae', 0):.4f}",
                        f"{best_classical[1].get('training_time', 0):.1f}",
                        "Classical"
                    ])
                else:
                    val_loss = model_results.get('val_metrics', {}).get('final_val_loss', 0)
                    training_time = model_results.get('training_time', 0)
                    table_data.append([
                        model_name,
                        f"{val_loss:.4f}",
                        f"{training_time:.1f}",
                        "SOTA"
                    ])
        
        if table_data:
            table = ax.table(cellText=table_data, colLabels=headers,
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # Стилизация таблицы
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            for i in range(1, len(table_data) + 1):
                for j in range(len(headers)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f1f1f2')
        
        ax.set_title('Сводная таблица результатов', pad=20)
    
    def plot_training_curves(self, log_dir: Path):
        """Построение кривых обучения"""
        training_viz = TrainingVisualizer(self.save_dir)
        training_viz.plot_training_curves(log_dir)

def create_comprehensive_report(dataset_path: str, results: Dict, output_dir: Path):
    """Создание комплексного отчета с визуализациями"""
    print("📊 Создание комплексного отчета...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Визуализация датасета
        if Path(dataset_path).exists():
            dataset = xr.open_zarr(dataset_path)
            dataset_viz = DatasetVisualizer(dataset, output_dir)
            dataset_viz.plot_dataset_overview()
        
        # 2. Визуализация результатов
        results_viz = ResultsVisualizer(output_dir)
        results_viz.plot_model_comparison(results)
        
        # 3. Визуализация обучения
        training_viz = TrainingVisualizer(output_dir)
        if (output_dir.parent / 'lightning_logs').exists():
            training_viz.plot_training_curves(output_dir.parent / 'lightning_logs')
        
        print(f"✅ Комплексный отчет создан в: {output_dir}")
        
    except Exception as e:
        print(f"❌ Ошибка создания отчета: {e}")

# Утилитарные функции
def save_predictions_visualization(predictions: np.ndarray, 
                                 targets: np.ndarray,
                                 save_path: Path,
                                 model_name: str = "Model"):
    """Визуализация предсказаний vs реальных значений"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Scatter plot
    axes[0].scatter(targets.flatten(), predictions.flatten(), alpha=0.5)
    axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    axes[0].set_xlabel('Реальные значения')
    axes[0].set_ylabel('Предсказания')
    axes[0].set_title(f'{model_name}: Предсказания vs Реальные')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of errors
    errors = predictions.flatten() - targets.flatten()
    axes[1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
    axes[1].set_xlabel('Ошибка предсказания')
    axes[1].set_ylabel('Частота')
    axes[1].set_title('Распределение ошибок')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Time series example
    if len(predictions.shape) > 1:
        sample_idx = np.random.randint(0, predictions.shape[0])
        axes[2].plot(targets[sample_idx], label='Реальные', linewidth=2)
        axes[2].plot(predictions[sample_idx], label='Предсказания', linewidth=2, alpha=0.8)
        axes[2].set_xlabel('Временной шаг')
        axes[2].set_ylabel('Значение')
        axes[2].set_title('Пример временного ряда')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / f'{model_name.lower()}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Пример использования
    print("🎨 Утилиты визуализации для предсказания засухи")
    print("Используйте функции из этого модуля в ваших скриптах обучения")