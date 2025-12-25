
"""
Тесты производительности
"""

import time
import numpy as np
import sys

sys.path.append('.')
from main2 import NeuralNetwork, GeneticAlgorithm, Car

def test_neural_network_performance():
    """Тест скорости нейронной сети"""
    print("\n" + "="*60)
    print("ТЕСТ СКОРОСТИ НЕЙРОННОЙ СЕТИ")
    print("="*60)
    
    nn = NeuralNetwork(7, 10, 4)
    
    test_inputs = np.random.rand(1000, 7).tolist()
    
    start_time = time.time()
    
    for i, inputs in enumerate(test_inputs):
        nn.predict(inputs)
        if i % 200 == 0: 
            print(f"  Обработано {i+1}/{len(test_inputs)} предсказаний...")
    
    end_time = time.time()
    elapsed = end_time - start_time

    speed = len(test_inputs) / elapsed
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"  Всего предсказаний: {len(test_inputs)}")
    print(f"  Затраченное время: {elapsed:.3f} сек")
    print(f"  Скорость: {speed:.1f} предсказаний/сек")
    
    if speed > 1000:
        print("✅ Производительность отличная (>1000 предсказаний/сек)")
        return True
    elif speed > 500:
        print("⚠️  Производительность приемлемая (>500 предсказаний/сек)")
        return True
    else:
        print(f"❌ Производительность низкая: {speed:.1f} предсказаний/сек")
        return False

def test_genetic_algorithm_scalability():
    """Тест масштабируемости генетического алгоритма"""
    print("\n" + "="*60)
    print("ТЕСТ МАСШТАБИРУЕМОСТИ ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
    print("="*60)
    
    results = []
    
    for size in [10, 20, 50]:
        print(f"\nТестирование популяции размером {size}...")
        
        start_time = time.time()

        ga = GeneticAlgorithm(size, 7, 10, 4)
        
        for net in ga.population:
            net.fitness = np.random.rand() * 100
        
        ga.evolve()
        
        elapsed = time.time() - start_time
        results.append((size, elapsed))
        
        print(f"  Время эволюции: {elapsed:.3f} сек")
    
    print("\n📊 АНАЛИЗ МАСШТАБИРУЕМОСТИ:")
    for i, (size, time_taken) in enumerate(results):
        print(f"  Популяция {size}: {time_taken:.3f} сек")
        
        if i > 0:
            prev_size, prev_time = results[i-1]
            time_ratio = time_taken / prev_time
            size_ratio = size / prev_size
            efficiency = size_ratio / time_ratio
            
            print(f"    Рост размера: x{size_ratio:.1f}, "
                  f"Рост времени: x{time_ratio:.1f}, "
                  f"Эффективность: {efficiency:.2f}")
    
    return True


def main():
    """Главная функция тестов производительности"""
    print("="*70)
    print("ТЕСТЫ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*70)
    
    tests = [
        ("Скорость нейронной сети", test_neural_network_performance),
        ("Масштабируемость ГА", test_genetic_algorithm_scalability)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n▶️  Запуск теста: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ Тест '{test_name}' пройден")
            else:
                print(f"❌ Тест '{test_name}' провален")
        except Exception as e:
            print(f"💥 Ошибка в тесте '{test_name}': {e}")
            results.append((test_name, False))
    
    # Сводка
    print("\n" + "="*70)
    print("ИТОГИ ТЕСТОВ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n📊 СВОДКА:")
    for test_name, success in results:
        status = "✅ ПРОЙДЕН" if success else "❌ ПРОВАЛЕН"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 РЕЗУЛЬТАТ: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОИЗВОДИТЕЛЬНОСТИ ПРОЙДЕНЫ!")
        return 0
    else:
        print(f"\n💥 {total - passed} ТЕСТОВ ПРОВАЛЕНО")
        return 1

if __name__ == '__main__':
    sys.exit(main())
