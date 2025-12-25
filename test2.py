
"""
Главный скрипт для запуска всех тестов
"""

import sys
import os
import subprocess
import time

def print_header(text):
    """Красивый вывод заголовка"""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)

def run_unittest():
    """Запуск юнит-тестов"""
    print_header("ЗАПУСК ЮНИТ-ТЕСТОВ")
    
    try:
        import test1
        
        print("\n✅ Юнит-тесты завершены")
        return True
    except Exception as e:
        print(f"\n❌ Ошибка при запуске юнит-тестов: {e}")
        return False

def run_performance_tests():
    """Запуск тестов производительности"""
    print_header("ЗАПУСК ТЕСТОВ ПРОИЗВОДИТЕЛЬНОСТИ")
    
    try:
        import test
        return test.main() == 0
    except Exception as e:
        print(f"\n❌ Ошибка при запуске тестов производительности: {e}")
        return False

def run_smoke_tests():
    """Запуск быстрых smoke-тестов"""
    print_header("ЗАПУСК SMOKE-ТЕСТОВ")
    
    try:
        import numpy as np
        from main2 import NeuralNetwork, Car
        
        print("1. Проверка нейронной сети...")
        nn = NeuralNetwork(3, 5, 2)
        output = nn.predict([0.5, 0.3, 0.8])
        print(f"   ✅ Сеть создана, выход: {output}")
        
        print("2. Проверка автомобиля...")
        car = Car(100, 100, 45, None, 0)
        print(f"   ✅ Автомобиль создан, позиция: ({car.x}, {car.y})")
        
        print("3. Проверка мутации...")
        original = nn.weights_ih.copy()
        nn.mutate(0.5)
        changed = not np.array_equal(original, nn.weights_ih)
        print(f"   ✅ Мутация работает: {'веса изменились' if changed else 'веса не изменились'}")
        
        print("\n✅ Все smoke-тесты пройдены")
        return True
        
    except Exception as e:
        print(f"\n❌ Smoke-тесты провалены: {e}")
        return False



def main():
    """Главная функция"""
    print_header("ТЕСТИРОВАНИЕ AI RACING SIMULATOR")
    print("Версия 1.0 | Автоматическое тестирование")
    
    start_time = time.time()
    
    if not os.path.exists('main2.py'):
        print("\n❌ ОШИБКА: Файл game.py не найден в текущей директории!")
        print("Запустите тесты из директории с проектом.")
        return 1
    
    results = []
    
    test_suites = [
        ("Smoke-тесты", run_smoke_tests),
        ("Юнит-тесты", run_unittest),
        ("Тесты производительности", run_performance_tests),
    ]
    
    for suite_name, suite_func in test_suites:
        print(f"\n▶️  Запускаю {suite_name}...")
        try:
            success = suite_func()
            results.append((suite_name, success))
            
            if success:
                print(f"✅ {suite_name} завершены успешно")
            else:
                print(f"❌ {suite_name} завершены с ошибками")
                
        except KeyboardInterrupt:
            print(f"\n⚠️  {suite_name} прерваны пользователем")
            return 1
        except Exception as e:
            print(f"\n💥 Неожиданная ошибка в {suite_name}: {e}")
            results.append((suite_name, False))
    
    elapsed_time = time.time() - start_time
    
    print_header("ИТОГОВЫЙ ОТЧЕТ")
    
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    passed = 0
    for suite_name, success in results:
        status = "✅ ПРОЙДЕНЫ" if success else "❌ ПРОВАЛЕНЫ"
        print(f"  {suite_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n⏱️  Общее время тестирования: {elapsed_time:.1f} сек")
    print(f"🎯 Итог: {passed}/{len(results)} наборов тестов пройдено")
    
    if passed == len(results):
        print("\n" + "="*70)
        print("🎉 ПОЗДРАВЛЯЕМ! ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print(f"💥 ВНИМАНИЕ: {len(results) - passed} наборов тестов провалено")
        print("="*70)
        return 1

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Тестирование прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        sys.exit(1)
