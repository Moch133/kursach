import unittest
import numpy as np
import json
import tempfile
import os
import sys

sys.path.append('.')
try:
    from main2 import NeuralNetwork, GeneticAlgorithm, Car, Button
except ImportError:
    print("❌ Ошибка: Не удалось импортировать модуль 'game'")
    print("Убедитесь, что файл game.py находится в текущей директории")
    sys.exit(1)

class TestNeuralNetwork(unittest.TestCase):
    """Ключевые тесты нейронной сети"""
    
    def setUp(self):
        print(f"\n{'='*60}")
        print(f"Тест: {self._testMethodName}")
        print(f"{'='*60}")
        self.nn = NeuralNetwork(5, 10, 3)
    
    def test_01_predict_output_range(self):
        """Выходные значения должны быть в диапазоне 0-1 (sigmoid)"""
        inputs = [0.5, 0.3, 0.8, 0.1, 0.9]
        output = self.nn.predict(inputs)
        print(f"Вход: {inputs}")
        print(f"Выход: {output}")
        self.assertEqual(len(output), 3, "Должно быть 3 выходных значения")
        for i, val in enumerate(output):
            self.assertTrue(0 <= val <= 1, f"Выход {i} вне диапазона: {val}")
        print("✅ Выходные значения в диапазоне 0-1")
    
    def test_02_mutate_changes_weights(self):
        """Мутация должна изменять веса"""
        original_weights = self.nn.weights_ih.copy()
        print(f"Веса до мутации (первые 5): {original_weights[0][:5]}")
        
        self.nn.mutate(mutation_rate=0.5)
        
        print(f"Веса после мутации (первые 5): {self.nn.weights_ih[0][:5]}")
        self.assertFalse(np.array_equal(original_weights, self.nn.weights_ih), 
                        "Веса должны измениться после мутации")
        print("✅ Мутация изменяет веса")
    
    def test_03_save_load_consistency(self):
        """Сохранение и загрузка должны сохранять состояние"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name
        
        try:
            self.nn.fitness = 100
            self.nn.best_lap_time = 30.5
            self.nn.save(filename)
            print(f"Сеть сохранена в: {filename}")
            
            self.assertTrue(os.path.exists(filename), "Файл не создан")
            file_size = os.path.getsize(filename)
            print(f"Размер файла: {file_size} байт")
            
            new_nn = NeuralNetwork(5, 10, 3)
            new_nn.load(filename)
            
            np.testing.assert_array_almost_equal(
                self.nn.weights_ih, new_nn.weights_ih,
                decimal=6, err_msg="Веса не совпадают после загрузки"
            )
            self.assertEqual(self.nn.fitness, new_nn.fitness, 
                           "Фитнес не совпадает после загрузки")
            self.assertEqual(self.nn.best_lap_time, new_nn.best_lap_time,
                           "Лучшее время не совпадает после загрузки")
            print("✅ Сохранение и загрузка работают корректно")
            
        finally:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Временный файл удален: {filename}")

class TestGeneticAlgorithm(unittest.TestCase):
    """Ключевые тесты генетического алгоритма"""
    
    def setUp(self):
        print(f"\n{'='*60}")
        print(f"Тест: {self._testMethodName}")
        print(f"{'='*60}")
        self.ga = GeneticAlgorithm(10, 5, 8, 2)
    
    def test_01_initialization(self):
        """Инициализация должна создавать корректную популяцию"""
        print(f"Размер популяции: {len(self.ga.population)}")
        print(f"Поколение: {self.ga.generation}")
        
        self.assertEqual(len(self.ga.population), 10, 
                        "Неправильный размер популяции")
        
        for i, network in enumerate(self.ga.population):
            self.assertEqual(network.input_nodes, 5, 
                           f"Сеть {i}: неправильное число входов")
            self.assertEqual(network.output_nodes, 2,
                           f"Сеть {i}: неправильное число выходов")
            self.assertEqual(network.fitness, 0,
                           f"Сеть {i}: фитнес должен быть 0 при инициализации")
        
        print(f"✅ Популяция из {len(self.ga.population)} сетей создана")
    
    def test_02_elitism_preserves_best(self):
        """Элитизм должен сохранять лучшие особи"""
        for i, net in enumerate(self.ga.population):
            net.fitness = i * 10
        
        best_network = self.ga.population[-1].copy()
        best_fitness = best_network.fitness
        print(f"Лучший фитнес до эволюции: {best_fitness}")
        
        # Эволюция
        self.ga.evolve()
        
        print(f"Поколение после эволюции: {self.ga.generation}")
        
        found = False
        for i, net in enumerate(self.ga.population[:2]):
            if np.array_equal(best_network.weights_ih, net.weights_ih):
                found = True
                print(f"Лучшая сеть найдена в позиции {i}")
                break
        
        self.assertTrue(found, "Лучшая сеть не сохранилась через элитизм")
        print("✅ Элитизм сохраняет лучшие особи")
    
    def test_03_generation_counter(self):
        """Счетчик поколений должен увеличиваться"""
        initial_gen = self.ga.generation
        print(f"Начальное поколение: {initial_gen}")
        
        for i in range(3):
            for net in self.ga.population:
                net.fitness = np.random.rand() * 100
            
            self.ga.evolve()
            print(f"После эволюции {i+1}: поколение {self.ga.generation}")
            self.assertEqual(self.ga.generation, initial_gen + i + 1,
                           f"Неправильный счетчик после эволюции {i+1}")
        
        print("✅ Счетчик поколений увеличивается правильно")

class TestCarPhysics(unittest.TestCase):
    """Тесты физики автомобиля"""
    
    def setUp(self):
        print(f"\n{'='*60}")
        print(f"Тест: {self._testMethodName}")
        print(f"{'='*60}")
        self.car = Car(100, 100, 0, None, car_id=0)
    
    def test_01_fitness_calculation(self):
        """Расчет фитнеса должен работать корректно"""
        test_cases = [
            {"laps": 0, "best_time": float('inf'), "expected_min": 0, "desc": "Нулевые значения"},
            {"laps": 1, "best_time": 60, "expected_min": 1000, "desc": "Один круг"},
            {"laps": 10, "best_time": 30, "expected_min": 10000, "desc": "10 кругов (победа)"},
        ]
        
        for case in test_cases:
            self.car.lap_count = case["laps"]
            self.car.best_lap_time = case["best_time"]
            
            fitness = self.car.calculate_fitness()
            
            print(f"{case['desc']}:")
            print(f"  Круги: {case['laps']}, Лучшее время: {case['best_time']}")
            print(f"  Фитнес: {fitness:.1f}")
            
            if case["laps"] == 0:
                self.assertEqual(fitness, 0, "Фитнес должен быть 0 без кругов")
            else:
                self.assertGreaterEqual(fitness, case["expected_min"], 
                                      f"Фитнес слишком низкий для {case['desc']}")
        
        print("✅ Расчет фитнеса работает корректно")
    
    def test_02_sensor_initialization(self):
        """Сенсоры должны быть инициализированы"""
        print(f"Углы сенсоров: {self.car.sensor_angles}")
        print(f"Расстояния сенсоров: {self.car.sensor_distances}")
        print(f"Макс. расстояние сенсора: {self.car.max_sensor_distance}")
        
        self.assertEqual(len(self.car.sensor_angles), 5, 
                        "Должно быть 5 углов сенсоров")
        self.assertEqual(len(self.car.sensor_distances), 5,
                        "Должно быть 5 расстояний сенсоров")
        self.assertEqual(self.car.max_sensor_distance, 300,
                        "Неправильное максимальное расстояние")
        
        expected_angles = [-90, -45, 0, 45, 90]
        for expected, actual in zip(expected_angles, self.car.sensor_angles):
            self.assertEqual(expected, actual, f"Угол сенсора неверный")
        
        for distance in self.car.sensor_distances:
            self.assertEqual(distance, 0, "Начальное расстояние должно быть 0")
        
        print("✅ Сенсоры инициализированы правильно")

class TestIntegration(unittest.TestCase):
    """Интеграционные тесты"""
    
    def setUp(self):
        print(f"\n{'='*60}")
        print(f"Тест: {self._testMethodName}")
        print(f"{'='*60}")
    
    def test_01_car_with_neural_network(self):
        """Автомобиль с нейронной сетью должен работать"""
        nn = NeuralNetwork(7, 10, 4)
        car = Car(100, 100, 45, nn, car_id=1)
        
        print(f"Создан автомобиль с сетью 7-10-4")
        print(f"Позиция: ({car.x}, {car.y}), Угол: {car.angle}°")
        
        track_data = {
            'points': [(0, 0), (200, 0), (200, 200)],
            'width': 50,
            'closed': False
        }
        
        print("Обновление автомобиля...")
        for i in range(5):
            car.update(track_data)
            print(f"  Шаг {i+1}: Позиция ({car.x:.1f}, {car.y:.1f}), "
                  f"Угол {car.angle:.1f}°, Жив: {car.is_alive}")
        
        self.assertTrue(car.is_alive, "Автомобиль должен остаться живым")
        
        inputs = car.get_inputs()
        print(f"Входные данные сети: {[f'{x:.3f}' for x in inputs]}")
        self.assertEqual(len(inputs), 7, "Должно быть 7 входов (5 сенсоров + скорость + угол)")
        
        print("✅ Автомобиль с нейронной сетью работает корректно")

def print_test_statistics(results):
    """Вывод статистики тестов"""
    print("\n" + "="*60)
    print("СТАТИСТИКА ТЕСТОВ")
    print("="*60)
    
    total_tests = results.testsRun
    failures = len(results.failures)
    errors = len(results.errors)
    skipped = len(results.skipped)
    successful = total_tests - failures - errors - skipped
    
    print(f"Всего тестов: {total_tests}")
    print(f"✅ Успешно: {successful}")
    print(f"❌ Провалено: {failures}")
    print(f"⚠️  Ошибок: {errors}")
    print(f"⏭️  Пропущено: {skipped}")
    
    if failures > 0:
        print("\nПРОВАЛЕННЫЕ ТЕСТЫ:")
        for test, traceback in results.failures:
            print(f"\n❌ {test}")
            print(traceback)
    
    if errors > 0:
        print("\nОШИБКИ В ТЕСТАХ:")
        for test, traceback in results.errors:
            print(f"\n🚨 {test}")
            print(traceback)
    
    success_rate = (successful / total_tests * 100) if total_tests > 0 else 0
    print(f"\n📊 Успешность: {success_rate:.1f}%")
    
    return successful == total_tests

if __name__ == '__main__':
    loader = unittest.TestLoader()
    
    test_suite = unittest.TestSuite()
    test_suite.addTests(loader.loadTestsFromTestCase(TestNeuralNetwork))
    test_suite.addTests(loader.loadTestsFromTestCase(TestGeneticAlgorithm))
    test_suite.addTests(loader.loadTestsFromTestCase(TestCarPhysics))
    test_suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    print("="*70)
    print("ЗАПУСК ЮНИТ-ТЕСТОВ AI RACING SIMULATOR")
    print("="*70)
    
    runner = unittest.TextTestRunner(verbosity=0) 
    results = runner.run(test_suite)
    
    all_passed = print_test_statistics(results)
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print("💥 НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ")
    print("="*70)
