# Транспортный бенчмарк в Новинске

Таксист, передвигается по сетке из `N x N` ячеек, он способен перемещаться на одну ячейку вверх, вниз, влево или вправо (если он не находится на краю). По этой сетке разбросаны пассажиры и их пункты назначения.

- **Размер сетки:** Для этой задачи `N = 7` или `N = 5`.
- **Особенности сетки:** Определенные ячейки содержат пассажиров, в то время как другие ячейки являются местами назначения.

### Задачи

#### 1. Описание марковского процесса

Ваша первая задача — описать марковский процесс для созданной среды:
- **Места пассажиров:** Всего существует 4 возможных места для пассажиров (A, B, C, D).
- **Пункты назначения:** Существует 4 возможных пункта назначения (a, b, c, d).
- Пассажир может находиться в любом из четырех мест и желать доехать до любого из четырех пунктов назначения.
- Таксист автоматически забирает пассажира, подъезжая к его месту.
- Для высадки пассажира в нужном месте таксисту необходимо выбрать соответствующее действие.
- Места `A, B, C, D` и `a, b, c, d` фиксированы.
- В начале эпизода случайным образом определяется пара (пассажир, место назначения).

#### 2. Реализация итеративного алгоритма

Вторая задача требует реализации итеративного алгоритма для одной из постановок задачи и описанного МПП (Марковского Процесса Принятия Решений):
- **Вариант 1:** Построение маршрута перевоза одного пассажира (`mdp_taxi_v1`). Здесь предполагается, построение сратегии для перевозки одного пассажира, mdp основано на среде Taxi-v3.
- **Вариант 2:** Определение порядка развоза пассажиров (`mdp_taxi_v2`). В данной постановке необходимо выбрать оптимальный порядок перевозки пассажиров, чтобы сэкономить топливо. 
- **Вариант 3:** Реализация кода, кодирующего МПП из задания 1, и предложение решения для этой постановки.

Для вариант 1 и варианта 2 - mdp и среды уже готовы, основная задача - реализация алгоритма, решающего целевую задачу постановки.

#### 3. Демонстрация работы обученной стратегии

Реализуйте инференс, демонстрирующий работу обученной стратегии на различных вариантах поля:
- Отрисуйте маршрут, по которому двигается агент.
- Покажите график, иллюстрирующий, как с итерациями меняется
