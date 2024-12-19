import sys
import copy
import heapq
import random

import networkx as nx
import matplotlib.pyplot as plt
import time  # Импортируем модуль time для измерения времени выполнения

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QLabel, QTextEdit, QHBoxLayout, QMessageBox,
    QTableWidget, QTableWidgetItem, QSpinBox, QHeaderView, QFileDialog, QPlainTextEdit
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Node:
    def __init__(self, matrix, path, level, path_cost, lower_bound, parent=None):
        self.matrix = matrix  # Текущая редуцированная матрица
        self.path = path  # Текущий путь (список индексов городов, 0-based)
        self.level = level  # Уровень узла (количество посещённых городов)
        self.path_cost = path_cost  # Сумма путевых расходов
        self.lower_bound = lower_bound  # Сумма редукционных расходов
        self.total_cost = self.path_cost + self.lower_bound  # Общая стоимость узла
        self.parent = parent  # Родительский узел

    def __lt__(self, other):
        return self.total_cost < other.total_cost


def reduce_matrix(matrix):
    n = len(matrix)
    reduced_matrix = copy.deepcopy(matrix)
    reduction_cost = 0

    # Строковая редукция
    for i in range(n):
        row = [x for x in reduced_matrix[i] if x != float('inf')]
        if row:
            min_val = min(row)
            if min_val > 0:
                reduction_cost += min_val
                for j in range(n):
                    if reduced_matrix[i][j] != float('inf'):
                        reduced_matrix[i][j] -= min_val

    # Столбцовая редукция
    for j in range(n):
        col = [reduced_matrix[i][j] for i in range(n) if reduced_matrix[i][j] != float('inf')]
        if col:
            min_val = min(col)
            if min_val > 0:
                reduction_cost += min_val
                for i in range(n):
                    if reduced_matrix[i][j] != float('inf'):
                        reduced_matrix[i][j] -= min_val

    return reduced_matrix, reduction_cost


def get_submatrix(matrix, i, j):
    """
    Создаёт подматрицу, исключая посещение города j после города i.
    """
    n = len(matrix)
    submatrix = copy.deepcopy(matrix)
    for k in range(n):
        submatrix[i][k] = float('inf')  # Запрет на возвращение из i
        submatrix[k][j] = float('inf')  # Запрет на повторное посещение j
    submatrix[j][0] = float('inf')  # Запрет на возвращение в начальный город до завершения пути
    return submatrix


def solve_tsp(matrix):
    n = len(matrix)
    initial_matrix, initial_reduction = reduce_matrix(matrix)
    root = Node(matrix=initial_matrix, path=[0], level=0, path_cost=0, lower_bound=initial_reduction)
    pq = []
    heapq.heappush(pq, root)
    best_cost = float('inf')
    best_path = []
    search_tree = nx.DiGraph()

    search_tree.add_node(tuple(root.path), label=' -> '.join(map(lambda x: str(x+1), root.path)))

    while pq:
        current = heapq.heappop(pq)
        if current.total_cost >= best_cost:
            continue
        if current.level == len(matrix) -1:
            last = current.path[-1]
            if matrix[last][0] != float('inf'):
                total_cost = current.path_cost + matrix[last][0]
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = current.path + [0]
            continue

        current_city = current.path[-1]
        for next_city in range(len(matrix)):
            if matrix[current_city][next_city] != float('inf') and next_city not in current.path:
                new_path = current.path + [next_city]
                new_path_cost = current.path_cost + matrix[current_city][next_city]
                new_matrix = get_submatrix(current.matrix, current_city, next_city)
                reduced_matrix, reduction_cost = reduce_matrix(new_matrix)
                new_lower_bound = reduction_cost
                total_cost = new_path_cost + new_lower_bound

                if total_cost < best_cost:
                    child = Node(matrix=new_matrix, path=new_path, level=current.level +1,
                                path_cost=new_path_cost, lower_bound=new_lower_bound, parent=current)
                    heapq.heappush(pq, child)
                    search_tree.add_edge(tuple(current.path), tuple(child.path), weight=matrix[current_city][next_city])
                    search_tree.nodes[tuple(child.path)]['label'] = ' -> '.join(map(lambda x: str(x+1), child.path))

    if best_path:
        best_path = [city +1 for city in best_path]
    return best_path, best_cost, search_tree

def hierarchical_layout(G):
    levels = {node:len(node) for node in G.nodes()}
    sorted_nodes = sorted(G.nodes(), key=lambda x: levels[x])
    level_dict = {}
    for node in sorted_nodes:
        level = levels[node]
        if level not in level_dict:
            level_dict[level] = []
        level_dict[level].append(node)

    pos = {}
    for level in sorted(level_dict.keys()):
        nodes = level_dict[level]
        num_nodes = len(nodes)
        for i, node in enumerate(nodes):
            x = i/(num_nodes-1) if num_nodes>1 else 0.5
            y = -level
            pos[node] = (x,y)
    return pos

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)

class SearchTreeWindow(QWidget):
    def __init__(self, search_tree):
        super().__init__()
        self.setWindowTitle("Дерево поиска алгоритма ветвей и границ")
        self.search_tree = search_tree
        self.showFullScreen()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.buttons_layout = QHBoxLayout()
        self.toggle_text_button = QPushButton("Показать дерево в текстовом формате")
        self.toggle_text_button.clicked.connect(self.toggle_text_view)
        self.buttons_layout.addWidget(self.toggle_text_button)
        layout.addLayout(self.buttons_layout)

        self.canvas = MplCanvas(self, width=12, height=10, dpi=100)
        layout.addWidget(self.canvas)
        self.text_view = QPlainTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setVisible(False)
        layout.addWidget(self.text_view)

        self.setLayout(layout)
        self.plot_search_tree()

    def plot_search_tree(self):
        ax = self.canvas.fig.add_subplot(111)
        ax.clear()
        G = self.search_tree

        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, 'Дерево поиска пусто', horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return

        pos = hierarchical_layout(G)
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=500, ax=ax)
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_color='gray', ax=ax)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
        ax.set_title("Дерево поиска алгоритма ветвей и границ", fontsize=14)
        ax.axis('off')
        self.canvas.draw()

    def show_tree_as_text(self):
        G = self.search_tree
        text_lines = []
        if G.number_of_nodes() == 0:
            text_lines.append("Дерево поиска пусто.")
        else:
            levels = {node:len(node) for node in G.nodes()}
            sorted_nodes = sorted(G.nodes(), key=lambda x: levels[x])
            level_dict = {}
            for node in sorted_nodes:
                level = levels[node]
                if level not in level_dict:
                    level_dict[level] = []
                level_dict[level].append(node)

            for level in sorted(level_dict.keys()):
                indent = "  " * (level-1)
                for node in level_dict[level]:
                    label = G.nodes[node].get('label', str(node))
                    text_lines.append(f"{indent}{label}")

        self.text_view.setPlainText("\n".join(text_lines))

    def toggle_text_view(self):
        if self.text_view.isVisible():
            self.text_view.setVisible(False)
            self.canvas.setVisible(True)
        else:
            self.text_view.setVisible(True)
            self.canvas.setVisible(False)
            self.show_tree_as_text()

class TSPWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Задача Коммивояжёра - Метод Ветвей и Границ")
        self.setGeometry(100, 100, 1600, 900)  # Увеличение размера главного окна
        self.search_tree = None  # Инициализация переменной для хранения дерева поиска
        self.initUI()

    def initUI(self):
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной вертикальный layout
        main_layout = QVBoxLayout()

        # Верхняя часть: настройки матрицы
        settings_layout = QHBoxLayout()

        self.size_label = QLabel("Количество городов")
        settings_layout.addWidget(self.size_label)

        self.size_spinbox = QSpinBox()
        self.size_spinbox.setMinimum(2)
        self.size_spinbox.setMaximum(50)  # Ограничение для производительности
        self.size_spinbox.setValue(5)
        self.size_spinbox.valueChanged.connect(self.on_city_count_changed)
        settings_layout.addWidget(self.size_spinbox)

        self.generate_button = QPushButton("Создать матрицу")
        self.generate_button.clicked.connect(self.generate_matrix_table)
        settings_layout.addWidget(self.generate_button)

        main_layout.addLayout(settings_layout)

        # Таблица для ввода матрицы расстояний
        self.table_widget = QTableWidget()
        main_layout.addWidget(self.table_widget)

        self.random_fill_button = QPushButton("Заполнить случайными значениями")
        self.random_fill_button.clicked.connect(self.fill_random_matrix)
        settings_layout.addWidget(self.random_fill_button)

        # Кнопки для запуска алгоритма и отображения дерева поиска
        buttons_layout = QHBoxLayout()

        self.run_button = QPushButton("Запустить")
        self.run_button.clicked.connect(self.run_tsp_with_time)  # Подключаем новую функцию
        buttons_layout.addWidget(self.run_button)

        self.show_tree_button = QPushButton("Показать дерево поиска")
        self.show_tree_button.clicked.connect(self.show_search_tree)
        self.show_tree_button.setEnabled(False)  # Активируется только после решения задачи
        buttons_layout.addWidget(self.show_tree_button)

        self.save_button = QPushButton("Сохранить результаты")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        buttons_layout.addWidget(self.save_button)

        self.load_button = QPushButton("Загрузить из файла")
        self.load_button.clicked.connect(self.load_from_file)
        self.load_button.setVisible(False)
        settings_layout.addWidget(self.load_button)

        self.save_condition_button = QPushButton("Сохранить условие")
        self.save_condition_button.clicked.connect(self.save_condition)
        # Активируем когда матрица будет хоть чем-то заполнена
        self.save_condition_button.setEnabled(True)
        settings_layout.addWidget(self.save_condition_button)

        main_layout.addLayout(buttons_layout)
   # Разделение на две части: текстовые результаты и графы
        content_layout = QHBoxLayout()

        # Левая часть: результаты
        results_layout = QVBoxLayout()


        self.result_label = QLabel("Результаты:")
        results_layout.addWidget(self.result_label)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        results_layout.addWidget(self.result_text)

        content_layout.addLayout(results_layout, 1)

        # Правая часть: графы
        graphs_layout = QVBoxLayout()

        # Граф переходов путей
        self.transition_canvas = MplCanvas(self, width=4, height=6, dpi=80)  # Увеличение размера графа
        graphs_layout.addWidget(self.transition_canvas)

        content_layout.addLayout(graphs_layout, 2)  # Увеличение пространства для графов

        main_layout.addLayout(content_layout)

        central_widget.setLayout(main_layout)

        # Инициализация таблицы
        self.generate_matrix_table()




    def on_city_count_changed(self):
        n = self.size_spinbox.value()
        self.load_button.setVisible(n > 5)


    def generate_matrix_table(self):
        n = self.size_spinbox.value()
        self.table_widget.clear()
        self.table_widget.setRowCount(n)
        self.table_widget.setColumnCount(n)
        self.table_widget.setHorizontalHeaderLabels([f"Город {i + 1}" for i in range(n)])
        self.table_widget.setVerticalHeaderLabels([f"Город {i + 1}" for i in range(n)])
        # Установка размеров колонок
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # Заполнение диагонали значениями infinity
        for i in range(n):
            item = QTableWidgetItem("∞")
            item.setFlags(Qt.ItemIsEnabled)  # Запрет редактирования диагональных ячеек
            self.table_widget.setItem(i, i, item)
        QMessageBox.information(self, "Матрица создана",
                                f"Создана матрица {n}x{n}. Введите расстояния между городами.\nДиагональные элементы установлены в ∞.")

    def fill_random_matrix(self):
        n = self.size_spinbox.value()
        for i in range(n):
            for j in range(n):
                if i != j:
                    value = random.randint(1, 100)
                    self.table_widget.setItem(i, j, QTableWidgetItem(str(value)))
                else:
                    self.table_widget.setItem(i, j, QTableWidgetItem("∞"))

    def load_from_file(self):
        n = self.size_spinbox.value()
        if n < 6:
            QMessageBox.warning(self, "Загрузка невозможна", "Количество городов должно быть больше пяти для загрузки из файла.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл с матрицей", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_path:
            try:
                matrix = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.read().strip().split('\n')
                    if len(lines) != n:
                        raise ValueError("Число строк в файле не совпадает с размером матрицы.")
                    for i, line in enumerate(lines):
                        parts = line.strip().split()
                        if len(parts) != n:
                            raise ValueError(f"Число значений в строке {i+1} не совпадает с размером матрицы.")
                        row = []
                        for val in parts:
                            if val.lower() in ['inf', '∞']:
                                row.append(float('inf'))
                            else:
                                v = float(val)
                                if v < 0:
                                    raise ValueError("Отрицательные расстояния недопустимы.")
                                row.append(v)
                        row[i] = float('inf')  # Диагональ должна быть ∞
                        matrix.append(row)

                # Заполняем таблицу значениями из matrix
                for i in range(n):
                    for j in range(n):
                        val = "∞" if matrix[i][j] == float('inf') else str(matrix[i][j])
                        item = QTableWidgetItem(val)
                        if i == j:
                            item.setFlags(Qt.ItemIsEnabled)
                        self.table_widget.setItem(i, j, item)

                QMessageBox.information(self, "Успех", "Матрица успешно загружена из файла.")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка загрузки", f"Произошла ошибка при загрузке матрицы:\n{str(e)}")



    def run_tsp_with_time(self):
        try:
            matrix = self.get_matrix_from_table()
            start_time = time.perf_counter()  # Начало измерения времени
            path, cost, search_tree = solve_tsp(matrix)
            end_time = time.perf_counter()  # Конец измерения времени
            elapsed_time = end_time - start_time  # Время выполнения

            self.search_tree = search_tree  # Сохраняем дерево поиска

            # Вывод результатов
            if path:
                path_str = ' -> '.join(map(str, path))
                result_str = f"Оптимальный путь: {path_str}\nОбщая стоимость: {cost}\nВремя выполнения: {elapsed_time:.6f} секунд\n\n"
            else:
                result_str = "Не найден оптимальный путь.\n\n"
            result_str += "Граф оптимального пути доступен на графике."
            self.result_text.setText(result_str)

            # Визуализация графа переходов
            self.plot_transition_graph(path, cost, matrix)

            # Активируем кнопку для отображения дерева поиска
            self.show_tree_button.setEnabled(True)
            self.save_button.setEnabled(True)

        except ValueError as ve:
            QMessageBox.warning(self, "Некорректный ввод", f"Произошла ошибка при вводе данных:\n{ve}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{str(e)}")

    def get_matrix_from_table(self):
        n = self.table_widget.rowCount()
        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                item = self.table_widget.item(i, j)
                if i == j:
                    row.append(float('inf'))
                    continue
                if item is None or item.text().strip() == "":
                    raise ValueError(f"Ячейка ({i + 1}, {j + 1}) пуста.")
                text = item.text().strip()
                if text.lower() in ['inf', 'infty', '∞']:
                    row.append(float('inf'))
                else:
                    try:
                        value = float(text)
                        if value < 0:
                            raise ValueError(f"Расстояние не может быть отрицательным (ячейка ({i + 1}, {j + 1})).")
                        row.append(value)
                    except ValueError:
                        raise ValueError(f"Некорректное значение в ячейке ({i + 1}, {j + 1}): '{text}'")
            matrix.append(row)
        return matrix

    def plot_transition_graph(self, path, cost, matrix):
        self.transition_canvas.fig.clf()
        ax = self.transition_canvas.fig.add_subplot(111)
        if not path:
            ax.text(0.5, 0.5, 'Оптимальный путь не найден', horizontalalignment='center', verticalalignment='center')
            self.transition_canvas.draw()
            return
        G = nx.DiGraph()

        # Добавляем узлы
        n = len(matrix)
        for i in range(n):
            G.add_node(i + 1)

        # Добавляем все рёбра
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != float('inf') and i != j:
                    G.add_edge(i + 1, j + 1, weight=matrix[i][j])

        pos = nx.circular_layout(G)  # Расположение узлов по кругу

        # Рисуем все рёбра
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray', alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=12, font_color='black', ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

        # Рисуем оптимальный путь
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2, arrowstyle='->', arrowsize=20,
                               ax=ax)

        ax.set_title(f"Оптимальный путь: {' -> '.join(map(str, path))} с стоимостью {cost}", fontsize=14)
        ax.axis('off')
        self.transition_canvas.draw()

    def show_search_tree(self):
        if self.search_tree is None:
            QMessageBox.warning(self, "Дерево поиска не найдено", "Пожалуйста, сначала решите задачу коммивояжёра.")
            return
        self.tree_window = SearchTreeWindow(self.search_tree)
        self.tree_window.show()


    def save_results(self):
        results = self.result_text.toPlainText().strip()
        if not results:
            QMessageBox.warning(self, "Нет результатов", "Сначала запустите алгоритм, чтобы получить результаты.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить результаты", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(results)
                QMessageBox.information(self, "Сохранение успешно", f"Результаты успешно сохранены в файл:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения", f"Произошла ошибка при сохранении файла:\n{str(e)}")

    def save_condition(self):
        # Сохранение текущей матрицы условий задачи
        n = self.table_widget.rowCount()
        if n == 0:
            QMessageBox.warning(self, "Нет данных", "Сначала создайте или загрузите матрицу.")
            return
        # Проверим, что таблица не пуста
        try:
            matrix = self.get_matrix_from_table()  # Используем уже написанную функцию
        except ValueError as ve:
            QMessageBox.warning(self, "Некорректная матрица", f"Проверьте введённые данные:\n{ve}")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить условие задачи", "",
                                                   "Text Files (*.txt);;All Files (*)", options=options)
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for i in range(n):
                        row_str = []
                        for j in range(n):
                            val = matrix[i][j]
                            if val == float('inf'):
                                row_str.append("∞")
                            else:
                                row_str.append(str(val))
                        f.write(" ".join(row_str) + "\n")
                QMessageBox.information(self, "Сохранение успешно",
                                        f"Условие задачи успешно сохранено в файл:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения", f"Произошла ошибка при сохранении файла:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    window = TSPWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
