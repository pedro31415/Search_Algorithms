from collections import deque
import heapq
import math
import tkinter as tk
from tkinter import ttk
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

graph = {
    "oradea": { "zerind": 71, "sibiu": 151 },
    "zerind": { "arad": 75, "oradea": 71 },
    "arad": { "timisoara": 118, "sibiu": 140, "zerind": 75 },
    "timisoara": {"arad": 118, "lugoj": 111},
    "lugoj": { "timisoara": 111, "mehadia": 70 },
    "mehadia": { "lugoj": 70, "drobeta": 75 },
    "drobeta": { "craiova": 120, "mehadia": 75 },
    "sibiu": { "arad": 140, "rimnicu_vilcea": 80, "fagaras": 99, "oradea": 151 },
    "rimnicu_vilcea": { "craiova": 146, "pitesti": 97, "sibiu": 80 },
    "craiova": { "drobeta": 120, "rimnicu_vilcea": 146, "pitesti": 138 },
    "fagaras": { "sibiu": 99, "bucharest": 211 },
    "pitesti": { "rimnicu_vilcea": 97, "craiova": 138, "bucharest": 101 },
    "bucharest": { "fagaras": 211, "pitesti": 101, "giurgiu": 90, "urziceni": 85 },
    "urziceni": { "bucharest": 85, "hirsova": 98, "vaslui": 142 },
    "neamt": { "iasi": 87 },
    "iasi": { "neamt": 87, "vaslui": 92 },
    "giurgiu": { "bucharest": 90 },
    "hirsova": { "urziceni": 98, "eforie": 86 },
    "eforie": { "hirsova": 86 },
    "vaslui": {"iasi": 92, "urziceni": 142}
}

# 1.1 -> Breadth First Search
# FIFO
def bfs_2(graph, inital, goal):
    queue_2 = deque([(inital, [inital], 0)])

    while queue_2:
        node, result, cost = queue_2.popleft()

        if node == goal:
            return result, cost

        for neighbour, weight in graph.get(node, {}).items():
            if neighbour not in result:
                queue_2.append((neighbour, result + [neighbour], cost + weight))
    return None



# 1.2 -> Uniform Cost Search
#priority queue
def ucs(graph, result, goal):
    queue = [(0, result, [])]
    visited = set()

    while queue:
        cost, node, path = heapq.heappop(queue)

        if node in visited:
            continue
        visited.add(node)

        path = path + [node]

        if node == goal:
            return path, cost

        for neighbour, weight in graph.get(node, {}).items():
            if neighbour not in visited:
                heapq.heappush(queue, (cost + weight, neighbour, path))
    return None



# 1.3 ->  Depth First Search
# FIFO

def dfs(graph, result, goal):
    stack = [(result, [result], 0),]
    visited = set()


    while stack:
        node, path, cost = stack.pop()

        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path, cost

        for neighbour, weight in graph.get(node, {}).items():
            if neighbour not in visited:
                stack.append((neighbour, path + [neighbour], cost + weight))
    return None

# 1.4 -> Depth Limitedd Search
def dls(graph, result, goal, limit, path=None, cost=0):
    if path is None:
        path = [result]

    if result == goal:
        return path, cost

    if limit <= 0:
        return None

    for neighbor, weight in graph.get(result, {}).items():
        if neighbor not in path:
            result = dls(graph, neighbor, goal, limit - 1, path + [neighbor], cost + weight)
            if result:
                return result
    return None

# 1.5 -> Irative Deepening Depth First Search
def iddfs(graph, start, goal, max_depth=10):
    for depth in range(max_depth + 1):
        result= dls(graph, start, goal, depth)
        if result:
            return result
    return None

# 1.6 Bidirectional Search
def bidirectional_bfs(graph, start, goal):
    if start == goal:
        return [start], 0

    forward_queue = deque([(start, [start], 0)])
    backward_queue = deque([(goal, [goal], 0)])

    forward_visited = {start: (0, [start])}
    backward_visited = {goal: (0, [goal])}

    while forward_queue and backward_queue:
        
        node, path, cost = forward_queue.popleft()
        for neighbor, weight in graph.get(node, {}).items():
            if neighbor not in forward_visited:
                new_cost = cost + weight
                new_path = path + [neighbor]
                forward_visited[neighbor] = (new_cost, new_path)
                forward_queue.append((neighbor, new_path, new_cost))

                if neighbor in backward_visited:
                    back_cost, back_path = backward_visited[neighbor]
                    return new_path + back_path[::-1][1:], new_cost + back_cost

        node, path, cost = backward_queue.popleft()
        for neighbor, weight in graph.get(node, {}).items():
            if neighbor not in backward_visited:
                new_cost = cost + weight
                new_path = path + [neighbor]
                backward_visited[neighbor] = (new_cost, new_path)
                backward_queue.append((neighbor, new_path, new_cost))

                if neighbor in forward_visited:
                    front_cost, front_path = forward_visited[neighbor]
                    return front_path + new_path[::-1][1:], front_cost + new_cost

    return None, float('inf')  



# 2.1 Greedy Best First Search

# problem algorithms Gredy Best First Search -> He need the heurisct value before the example

def greedy_best_first_search(graph, heuristics, start, goal):

    if start == goal:
        return [start]

    priority_queue = [(heuristics[start], start, [start], 0)]

    visited = set()

    while priority_queue:
        _, current_node, path, cost = heapq.heappop(priority_queue)

        if current_node == goal:
            return path, cost

        visited.add(current_node)

        for neighbor, weidth in graph.get(current_node, {}).items():
            if neighbor not in visited:
                new_path = path + [neighbor]
                new_cost = cost + weidth
                heapq.heappush(priority_queue, (heuristics[neighbor], neighbor, new_path, new_cost))
    return None

heuristic = {
    "oradea": 380,
    "zerind": 374,
    "arad": 366,
    "sibiu": 253,
    "timisoara": 329,
    "lugoj": 244,
    "mehadia": 241,
    "drobeta": 242,
    "craiova": 160,
    "rimnicu_vilcea": 193,
    "fagaras": 178,
    "pitesti": 98,
    "bucharest": 0,
    "giurgiu": 77,
    "urziceni": 80,
    "hirsova": 151,
    "eforie": 161,
    "vaslui": 199,
    "iasi": 226,
    "neamt": 234
}

# 2.1 A *
coordinates = {
    "oradea": (1, 5),
    "zerind": (2, 6),
    "arad": (3, 4),
    "sibiu": (5, 5),
    "timisoara": (4, 2),
    "lugoj": (6, 3),
    "mehadia": (7, 2),
    "drobeta": (8, 1),
    "craiova": (9, 2),
    "rimnicu_vilcea": (7, 5),
    "fagaras": (8, 6),
    "pitesti": (10, 4),
    "bucharest": (12, 5),
    "giurgiu": (13, 3),
    "urziceni": (12, 6),
    "vaslui": (14, 7),
    "iasi": (13, 8),
    "neamt": (12, 9),
    "hirsova": (14, 5),
    "eforie": (15, 4)
}

def euclidean_distance(node1, node2):
    x1, y1 = coordinates[node1]
    x2,y2 = coordinates[node2]
    return math.sqrt((x2 - x1) **2 + (y2 - y1) **2)

def a_start_search_dynamic(graph, start, goal):
    priority_queue = [(euclidean_distance(start,goal), 0, start,[start])]
    heapq.heapify(priority_queue)
    visited = set()

    while priority_queue:
        f, g, current_node, path = heapq.heappop(priority_queue)

        if current_node == goal:
            return path, g
        if current_node in visited:
            continue
        visited.add(current_node)

        for neighbor, cost in graph[current_node].items():
            if neighbor not in visited:
                new_g = g + cost
                new_h = euclidean_distance(neighbor, goal)
                new_f = new_g + new_h
                heapq.heappush(priority_queue, (new_f, new_g, neighbor, path + [neighbor]))
    return None


# draw graph

def draw_graph(path):
    G = nx.Graph()

    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)
    pos = nx.kamada_kawai_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(10,6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    plt.show()

# execute search

def execute_search(search_function):
    start = start_var.get()
    goal = goal_var.get()

    if start and goal:
        result = search_function(graph, start, goal)
        if result:
            path, cost = result
            result_label.config(text=f"Caminho: {'->'.join(path)}\nCusto: {cost}")
            draw_graph(path)
        else:
            result_label.config(text="Nenhum caminho encontrado.")

# Graphics Interface

root = tk.Tk()
root.title("Algoritmos de Busca em Grafos")

frame = ttk.Frame(root, padding=20)
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

start_label = ttk.Label(frame, text="Cidade Inicial:")
start_label.grid(row=0, column=0)
start_var = tk.StringVar()
start_entry = ttk.Combobox(frame, textvariable=start_var, values=list(graph.keys()))
start_entry.grid(row=0, column=1)

goal_label = ttk.Label(frame, text="Cidade Destino:")
goal_label.grid(row=1, column=0)
goal_var = tk.StringVar()
goal_entry = ttk.Combobox(frame, textvariable=goal_var, values=list(graph.keys()))
goal_entry.grid(row=1, column=1)

result_label = ttk.Label(frame, text="")
result_label.grid(row=3, column=0, columnspan=2)

search_methods = {
    "BFS": bfs_2,
    "UCS": ucs,
    "DFS": dfs,
    "DLS": lambda g, s, d: dls(g, s, d, 6),
    "Bidirectional BFS": bidirectional_bfs,
    "IDDFS": iddfs,
    "Greedy Best First": lambda g, s, d: greedy_best_first_search(g, heuristic, s, d),
    "A*": a_start_search_dynamic
}

# Same Search method

search_method_label = ttk.Label(frame, text="Método de Busca:")
search_method_label.grid(row=2, column=0)
search_method_var = tk.StringVar(value="BFS")
search_method_combobox = ttk.Combobox(frame, textvariable=search_method_var, values=list(search_methods.keys()))
search_method_combobox.grid(row=2, column=1)

def execute_selected_search():
    selected_method = search_method_var.get()
    search_function = search_methods.get(selected_method)

    if search_methods:
        execute_search(search_function)

search_button = ttk.Button(frame, text="Buscar", command=execute_selected_search)
search_button.grid(row=3, column=0, columnspan=2, pady=5)

# Result
result_label = ttk.Label(frame, text="")
result_label.grid(row=4, column=0, columnspan=2)

# image
image_path = "C:\\Users\\Pedro\\Downloads\\ProjetoIA\\img\\grafosIA.png"
img = Image.open(image_path)
img = img.resize((400,400), Image.LANCZOS)
photo = ImageTk.PhotoImage(img)

# frame image
frame = tk.Frame(root)
frame.grid(row=1, column=0, padx=20, pady=20)

# plot image
image_label = tk.Label(root, image=photo)
image_label.grid(row=1, column=1, padx=20, pady=20)

root.mainloop()
