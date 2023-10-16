import random
import timeit
import matplotlib.pyplot as plt
import numpy as np
import queue

MAX_NUMBER = 1000000

def generate_fully_connected_graph(num_vertices, max_edge_weight):
    graph = [[] for _ in range(num_vertices)]  # Initialize an empty adjacency list

    # Generate edges between all pairs of vertices
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            weight = random.randint(1, max_edge_weight)
            # Add an edge to both vertices in the adjacency list
            graph[i].append((j, weight))
            graph[j].append((i, weight))

    return graph

def generate_non_fully_connected_graph(num_vertices, max_edge_weight):
    graph = [[] for _ in range(num_vertices)]  # Initialize an empty adjacency list
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            if random.randint(0, 1) == 1:
                weight = random.randint(1, max_edge_weight)
                # Add an edge to both vertices in the adjacency list
                graph[i].append((j, weight))
                graph[j].append((i, weight))
    return graph

def dijkstra_adjacency_list(graph, source):
    V = len(graph)
    distances = [MAX_NUMBER] * V
    distances[source] = 0

    # Initialize a priority queue (min heap) to keep track of vertices and their distances.
    pq = queue.PriorityQueue()
    pq.put((0, source))  # Tuple: (distance, vertex), priority queue sorts according to first element of each tupple, so distance has to be the first followed by vertex

    while not pq.empty():
        dist_u, u = pq.get()

        # If the vertex has been processed already, skip it.
        if distances[u] < dist_u:
            continue

        for v, weight in graph[u]:
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                pq.put((distances[v], v))

    return distances

def measure_NFC_runtime(num_vertices_list, max_edge_weight, num_runs):
    NFC_runtimes = np.zeros(len(num_vertices_list))
    for i, num_vertices in enumerate(num_vertices_list):
        #Non fully connected graph
        total_runtime = 0
        for _ in range(num_runs):
            non_fully_connected_graph = generate_non_fully_connected_graph(num_vertices, max_edge_weight)
            source = random.randint(0, num_vertices - 1)
            runtime = timeit.timeit(lambda: dijkstra_adjacency_list(non_fully_connected_graph, source), number=1)
            total_runtime += runtime
        avg_runtime = total_runtime / num_runs
        NFC_runtimes[i] = avg_runtime
        print(f"Vertices: {num_vertices}, Average Runtime: {avg_runtime} seconds")
    return NFC_runtimes

def measure_FC_runtime(num_vertices_list, max_edge_weight, num_runs):
    FC_runtimes = np.zeros(len(num_vertices_list))
    for i, num_vertices in enumerate(num_vertices_list):
        #Fully connected graph
        total_runtime = 0
        for _ in range(num_runs):
            fully_connected_graph = generate_fully_connected_graph(num_vertices, max_edge_weight)
            source = random.randint(0, num_vertices - 1)
            runtime = timeit.timeit(lambda: dijkstra_adjacency_list(fully_connected_graph, source), number=1)
            total_runtime += runtime
        avg_runtime = total_runtime / num_runs
        FC_runtimes[i] = avg_runtime
        print(f"Vertices: {num_vertices}, Average Runtime: {avg_runtime} seconds")
    return FC_runtimes

def theroretical_runtime(num_vertices_list, scaling_factor):
    return [scaling_factor * num_vertices**2 for num_vertices in num_vertices_list]

if __name__ == "__main__":
    num_vertices_list = [10, 50, 100, 200, 300, 400, 500]
    max_edge_weight = 100
    num_runs = 5

    empirical_NFC_runtimes = measure_NFC_runtime(num_vertices_list, max_edge_weight, num_runs)
    empirical_FC_runtimes = measure_FC_runtime(num_vertices_list, max_edge_weight, num_runs)

    max_empirical_runtime = (max(empirical_NFC_runtimes) + max(empirical_FC_runtimes))/2
    scaling_factor = max_empirical_runtime / (num_vertices_list[-1]**2)

    theroretical_runtimes = theroretical_runtime(num_vertices_list, scaling_factor)

    plt.plot(num_vertices_list, empirical_NFC_runtimes, marker="o", label="NFC Empirical Runtime")
    plt.plot(num_vertices_list, empirical_FC_runtimes, marker="o", label="FC Empirical Runtime")
    plt.plot(num_vertices_list, theroretical_runtimes, linestyle="--", label="Theoretical O(V^2)")
    plt.title("AdjList: Runtime vs. Number of Vertices")
    plt.xlabel("Number of Vertices")
    plt.ylabel("Average Runtime (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()
