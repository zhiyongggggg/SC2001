import random
import timeit
import matplotlib.pyplot as plt
import numpy as np
import queue
import math

MAX_NUMBER = 1000000

def generate_fully_connected_list(num_vertices, max_edge_weight):
    graph = [[] for _ in range(num_vertices)]  # Initialize an empty adjacency list

    # Generate edges between all pairs of vertices
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            weight = random.randint(1, max_edge_weight)
            # Add an edge to both vertices in the adjacency list
            graph[i].append((j, weight))
            graph[j].append((i, weight))

    return graph

def generate_non_fully_connected_list(num_vertices, max_edge_weight):
    graph = [[] for _ in range(num_vertices)]  # Initialize an empty adjacency list
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            if random.random() <= 0.1:
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

#=============================================================================

def adjacency_list_to_matrix(adjacency_list, num_vertices):
    # Initialize an empty adjacency matrix filled with zeros
    adjacency_matrix = [[0] * num_vertices for _ in range(num_vertices)]

    # Fill in the adjacency matrix based on the adjacency list
    for vertex in range(num_vertices):
        for neighbor, weight in adjacency_list[vertex]:
            adjacency_matrix[vertex][neighbor] = weight
            adjacency_matrix[neighbor][vertex] = weight 

    return adjacency_matrix

def dijkstra_adjacency_matrix(graph, source):
    V = len(graph)
    #Initializes the d, pi, and S
    distances = [MAX_NUMBER] * V
    distances[source] = 0
    visited = [False] * V
    for _ in range(V):
        #Function below returns the next nearest vertice to visit
        u = min_distance(distances, visited)
        visited[u] = True #First iteration, marks source as True

        #Dijkstra's Algorithm updates distance here
        for v in range(V):
            if not visited[v] and (graph[u][v] != 0) and (distances[u] != MAX_NUMBER) and (distances[u] + graph[u][v] < distances[v]):
                distances[v] = distances[u] + graph[u][v]
    #print(f"V: {V}, Distances: {distances}, Visited = {visited}")
    return distances

def min_distance(distances, visited):
    min_dist = MAX_NUMBER
    min_index = -1
    for v in range(len(distances)):
        if distances[v] < min_dist and not visited[v]:
            min_dist = distances[v]
            min_index = v

    return min_index

def theoretical_runtime_A(num_vertices_list, scaling_factor):
    return [scaling_factor * num_vertices**2 for num_vertices in num_vertices_list]

def theoretical_runtime_B(num_vertices_list, scaling_factor):
    return [scaling_factor * (num_vertices + num_vertices**2)* math.log2(num_vertices) for num_vertices in num_vertices_list]

if __name__ == "__main__":
    num_vertices_list = [1000, 2000, 4000, 6000, 8000, 10000]
    max_edge_weight = num_vertices_list[-1]
    num_runs = 10

    NFC_list_runtimes = np.zeros(len(num_vertices_list))
    FC_list_runtimes = np.zeros(len(num_vertices_list))
    NFC_matrix_runtimes = np.zeros(len(num_vertices_list))
    FC_matrix_runtimes = np.zeros(len(num_vertices_list))
    for i, num_vertices in enumerate(num_vertices_list):
        total_NFC_list_runtime = 0
        total_FC_list_runtime = 0
        total_NFC_matrix_runtime = 0
        total_FC_matrix_runtime = 0
        for _ in range(num_runs):
            NFC_list = generate_non_fully_connected_list(num_vertices, max_edge_weight)
            NFC_matrix = adjacency_list_to_matrix(NFC_list, num_vertices)
            FC_list = generate_fully_connected_list(num_vertices, max_edge_weight)
            FC_matrix = adjacency_list_to_matrix(FC_list, num_vertices)
            
            source = random.randint(0, num_vertices - 1)

            runtime_NFC_list = timeit.timeit(lambda: dijkstra_adjacency_list(NFC_list, source), number=1)
            total_NFC_list_runtime += runtime_NFC_list

            runtime_FC_list = timeit.timeit(lambda: dijkstra_adjacency_list(FC_list, source), number=1)
            total_FC_list_runtime += runtime_FC_list

            runtime_NFC_matrix = timeit.timeit(lambda: dijkstra_adjacency_matrix(NFC_matrix, source), number=1)
            total_NFC_matrix_runtime += runtime_NFC_matrix

            runtime_FC_matrix = timeit.timeit(lambda: dijkstra_adjacency_matrix(FC_matrix, source), number=1)
            total_FC_matrix_runtime += runtime_FC_matrix

        NFC_list_runtimes[i] = total_NFC_list_runtime / num_runs
        FC_list_runtimes[i] = total_FC_list_runtime / num_runs
        NFC_matrix_runtimes[i] = total_NFC_matrix_runtime / num_runs
        FC_matrix_runtimes[i] = total_FC_matrix_runtime / num_runs

        print(f"Vertices: {num_vertices} \nAverage Runtime for NFC List: {NFC_list_runtimes[i]}, FC List: {FC_list_runtimes[i]}, NFC Matrix: {NFC_matrix_runtimes[i]}, FC Matrix: {FC_matrix_runtimes[i]}.")



    max_empirical_runtime = (max(NFC_list_runtimes) + max(FC_list_runtimes) + max(NFC_matrix_runtimes) + max(FC_matrix_runtimes))/4
    scaling_factor_A = max_empirical_runtime / (num_vertices_list[-1]**2)

    max_empirical_runtime = max(NFC_list_runtimes[-1], FC_list_runtimes[-1], NFC_matrix_runtimes[-1], FC_matrix_runtimes[-1])
    theoretical_runtime = (num_vertices_list[-1] + num_vertices_list[-1]**2) * math.log2(num_vertices_list[-1])
    scaling_factor_B = max_empirical_runtime / theoretical_runtime


    theroretical_NFC_runtimes_A = theoretical_runtime_A(num_vertices_list, scaling_factor_A)
    theroretical_NFC_runtimes_B = theoretical_runtime_B(num_vertices_list, scaling_factor_B)

    plt.plot(num_vertices_list, NFC_list_runtimes, marker="o", label="NFC List Empirical")
    plt.plot(num_vertices_list, FC_list_runtimes, marker="o", label="FC List Empirical")
    plt.plot(num_vertices_list, NFC_matrix_runtimes, marker="o", label="NFC Matrix Empirical")
    plt.plot(num_vertices_list, FC_matrix_runtimes, marker="o", label="FC Matrix Empirical")
    plt.plot(num_vertices_list, theroretical_NFC_runtimes_A, linestyle="--", label="Theoretical O(V^2)")
    plt.plot(num_vertices_list, theroretical_NFC_runtimes_B, linestyle="--", label="Theoretical O((V+E)logV)")
    plt.title("AdjMatrix/Array: Runtime vs. Number of Vertices")
    plt.xlabel("Number of Vertices")
    plt.ylabel("Average Runtime (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()

