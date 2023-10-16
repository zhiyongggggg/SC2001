import random
import timeit
import matplotlib.pyplot as plt
import numpy as np
import queue


MAX_NUMBER = 1000000

def generate_list(num_vertices, max_edge_weight, probability):
    edge_count = 0
    graph = [[] for _ in range(num_vertices)]  # Initialize an empty adjacency list
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            if (random.randint(0,10)/10) <= probability:
                edge_count += 1
                weight = random.randint(1, max_edge_weight)
                # Add an edge to both vertices in the adjacency list
                graph[i].append((j, weight))
                graph[j].append((i, weight))
    return graph, edge_count

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


if __name__ == "__main__":
    num_vertices = 4500
    probability_list = [0.2, 0.4, 0.6, 0.8, 1]
    max_edge_weight = 4500
    num_runs = 1
    edge_count_list = []
    list_runtimes = np.zeros(len(probability_list))
    matrix_runtimes = np.zeros(len(probability_list))

    for i, probability in enumerate(probability_list):
        total_list_runtime = 0
        total_matrix_runtime = 0
        for _ in range(num_runs):
            list, edge_count = generate_list(num_vertices, max_edge_weight, probability)
            edge_count_list.append(edge_count)
            matrix = adjacency_list_to_matrix(list, num_vertices)
            
            source = random.randint(0, num_vertices - 1)

            runtime_list = timeit.timeit(lambda: dijkstra_adjacency_list(list, source), number=1)
            total_list_runtime += runtime_list

            runtime_matrix = timeit.timeit(lambda: dijkstra_adjacency_matrix(matrix, source), number=1)
            total_matrix_runtime += runtime_matrix

        list_runtimes[i] = total_list_runtime / num_runs
        matrix_runtimes[i] = total_matrix_runtime / num_runs

        print(f"Edges: {edge_count_list[-1]} -> Average Runtime for List: {list_runtimes[i]}, Matrix: {matrix_runtimes[i]}.")


    plt.plot(edge_count_list, list_runtimes, marker="o", label="List Empirical")
    plt.plot(edge_count_list, matrix_runtimes, marker="o", label="Matrix Empirical")
    plt.title("AdjMatrix/Array: Runtime vs. Number of Edges")
    plt.xlabel("Number of Edges")
    plt.ylabel("Average Runtime (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()

