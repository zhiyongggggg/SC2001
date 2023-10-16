import random
import timeit
import matplotlib.pyplot as plt
import numpy as np

MAX_NUMBER = 1000000

def generate_fully_connected_graph(num_vertices, max_edge_weight):
    graph = [[0] * num_vertices for i in range(num_vertices)]
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            weight = random.randint(1, max_edge_weight)
            graph[i][j] = weight
            graph[j][i] = weight

    return graph

def generate_non_fully_connected_graph(num_vertices, max_edge_weight):
    graph = [[0] * num_vertices for i in range(num_vertices)]
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            if random.randint(0,1 )==1:
                weight = random.randint(1, max_edge_weight)
                graph[i][j] = weight
                graph[j][i] = weight

    return graph

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

def measure_NFC_runtime(num_vertices_list, max_edge_weight, num_runs):
    NFC_runtimes = np.zeros(len(num_vertices_list))  # Initialize a numpy array for NFC_runtimes
    for i, num_vertices in enumerate(num_vertices_list): #i is the enumerated number; eg.1,2,3... 
        total_runtime = 0                                   #num_vertices is the value of index at 1,2,3...
        for _ in range(num_runs):
            graph = generate_non_fully_connected_graph(num_vertices, max_edge_weight)
            source = random.randint(0, num_vertices - 1)
            runtime = timeit.timeit(lambda: dijkstra_adjacency_matrix(graph, source), number=1) #Lambda allows us to pass a function instead of the results of the function
            total_runtime += runtime
        avg_runtime = total_runtime / num_runs
        NFC_runtimes[i] = avg_runtime  # Store the average runtime in the numpy array
        print(f"Vertices: {num_vertices}, Average Runtime: {avg_runtime} seconds")

    return NFC_runtimes

def measure_FC_runtime(num_vertices_list, max_edge_weight, num_runs):
    FC_runtimes = np.zeros(len(num_vertices_list))  # Initialize a numpy array for NFC_runtimes
    for i, num_vertices in enumerate(num_vertices_list): #i is the enumerated number; eg.1,2,3... 
        total_runtime = 0                                   #num_vertices is the value of index at 1,2,3...
        for _ in range(num_runs):
            graph = generate_fully_connected_graph(num_vertices, max_edge_weight)
            source = random.randint(0, num_vertices - 1)
            runtime = timeit.timeit(lambda: dijkstra_adjacency_matrix(graph, source), number=1) #Lambda allows us to pass a function instead of the results of the function
            total_runtime += runtime
        avg_runtime = total_runtime / num_runs
        FC_runtimes[i] = avg_runtime  # Store the average runtime in the numpy array
        print(f"Vertices: {num_vertices}, Average Runtime: {avg_runtime} seconds")

    return FC_runtimes

def theroretical_runtime(num_vertices_list, scaling_factor):
    return [scaling_factor * num_vertices**2 for num_vertices in num_vertices_list]

if __name__ == "__main__":
    num_vertices_list = [10, 50, 100, 200, 300, 400, 500]
    max_edge_weight = 100
    num_runs = 10

    empirical_NFC_runtimes = measure_NFC_runtime(num_vertices_list, max_edge_weight, num_runs)
    empirical_FC_runtimes = measure_FC_runtime(num_vertices_list, max_edge_weight, num_runs)

    max_empirical_runtime = (max(empirical_NFC_runtimes) + max(empirical_FC_runtimes))/2
    scaling_factor = max_empirical_runtime / (num_vertices_list[-1]**2)

    theroretical_NFC_runtimes = theroretical_runtime(num_vertices_list, scaling_factor)

    plt.plot(num_vertices_list, empirical_NFC_runtimes, marker="o", label="NFC Empirical Runtime")
    plt.plot(num_vertices_list, empirical_FC_runtimes, marker="o", label="FC Empirical Runtime")
    plt.plot(num_vertices_list, theroretical_NFC_runtimes, linestyle="--", label="Theoretical O(V^2)")
    plt.title("AdjMatrix/Array: Runtime vs. Number of Vertices")
    plt.xlabel("Number of Vertices")
    plt.ylabel("Average Runtime (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()

