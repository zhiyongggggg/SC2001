import random
import time
import pandas as pd

THRESHOLD = 4

def generate_random_dataset(size, max_value):
    return [random.randint(1, max_value) for i in range(size)]

def insertionSort(arr):
    keyCompare = 0
    for i in range(1, len(arr)):
        j = i
        while j > 0:
            keyCompare += 1
            if (arr[j] < arr[j-1]):
                temp = arr[j]
                arr[j] = arr[j-1]
                arr[j-1] = temp
                j -= 1
    return keyCompare        

def mergeSort(arr):
    if len(arr) > THRESHOLD:
        if len(arr) > 1: #Not really needed, but acts as a safety net in case THRESHOLD is set to 0

            #Splitting the array into left and right
            left_array = arr[0:len(arr)//2]
            right_array = arr[len(arr)//2:]

            #Recursively call mergeSort on left and right array
            comparisonLeft = mergeSort(left_array)
            comparisonRight = mergeSort(right_array)
            keyCompare = 0
            #Merging the two subarrays
            i = 0 #Trackers for each array
            j = 0
            k = 0
            while i < len(left_array) and j < len(right_array):
                keyCompare += 1
                if left_array[i] < right_array[j]:
                    arr[k] = left_array[i]
                    i += 1
                    k += 1
                else: 
                    arr[k] = right_array[j]
                    j += 1
                    k += 1

            #In the event that 1 subarray is empty and the other is not
            while i < len(left_array):
                arr[k] = left_array[i]
                k += 1
                i += 1
            while j < len(right_array):
                arr[k] = right_array[j]
                k += 1
                j += 1

    else: #Size of array smaller than threshold, call insertion sort instead
        return insertionSort(arr)
    return comparisonLeft + comparisonRight + keyCompare

#===========================MANUAL TESTING=========================================
""" #Randomly generate array
randomly_generated_array = generate_random_dataset(50000, 100000) #Change argument here to vary (size,max value)

#Record the start time
start_time = time.time()

#Start sorting
sumKeyCompare = 0
print(mergeSort([randomly_generated_array]))

#Record the end time
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time Taken: {elapsed_time:.6f} seconds") """
#==================================================================================
#Part (i)
def main():
    #Analysis
    results = []
    start_time = time.time()
    for array_size in range(1000000, 10000001, 1000000):
        print("ONE")
        randomly_generated_array = generate_random_dataset(array_size, array_size*2)
        results.append((array_size, mergeSort(randomly_generated_array)))
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    df = pd.DataFrame(results, columns=["Array Size", "Key Comparisons"])
    df.to_excel("sorting_results.xlsx", index=False)
    print(f"Time Taken: {elapsed_time:.6f} seconds")


main()
