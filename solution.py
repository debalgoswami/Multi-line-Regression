# THE METHOD TO RUN THIS CODE IS PROVIDED AT THE BOTTOM, IN THE __main__ SECTION

import pickle
import numpy as np

def calculate_regression_parameters(n, x_values, y_values, c):
    errors = np.zeros([n, n])
    x_sums = [0] * (n)
    y_sums = [0] * (n)
    x_sqr_sums = [0] * (n)
    xy_sums = [0] * (n)
    y_sqr_sums = [0] * (n)

    for j in range(n):
        for i in range(j, -1, -1):
            x_sums[i] += x_values[j]
            y_sums[i] += y_values[j]
            x_sqr_sums[i] += x_values[j] ** 2
            xy_sums[i] += x_values[j] * y_values[j]
            y_sqr_sums[i] += y_values[j] ** 2
            nn = j - i + 1

            if (((nn * x_sqr_sums[i]) - (x_sums[i] ** 2)) != 0):
                a = ((nn * xy_sums[i]) - (x_sums[i] * y_sums[i])) / ((nn * x_sqr_sums[i]) - (x_sums[i] ** 2))
            else:
                a = 1e-8

            b = (y_sums[i] - (a * x_sums[i])) / nn

            errors[i][j] = (a * a * x_sqr_sums[i] + 2 * a * b * x_sums[i] - 2 * a * xy_sums[i] + nn * b * b - 2 * b * y_sums[i] + y_sqr_sums[i])

    return a, b, errors

def calculate_segmentation_cost(pts, n, c, a, b, err):
    min_cost = [0] * (n + 1)
    ret_index = [0] * (n + 1)

    for j in range(1, n + 1):
        min_cost[j] = err[0][j - 1] + c
        ret_index[j] = 0

        for i in range(1, j + 1):
            if min_cost[i - 1] + err[i - 1][j - 1] + c < min_cost[j]:
                min_cost[j] = min_cost[i - 1] + err[i - 1][j - 1] + c
                ret_index[j] = i

    ret = []
    k_list = []
    curr_ind = n

    while curr_ind >= 1:
        next_ind = ret_index[curr_ind]
        k_list.append(curr_ind - 1)
        if next_ind == curr_ind:
            ret.append((pts[curr_ind - 1][0], pts[curr_ind - 1][1], pts[curr_ind - 1][0], pts[curr_ind - 1][1]))
        else:
            x1 = pts[next_ind - 1][0]
            y1 = x1 * a + b
            x2 = pts[curr_ind - 1][0]
            y2 = x2 * a + b
            ret.append((int(x1), int(y1), int(x2), int(y2)))
        curr_ind = next_ind - 1

    k_list.reverse()

    return min_cost[n], ret, k_list

def solve_segmentation(n_list, x_list, y_list, c_list):
    results = {
        'k_list': [],
        'last_points_list': [],
        'OPT_list': []
    }

    for n, x, y, c in zip(n_list, x_list, y_list, c_list):
        points = [(x[i], y[i]) for i in range(n)]
        points_break = []
        a, b, err = calculate_regression_parameters(n, x, y, c)
        min_cost, segments, points_break = calculate_segmentation_cost(points, n, c, a, b, err)

        k = len(segments)
        last_points = [segment[2] for segment in segments]

        results['k_list'].append(k)
        results['last_points_list'].append(points_break)
        results['OPT_list'].append(min_cost)

    return results

if __name__ == "__main__":

    # METHODS TO RUN THE CODE :
    # 1. Enter the name of the file in the pickle load command section line 100
    # 2. Make sure to enter the name of the output file in pickle dump section at the end of the CODE line 110
    # 3. Make sure the libraries used are installed. The list is mentioned above at the top
    # 4. Run the code when the input and output file has been mentioned. All the variables required by the user input 
    #    mentioned in the main function
     
    # Enter the name of the file (store in variable input_data) :
    input_data = pickle.load(open('examples_of_instances', 'rb'))

    n_list = input_data['n_list']
    x_list = input_data['x_list']
    y_list = input_data['y_list']
    c_list = input_data['C_list']

    results = solve_segmentation(n_list, x_list, y_list, c_list)  # Final calculated result list

    # For dumping the results in the pickle file, enter the name of the file
    #pickle.dump(results, open("solution_of_large_instances", "wb"))

    solution = pickle.load(open('examples_of_solutions', 'rb'))
    print(results)
    print(solution)
