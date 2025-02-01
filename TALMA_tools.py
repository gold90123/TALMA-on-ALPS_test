import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
from collections import Counter
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.lines as mlines
import textwrap
import cv2
def calculate_joint_vectors(video):
        vectors = []
        vectors.append(video[:, 10, :] - video[:, 9, :]) #1 (0)
        vectors.append(video[:, 9, :] - video[:, 8, :]) #2 (1)
        vectors.append(video[:, 11, :] - video[:, 8, :]) #3 (2)
        vectors.append(video[:, 12, :] - video[:, 11, :]) #4 (3)
        vectors.append(video[:, 13, :] - video[:, 12, :]) #5 (4)
        vectors.append(video[:, 14, :] - video[:, 8, :]) #6 (5)
        vectors.append(video[:, 15, :] - video[:, 14, :]) #7 (6)
        vectors.append(video[:, 16, :] - video[:, 15, :]) #8 (7)
        vectors.append(video[:, 7, :] - video[:, 8, :]) #9 (8)
        vectors.append(video[:, 0, :] - video[:, 7, :]) #10 (9)
        vectors.append(video[:, 4, :] - video[:, 0, :]) #11 (10)
        vectors.append(video[:, 5, :] - video[:, 4, :]) #12 (11)
        vectors.append(video[:, 6, :] - video[:, 5, :]) #13 (12)
        vectors.append(video[:, 1, :] - video[:, 0, :]) #14 (13)
        vectors.append(video[:, 2, :] - video[:, 1, :]) #15 (14)
        vectors.append(video[:, 3, :] - video[:, 2, :]) #16 (15)

        vectors = np.stack(vectors, axis=1)
        return vectors

def calculate_angle_matrix_Full(vectors):
    angle_matrix = []
    # indices matches to the keypoints of Fig.4(c) in RAL tele-physiotherapy paper
    indices = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5),
            (5, 6), (6, 7), (8, 2), (8, 5), (8, 9),
            (9, 10), (10, 11), (11, 12), (9, 13),
            (13, 14), (14, 15)]
    for t in range(len(vectors)):
        similarity_list_for_each_frame = []
        for i, j in indices:
            similarity = cosine_similarity(vectors[t, i, :].reshape(1,-1),
                                            vectors[t, j, :].reshape(1,-1))[0,0]
            similarity_list_for_each_frame.append(similarity)
        angle_matrix.append(similarity_list_for_each_frame)
    angle_matrix = np.stack(angle_matrix)
    return angle_matrix

# calculate the angle matrix based on Full Limb, Left Limb and Right Limb
def calculate_angle_matrix_3part(vectors):
    all_angle_matrix = []
    left_angle_matrix = []
    right_angle_matrix = []
    # Full Limb
    indices = [(0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 5),
            (5, 6),
            (6, 7),
            (8, 2),
            (8, 5),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (9, 13),
            (13, 14),
            (14, 15)]
    
    # Left Limb (This is the left side body of medical definition)
    indices_left = [(0, 1),
            (1, 5),
            (5, 6),
            (6, 7),
            (8, 5),
            (8, 9),
            (9, 13),
            (13, 14),
            (14, 15)]
    
    # Right Limb (This is the right side body of medical definition)
    indices_right = [(0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (8, 2),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12)]

    for t in range(len(vectors)):
        similarity_list_for_each_frame = []
        for i, j in indices:
            similarity = cosine_similarity(vectors[t, i, :].reshape(1,-1),
                                            vectors[t, j, :].reshape(1,-1))[0,0]
            similarity_list_for_each_frame.append(similarity)
        all_angle_matrix.append(similarity_list_for_each_frame)
    all_angle_matrix = np.stack(all_angle_matrix)

    for t in range(len(vectors)):
        similarity_list_for_each_frame = []
        for i, j in indices_left:
            similarity = cosine_similarity(vectors[t, i, :].reshape(1,-1),
                                            vectors[t, j, :].reshape(1,-1))[0,0]
            similarity_list_for_each_frame.append(similarity)
        left_angle_matrix.append(similarity_list_for_each_frame)
    left_angle_matrix = np.stack(left_angle_matrix)

    for t in range(len(vectors)):
        similarity_list_for_each_frame = []
        for i, j in indices_right: # 9
            similarity = cosine_similarity(vectors[t, i, :].reshape(1,-1),
                                            vectors[t, j, :].reshape(1,-1))[0,0]
            similarity_list_for_each_frame.append(similarity)
        right_angle_matrix.append(similarity_list_for_each_frame)
    right_angle_matrix = np.stack(right_angle_matrix)

    return all_angle_matrix, left_angle_matrix, right_angle_matrix

def cosine_similarityy(vec1, vec2):
    # Calculate the cosine similarity between two vectors.
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    norm_vec1 = sqrt(sum(v ** 2 for v in vec1))
    norm_vec2 = sqrt(sum(v ** 2 for v in vec2))
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
    return cosine_sim

def DTW_np(A_h_I, A_c_I):
    # Convert input lists to numpy arrays for efficient computation
    A_h_I = np.array(A_h_I) # (601, 16)
    A_c_I = np.array(A_c_I) # (11, 16)
    print("len(A_h_I):", len(A_h_I)) # 601
    print("len(A_c_I):", len(A_c_I)) # 11
    Z = np.full((len(A_h_I) + 1, len(A_c_I) + 1), np.inf)
    Z[0, 0] = 0

    # Fill the distance matrix based on cosine similarity
    for i in range(1, len(A_h_I) + 1):
        for j in range(1, len(A_c_I) + 1):
            Z[i, j] = 1 - cosine_similarityy(A_h_I[i-1], A_c_I[j-1])

    # Compute the accumulated cost matrix
    for i in range(1, len(A_h_I) + 1):
        for j in range(1, len(A_c_I) + 1):
            Z[i, j] += np.min([Z[i-1, j-1], Z[i-1, j], Z[i, j-1]])

    D = [row[1:] for row in Z[1:]] # delete the first row and first column

    # Backtrack to find the optimal path
    i = len(A_h_I)-1
    j = len(A_c_I)-1
    p, q = [i], [j]
    while i > 0 or j > 0:
        # Ensure that boundary conditions are handled properly to prevent out-of-bounds errors.
        if i > 0 and j > 0:
            tb = np.argmin((D[i-1][j-1], D[i][j-1], D[i-1][j]))
            if tb == 0:
                i -= 1
                j -= 1
            elif tb == 1:
                j -= 1
            else:  # tb == 2
                i -= 1
        elif i > 0: # Only allowed to move upward
            i -= 1
        elif j > 0: # Only allowed to move left
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return Z[1:, 1:], (p, q) # Remove the first row and column used for initialization

def DTW_np_adaptive_soft_regulation(A_h_I, A_c_I, proportions):
    # Calculate the interval between each action in the convalescent's video based on the ratio
    intervals = []
    for proportion in proportions:
        interval = proportion * len(A_h_I)
        intervals.append(interval)

    # Calculate mean and standard deviation of intervals
    mean_intervals = np.mean(intervals)
    std_intervals = np.std(intervals)
    k = std_intervals / mean_intervals

    # Calculate the frame count for each action in the convalescent video based on the ratio
    estimate_anchor_frames = []
    current_sum = 0
    for interval in intervals:
        current_sum += interval
        estimate_anchor_frames.append(current_sum)

    # Calculate the adaptive sigma for each action based on its adaptive interval
    sigmas = [interval * k for interval in intervals]

    # Penalty function: adjust the penalty based on the offset
    def penalty_function(offset, anchor_num):
        if abs(offset) <= sigmas[anchor_num]:
            return 1 # No penalty within the normal range
        else:
            # Beyond the range, the penalty increases quadratically with the offset
            return 1 + ((abs(offset) - sigmas[anchor_num]) / sigmas[anchor_num]) ** 2

    # Convert input lists to numpy arrays
    A_h_I = np.array(A_h_I)  # Client frames
    A_c_I = np.array(A_c_I)  # Highlight points
    len_h, len_c = len(A_h_I), len(A_c_I)

    # Initialize accumulated cost matrix
    Z = np.full((len_h + 1, len_c + 1), np.inf)
    Z[0, 0] = 0

    # Compute distance matrix (Count in ascending order)
    # This matrix's rows represent convalescent frames, and columns represent mentor anchors, recording the distance from each convalescent frame to each mentor anchor
    for i in range(1, len_h + 1):
        for j in range(1, len_c + 1):
            base_cost = 1 - cosine_similarityy(A_h_I[i - 1], A_c_I[j - 1]) # Purely calculate action similarity
            
            # Calculate offset using the estimated anchor frames
            offset = estimate_anchor_frames[j-1] - i
            Z[i, j] = base_cost * penalty_function(offset, j-1) # Introduce the concept of time penalty

    # Compute accumulated cost matrix
    for i in range(1, len_h + 1):
        for j in range(1, len_c + 1):
            Z[i, j] += np.min([Z[i - 1, j - 1], Z[i - 1, j], Z[i, j - 1]])

    # Remove first row and column for backtracking
    D = [row[1:] for row in Z[1:]]

    # Backtracking with soft regulation
    i, j = len_h - 1, len_c - 1
    p, q = [i], [j]

    # Proceed only if one side is greater than 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            tb = np.argmin([D[i-1][j-1], np.inf, D[i - 1][j]]) # Horizontal movement is prohibited, as it implies one client frame is paired with multiple anchors
            if tb == 0: # Diagonal movement indicates moving to the next anchor pairing
                i -= 1
                j -= 1
            elif tb == 2: # Vertical movement indicates one anchor is paired with different client frames
                i -= 1
        elif i > 0: # Indicates that all anchors are paired, and the remaining client frames are paired with the first anchor
            i -= 1
        p.insert(0, i)
        q.insert(0, j)
    return Z[1:, 1:], (p, q)

def filter_one_to_one(path, A_h_I, A_c_I):
    path_p, path_q = path
    # The lengths of path_p and path_q are the same; one represents the number of Patient frames, and the other represents the anchor each frame is classified into. 
    # anchor can be one-to-many (one anchor paired with multiple Patient frames).
    unique_path_p = []
    unique_path_q = []
    similarity_matrix = []

    # stores the similarity
    similarity_dict = {}

    for i, j in zip(path_p, path_q):
        similarity = cosine_similarityy(A_h_I[i], A_c_I[j])
        # Focus on j, and determine which i has the highest similarity to j if j is paired with multiple i, and filter out all other i.
        if j not in similarity_dict or similarity > similarity_dict[j][1]:
            similarity_dict[j] = (i, similarity)

    # Organize the matched i, j pairs and return the result.
    for j in similarity_dict:
        i, sim = similarity_dict[j]
        unique_path_p.append(i)
        unique_path_q.append(j)
        similarity_matrix.append(sim)
    return (unique_path_p, unique_path_q), similarity_matrix


def calculate_avg_similarity(A_h_I, A_c_I):
    similarity_avg = 0
    total_similarity = 0
    for patient_frame in A_h_I:
        # 更新平均相似度
        total_similarity += cosine_similarityy(patient_frame, A_c_I)
    similarity_avg = total_similarity / len(A_h_I)
    return similarity_avg

# Calculate the average similarity of the data within the top_percent range.
def calculate_segmented_similarity_avg(A_h_I, A_c_I, top_percent=0.15):
    similarity_scores = []

    # Calculate all similarities
    for patient_frame in A_h_I:
        similarity_scores.append(cosine_similarityy(patient_frame, A_c_I))

    # Sort the similarities and take the top percentage of data
    similarity_scores.sort(reverse=True) # Sort in descending order
    std_dev = np.std(similarity_scores)
    top_n = int(len(similarity_scores) * top_percent)
    if top_n == 0: # If top_n is 0, it means there are not enough data points to select the top_percent values
        top_similarity_scores = [similarity_scores[-1]] # The smallest similarity in the selected range is returned to narrow down the similarity threshold and prioritize the closest match. This prevents a situation where there are remaining anchors to match, but not enough frames left for pairing.
    else:
        top_similarity_scores = similarity_scores[:top_n]  

    return np.mean(top_similarity_scores)

def filter_one_to_one_look_ahead(path, A_h_I, A_c_I, proportions, similarity_threshold=0.75, top_percent=0.15):
    # Calculate the interval between each action in the convalescent's video based on the ratio
    intervals = []
    for proportion in proportions:
        interval = proportion * len(A_h_I)
        intervals.append(interval)
    # Use the standard deviation as the minimum interval
    min_frames = int(np.std(intervals))

    path_p, path_q = path
    unique_path_p = []
    unique_path_q = []
    similarity_matrix = []

    # Store the best i and its similarity for each j
    similarity_dict = {}

   # Calculate the occurrence count of each value
    count_dict = Counter(path_q)

    # First pass: filter one-to-many matches and find the maximum value (Step 2.1 in the pseudo-code of the RAL tele-physiotherapy paper)
    for i, j in zip(path_p, path_q):
        similarity = cosine_similarityy(A_h_I[i], A_c_I[j])
        if j == 0:
            if i == 0:
                similarity_dict[j] = (i, similarity)
            else:
                if similarity > similarity_dict[j][1]:
                    similarity_dict[j] = (i, similarity)
        else:
            # Check if j is not in similarity_dict
            if j not in similarity_dict:
                # If it's the first frame or the interval from the previous action exceeds min_frames, add it directly
                if count_dict[j] < min_frames:
                    # If the DTW_np_adaptive_soft_regulation classification(Step 1) is insufficient, add it directly at the beginning
                    similarity_dict[j] = (i, similarity)
                else:
                    if i > similarity_dict[j-1][0]+min_frames:
                        similarity_dict[j] = (i, similarity)
                        continue
            # Record only the pair with the highest similarity (add a restriction that the interval from the previous selected action must be > min_frames)
            elif similarity > similarity_dict[j][1]:
                similarity_dict[j] = (i, similarity) # Record the matched i frame and its similarity

    # Second pass (Step 2.2): filter by checking if the similarity in similarity_dict meets the threshold and handle the Re-match
    ReMatch_anchors = [] # Record the anchors that needs to be re-matched
    for index, j in enumerate(sorted(similarity_dict)): # Sort the dictionary keys in ascending order
        i, sim = similarity_dict[j]

        # If it's the last loop, handle it specially since there won't be another loop to process it.
        if index == len(similarity_dict)-1:
            # If there are unprocessed re-match anchors, fix them starting from the last
            for previous_j in ReMatch_anchors:
                # In the re-match region, find the first frame with similarity greater than the similarity_pr as the second best solution (the first best solution is when we find it in Step 1).
                # -min_frames is used to avoid pairing this anchor with the same action as the previous anchor
                start_index = unique_path_p[-1]+min_frames if unique_path_p else 0
                end_index = i-min_frames
                # similarity_avg = calculate_avg_similarity(A_h_I[start_index:end_index], A_c_I[previous_j])
                # The method of using similarity_pr as the threshold is better than similarity_svg.
                similarity_pr = calculate_segmented_similarity_avg(A_h_I[start_index:end_index], A_c_I[previous_j], top_percent)
                for backtrack_i in range(unique_path_p[-1]+min_frames if unique_path_p else 0, i-min_frames):
                    backtrack_sim = cosine_similarityy(A_h_I[backtrack_i], A_c_I[previous_j])
                    # Find the first frame with similarity greater than similarity_pr as the cross-region match result
                    if backtrack_sim >= similarity_pr:
                        unique_path_p.append(backtrack_i)
                        unique_path_q.append(previous_j)
                        similarity_matrix.append(backtrack_sim)
                        break   
            ReMatch_anchors.clear() # Clear all Re-match anchors for this round since they are processed
            # For the last j, append it directly without review
            unique_path_p.append(i)
            unique_path_q.append(j)
            similarity_matrix.append(sim)
            continue

        # If the similarity for this anchor does not meet the threshold, do not add this point yet, move to review the next point
        if sim < similarity_threshold:
            ReMatch_anchors.append(j) # Record anchors with similarity below the threshold
            continue # Skip this match and look for the pairing point for the next anchor

        # If the match point of the next anchor is found, check if there are failed matches (anchors that required re-match) recorded before this point
        for previous_j in ReMatch_anchors:
            # In the re-match region, find the first frame with similarity greater than the similarity_pr as the second best solution (the first best solution is when we find it in Step 1).
            # -min_frames is used to avoid pairing this anchor with the same action as the previous anchor
            start_index = unique_path_p[-1] + min_frames if unique_path_p else 0
            end_index = i-min_frames
            # similarity_avg = calculate_avg_similarity(A_h_I[start_index:end_index], A_c_I[previous_j])
            # The method of using similarity_pr as the threshold is better than similarity_svg.
            similarity_pr = calculate_segmented_similarity_avg(A_h_I[start_index:end_index], A_c_I[previous_j], top_percent)
            for backtrack_i in range(unique_path_p[-1]+min_frames if unique_path_p else 0, i-min_frames): # 第三個值 -1 代表每次向前移動一個索引，因為他是從後往前找（從遺失的 action 的下一個 action 找到上一個 action）
                backtrack_sim = cosine_similarityy(A_h_I[backtrack_i], A_c_I[previous_j])
                if backtrack_sim >= similarity_pr:
                    unique_path_p.append(backtrack_i)
                    unique_path_q.append(previous_j)
                    similarity_matrix.append(backtrack_sim)
                    break
        ReMatch_anchors.clear() # Clear all Re-match anchors for this round since they are processed

        # Properly record the matches for the current region
        unique_path_p.append(i)
        unique_path_q.append(j)
        similarity_matrix.append(sim)

    return (unique_path_p, unique_path_q), similarity_matrix

# From the given lists, select the value with the highest similarity at each corresponding position in the lists
def select_by_highest_similarity(lists, similarities):
    results = [] # Store the final selected frame results
    num_items = len(lists[0]) # All lists have the same length

    for i in range(num_items):
        # Collect candidate values for the current frame and their corresponding similarities
        candidates = [lists[j][i] for j in range(len(lists))]
        candidate_similarities = [similarities[j][i] for j in range(len(similarities))]
        
        # Find the candidate with the highest similarity
        max_similarity_index = candidate_similarities.index(max(candidate_similarities))
        selected_frame = candidates[max_similarity_index]
        
        # Ensure the selected frame number is greater than the previous result (to avoid the issue where the frame number of a later selected match is smaller than that of an earlier match)
        if not results or selected_frame > results[-1]:
            results.append(selected_frame)
        else:
            # Create an index list, sorted in descending order of similarity
            sorted_indices = sorted(range(len(candidate_similarities)), 
                                    key=lambda k: candidate_similarities[k], 
                                    reverse=True) # Sort in descending order of similarity
            for index in sorted_indices[1:]: # Iterate through the index list and break out of the loop early if the condition is met (the first iteration is guaranteed to fail, so it doesn't need to go through the check)
                selected_frame = candidates[index]
                if not results or selected_frame > results[-1]:
                    results.append(selected_frame)
                    break # Condition met, exit the loop
    return results

# custom class for visualizing TALMA matching behavior
class CustomConvalescentHandler(HandlerLine2D):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        line = super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)
        for artist in line:
            artist.set_markersize(24) # Modify the marker size (original size for Convalescent was 16 -> now 24; Mentor was already 24, so no changes)
        return line
        
# Final result plot after select_by_highest_similarity, including visualization of results from Full Limb, Left Limb, Right Limb processes (vector length)
def visualize_TALMA_matching_bahavior(mentor_list, x, x_left, x_right, y, y_left, y_right, path, path_left, path_right, 
                      filtered_path, filtered_path_left, filtered_path_right, result):
    
    x_values_length = [np.linalg.norm(x[i]) for i in path[0]] # Patient frame
    x_values_left_length = [np.linalg.norm(x_left[i]) for i in path_left[0]]
    x_values_right_length = [np.linalg.norm(x_right[i]) for i in path_right[0]]
    indices_y_mapped = [mentor_list[x] for x in path[1]]
    indices_y_mapped_left = [mentor_list[x] for x in path_left[1]]
    indices_y_mapped_right = [mentor_list[x] for x in path_right[1]]
    y_values_length = [np.linalg.norm(y[j]) for j in path[1]]  # Mentor highlight frame
    y_values_left_length = [np.linalg.norm(y_left[j]) for j in path_left[1]]
    y_values_right_length = [np.linalg.norm(y_right[j]) for j in path_right[1]]

    y_pairs_length = list(set(zip(indices_y_mapped, y_values_length)))
    unique_y_pairs_length = sorted(y_pairs_length, key=lambda x: x[0])
    y_pairs_left_length = list(set(zip(indices_y_mapped_left, y_values_left_length)))
    unique_y_pairs_left_length = sorted(y_pairs_left_length, key=lambda x: x[0])
    y_pairs_right_length = list(set(zip(indices_y_mapped_right, y_values_right_length)))
    unique_y_pairs_right_length = sorted(y_pairs_right_length, key=lambda x: x[0])

    plt.figure(figsize=(17, 10))
    # Adjust the size of numbers on the x-axis and y-axis
    plt.tick_params(axis='both', which='major', labelsize=35)
    # Format y-axis labels to one decimal place
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
    # Draw Convalescent first, then Mentor, since Mentor is the main focus and should be displayed on top
    # Plot Convalescent data
    plt.plot(path[0], x_values_length, 'bo-', markerfacecolor='blue', markeredgewidth=1, label='Convalescent (Full)', alpha=0.4, markersize=16)
    plt.plot(path_left[0], x_values_left_length, 'c^-', markerfacecolor='cyan', markeredgewidth=1, label='Convalescent (Left)', alpha=0.4, markersize=16)
    plt.plot(path_right[0], x_values_right_length, marker='s', linestyle='-', color='#FFBF00', markerfacecolor='#FFD700', markeredgewidth=1, label='Convalescent (Right)', alpha=0.4, markersize=16)
    # Plot Mentor data
    plt.plot(indices_y_mapped, y_values_length, 'ro', label='Mentor (Full)', alpha=0.7, markersize=24)
    plt.plot(indices_y_mapped_left, y_values_left_length, 'r^', label='Mentor (Left)', alpha=0.7, markersize=24)
    plt.plot(indices_y_mapped_right, y_values_right_length, 'rs', label='Mentor (Right)', alpha=0.7, markersize=24)
    
    # Get the current y-axis range
    y_min, y_max = plt.gca().get_ylim()
    plt.ylim(1.2, 3.3) # Fix the y-axis to ensure all three plots have consistent y-axis scaling

    # Label \alpha_i at the bottom of the chart
    # Find the 11 unique Mentor data points
    alpha_positions = sorted(set(indices_y_mapped))
    # Annotate \alpha_i below the Mentor data points
    for i, x_pos in enumerate(alpha_positions):
        plt.text(x_pos, 1.45, rf'$\mathbf{{\alpha_{{{i+1}}}}}$', ha='center', va='top', fontsize=35, color='red')
        # Add arrows with starting points slightly below the text annotations
        plt.annotate('', xy=(x_pos, 1.2), xytext=(x_pos, 1.33), 
                    arrowprops=dict(color='red', arrowstyle='-|>, head_width=0.8, head_length=0.8', lw=9))

    # Label \beta_i at the bottom of the chart 
    # Find the 11 unique values from the Patient data
    unique_beta_positions = result
    # Annotate bold \beta_i values at the bottom of the chart
    for i, x_pos in enumerate(unique_beta_positions):
        plt.text(x_pos, 1.55, rf'$\mathbf{{\beta_{{{i+1}}}}}$', ha='center', va='top', fontsize=35, color='green')
        # Add arrows with starting points slightly below the text annotations
        plt.annotate('', xy=(x_pos, 1.2), xytext=(x_pos, 1.33),
                    arrowprops=dict(color='green', arrowstyle='-|>, head_width=0.8, head_length=0.8', lw=9))

    # Handle dashed lines and text annotations
    text_offset = 0.9  # Initial height multiplier for text, indicating the relative position compared to the maximum y-value
    text_decrement = 0.03  # Height multiplier decrement after each annotation
    displayed_texts = set() # Set of displayed text annotations (to avoid overlaps in limited space)

    # Find the rightmost mentor_frame_index
    max_mentor_frame_index = max(mentor_list)
    # Handle dashed lines and text annotations
    for frame in result:
        if frame not in filtered_path[0]:
            # Locate the corresponding Mentor frame index
            mentor_frame_index = mentor_list[result.index(frame)]
            
            # Draw a black dashed line
            plt.axvline(x=mentor_frame_index, color='black', linestyle='--', linewidth=5, alpha=0.7)

            # Determine the text content for annotation
            text = ""
            if frame in filtered_path_left[0] and frame in filtered_path_right[0]:
                text = "Matching is determined by ALPS Left & Right"
            elif frame in filtered_path_left[0]:
                text = "Matching is determined by ALPS Left"
            elif frame in filtered_path_right[0]:
                text = "Matching is determined by ALPS Right"

            # Check if the text has already been displayed
            wrapped_text = textwrap.fill(text, width=12) # Automatically wrap text if it exceeds the line width limit
            if text not in displayed_texts:
                y_position = y_max * text_offset
                plt.text(max_mentor_frame_index-5, y_position, wrapped_text,
                        rotation=0, verticalalignment='bottom', horizontalalignment='center', fontsize=36, color='black')
                displayed_texts.add(text)
                text_offset -= text_decrement

    # Handle green lines
    for frame in result:
        if frame in filtered_path[0]:
            # Full Limb
            i = frame
            unique_index_all = filtered_path[0].index(i)
            unique_mentor_x, unique_mentor_y = unique_y_pairs_length[unique_index_all] 
            plt.plot([i, unique_mentor_x], [x_values_length[path[0].index(i)], unique_mentor_y],
                    color='green', alpha=1, linewidth=6)
        if frame in filtered_path_left[0]:
            # Left Limb
            i = frame
            unique_index_left = filtered_path_left[0].index(i)
            unique_mentor_x, unique_mentor_y = unique_y_pairs_left_length[unique_index_left]
            plt.plot([i, unique_mentor_x], [x_values_left_length[path_left[0].index(i)], unique_mentor_y],
                    color='green', alpha=1, linewidth=6)
        if frame in filtered_path_right[0]:
            # Right Limb
            i = frame
            unique_index_right = filtered_path_right[0].index(i)
            unique_mentor_x, unique_mentor_y = unique_y_pairs_right_length[unique_index_right]
            plt.plot([i, unique_mentor_x], [x_values_right_length[path_right[0].index(i)], unique_mentor_y],
                    color='green', alpha=1, linewidth=6)

    # Draw the legend with two rows: the first row for Mentor, the second for Convalescent
    handles, labels = plt.gca().get_legend_handles_labels() # Get the current legend handles and labels
    # Manually adjust the legend order: display Mentor first, then Convalescent
    mentor_labels = ['Mentor (Full)', 'Convalescent (Full)', 'Mentor (Right)']
    convalescent_labels = ['Convalescent (Right)', 'Mentor (Left)', 'Convalescent (Left)']
    sorted_labels = mentor_labels + convalescent_labels
    sorted_handles = [handles[labels.index(lbl)] for lbl in sorted_labels]
    # Add hidden items (transparent placeholders) to enforce the layout
    hidden_handle = mlines.Line2D([], [], color='none', label='')
    while len(sorted_handles) % 3 != 0:
        sorted_handles.append(hidden_handle)
        sorted_labels.append('')
    legend = plt.legend(
        sorted_handles, 
        sorted_labels,
        loc='upper left',
        fontsize=21.3,
        ncol=3, # Enforce 3 columns (since there are 6 labels, this results in 2 rows)
        bbox_to_anchor=(-0.005, 1), # Adjust the legend position slightly to the left
        handler_map={plt.Line2D: CustomConvalescentHandler()} # Customize the handler to enlarge the Convalescent marker
    )
    for text in legend.get_texts():
        text.set_fontweight('bold') # Set text to bold

    # Add a title and axis labels
    # plt.title(f'TALMA P3Matching (Front-{video_title})', fontsize=16)
    plt.xlabel('Frame Index', fontsize=50)
    plt.ylabel('Vector Length of ALPS', fontsize=50)
    # Adjust the layout
    plt.tight_layout(pad=2.0)
    # Modify the border line width
    ax = plt.gca() # Get the current axis
    ax.spines['top'].set_linewidth(3) # Top border line
    ax.spines['right'].set_linewidth(3) # Right border line
    ax.spines['left'].set_linewidth(3) # Left border line
    ax.spines['bottom'].set_linewidth(3) # Bottom border line

    return plt

def getframe(vidname, frame_index):
    frame_index += 1
    cap = cv2.VideoCapture(vidname)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")

    # Set the frame position to read
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1) # frame_index is 1-based

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Cannot read frame {frame_index}.")
    return frame

def save_pair_pic(mentor_video, mentor_HightlightFrame, convalescent_video, result, pair_pic_folder_name, similarity_full):
    for i,content in enumerate(result):
        mentor_hlt_frame = getframe(mentor_video, int(mentor_HightlightFrame[i]))
        mentor_hlt_frame = cv2.resize(mentor_hlt_frame[650:mentor_hlt_frame.shape[0]-150,0:mentor_hlt_frame.shape[1]], (1120, 1080))
        convalescent_hlt_frame = getframe(convalescent_video, result[i])
        convalescent_hlt_frame = cv2.resize(convalescent_hlt_frame[400:convalescent_hlt_frame.shape[0]-150,0:convalescent_hlt_frame.shape[1]-100], (1120, 1080))
        conbine_img = np.vstack((mentor_hlt_frame,convalescent_hlt_frame))
        if similarity_full[i] >= 0.7: # threshold
            cv2.putText(conbine_img, f'{round(similarity_full[i],3)}', (10, 1200), cv2.FONT_HERSHEY_SIMPLEX, 5, (34, 139, 34), 13)
        else:
            cv2.putText(conbine_img, f'{round(similarity_full[i],3)}', (10, 1200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 13)
  
        cv2.imwrite(f"{pair_pic_folder_name}/{i+1}.png", conbine_img)

# Display matching results between mentor anchors and TALMA-calculated convalescent frames with a simple terminal GUI using arrows and borders.
def display_matching_results(mentor_HightlightFrame, result, video_title):

    # Define frame width and spacing parameters
    line_width = 100
    number_spacing = 4  # Adjusts the spacing between each number
    arrow_spacing = number_spacing  # Adjusts arrow alignment

    # Helper function to create formatted lines with spacing
    def format_numbers(numbers):
        return " ".join(f"{num:>{number_spacing}}" for num in numbers)

    # Title box
    title = f"Mentor and Convalescent Matching Results by TALMA (Camera's Position: Front-{video_title})"
    print("+" + "-" * (line_width - 2) + "+")
    print(f"| {title.center(line_width - 4)} |")
    print("+" + "-" * (line_width - 2) + "+")

    # Frame data display
    mentor_line = "Mentor anchor frames:                  " + format_numbers(mentor_HightlightFrame)
    convalescent_line = "Convalescent matching result by TALMA: " + format_numbers(result)
    arrow_line = " " * 40 + " ".join("↓".center(arrow_spacing) for _ in mentor_HightlightFrame)

    # Print the lines within the frame
    print(f"| {mentor_line.ljust(line_width - 4)} |")
    print(f"| {arrow_line.ljust(line_width - 4)} |")
    print(f"| {convalescent_line.ljust(line_width - 4)} |")

    # Close the frame
    print("+" + "-" * (line_width - 2) + "+\n")

if __name__ == '__main__':
    print("This is the toolbox for TALMA-on-ALPS, inveneted by NCKU CIoT Lab.")