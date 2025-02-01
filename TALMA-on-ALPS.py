import json
import numpy as np
import os
from datetime import datetime
from TALMA_tools import calculate_joint_vectors, calculate_angle_matrix_3part, DTW_np_adaptive_soft_regulation, filter_one_to_one_look_ahead, select_by_highest_similarity, visualize_TALMA_matching_bahavior, save_pair_pic, display_matching_results
import textwrap

def main_fun(mentor, convalescent, folder_name):        

    # Reads the mentor's json file
    with open(f'annotate/{mentor}.json', 'r') as file:
        data = json.load(file)
    # Retrieve the anchor frame's information.
    mentor_HightlightFrame = [item['frameCount'] for item in data]

    # Load video
    mentorVideo = np.load(f'3D_Model/{mentor}.npy')
    ConvalescentVideo = np.load(f'3D_Model/{convalescent}.npy')

    # Set video title
    video_title = "ConvalescentAngle"
    if convalescent == "DPConvalescent":
        video_title = "Front"
    elif convalescent == "DPConvalescentRight":
        video_title = "Right"
    elif convalescent == "DPConvalescentLeft":
        video_title = "Left"

    # Load mentor and convalescent's video vectors
    mentorVideo_vectors = calculate_joint_vectors(mentorVideo)
    ConvalescentVideo_vectors = calculate_joint_vectors(ConvalescentVideo)
    # print("mentorVideo_vectors.shape:", mentorVideo_vectors.shape)
    # print("ConvalescentVideo_vectors.shape:", ConvalescentVideo_vectors.shape)

    frame_counts_array = np.array(mentor_HightlightFrame, dtype=int)

    # Retrieve the Full, Left, Right Limb angles
    mentorVideo_vectors_AngleMatrix, mentorVideo_vectors_AngleMatrix_left, mentorVideo_vectors_AngleMatrix_right = calculate_angle_matrix_3part(mentorVideo_vectors)
    convalescentVideo_vectors_AngleMatrix, convalescentVideo_vectors_AngleMatrix_left, convalescentVideo_vectors_AngleMatrix_right = calculate_angle_matrix_3part(ConvalescentVideo_vectors)

    # Calculate the proportion of each action in the total video length
    proportions = []
    mentorVideo_length = len(mentorVideo_vectors_AngleMatrix)
    # Add a starting point (frame 0) for calculations
    frame_counts_with_start = np.insert(frame_counts_array, 0, 0) # Insert 0 to the first element of the frame_counts_array
    # Calculate proportions
    for i in range(len(frame_counts_array)):
        proportion = (frame_counts_array[i] - frame_counts_with_start[i]) / mentorVideo_length
        proportions.append(proportion)

    # ------------- Mentor Full Limb -------------
    mentor_selected_frames = mentorVideo_vectors_AngleMatrix[frame_counts_array].reshape(-1,16)
    # ------------- Mentor Left Limb -------------
    mentor_selected_frames_left = mentorVideo_vectors_AngleMatrix_left[frame_counts_array].reshape(-1,9)
    # ------------- Mentor Right Limb -------------
    mentor_selected_frames_right = mentorVideo_vectors_AngleMatrix_right[frame_counts_array].reshape(-1,9)

    # ------------- Convalescent Full Limb -------------
    convalescent_AllFrame = convalescentVideo_vectors_AngleMatrix.reshape(-1,16)
    # ------------- Convalescent Left Limb -------------
    convalescent_AllFrame_left = convalescentVideo_vectors_AngleMatrix_left.reshape(-1,9)
    # ------------- Convalescent Right Limb -------------
    convalescent_AllFrame_right = convalescentVideo_vectors_AngleMatrix_right.reshape(-1,9)

    x = convalescent_AllFrame
    y = mentor_selected_frames
    x_left = convalescent_AllFrame_left
    y_left = mentor_selected_frames_left
    x_right = convalescent_AllFrame_right
    y_right = mentor_selected_frames_right

    # Full Limb
    # ------------- (Rough-Matching Phase) Obtain one-to-multiple matching sets (Phase 1) -------------
    acc_cost_matrix, path = DTW_np_adaptive_soft_regulation(x, y, proportions)
    # ------------- (Fine-Matching Phase) Obtain one-to-one and qualified matching pair sets for Phase-1 results. (Phase 2) -------------
    filtered_path, similarity_full = filter_one_to_one_look_ahead(path, x, y, proportions)

    # Left Limb
    # ------------- (Rough-Matching Phase) Obtain one-to-multiple matching sets (Phase 2) -------------
    acc_cost_matrix_left, path_left = DTW_np_adaptive_soft_regulation(x_left, y_left, proportions)
    # ------------- (Fine-Matching Phase) Obtain one-to-one and qualified matching pair sets for Phase-1 results. (Phase 2) -------------
    filtered_path_left, similarity_left = filter_one_to_one_look_ahead(path_left, x, y, proportions)

    # Right Limb
    # ------------- (Rough-Matching Phase) Obtain one-to-multiple matching sets (Phase 3) -------------
    acc_cost_matrix_right, path_right = DTW_np_adaptive_soft_regulation(x_right, y_right, proportions)
    # ------------- (Fine-Matching Phase) Obtain one-to-one and qualified matching pair sets for Phase-1 results. (Phase 2) -------------
    filtered_path_right, similarity_right = filter_one_to_one_look_ahead(path_right, x, y, proportions)

    result = select_by_highest_similarity(
        [filtered_path[0], filtered_path_left[0], filtered_path_right[0]],
        [similarity_full, similarity_left, similarity_right]
    )

    # Use an outlined box with arrow indicators: The first line shows the mentor's anchor frame, and the line below displays the TALMA-matched convalescent result.
    display_matching_results(mentor_HightlightFrame, result, video_title)

    mentor_list = [int(item) for item in mentor_HightlightFrame]
    fig = visualize_TALMA_matching_bahavior(mentor_list, x, x_left, x_right, y, y_left, y_right, path, path_left, path_right, filtered_path, filtered_path_left, filtered_path_right, result)

    fig.savefig(f'{folder_name}/TALMA P3Matching(Front-{video_title}).png')
    
    print(f" The 'Visualize TALMA Matching Behavior' chart was successfully generated and saved as '{folder_name}/TALMA P3Matching(Front-{video_title}).png'.")
    
    # Create a folder to store the pair pictures
    os.makedirs(f"{folder_name}/Front-{video_title}", exist_ok=True)
    save_pair_pic(fr"input_video/{mentor}.mp4", mentor_HightlightFrame, fr"input_video/{convalescent}.mp4", result, f"{folder_name}/Front-{video_title}", similarity_full)
    print(f" The 'Matching Picture Pairs' were successfully generated and saved in the '{folder_name}/Front-{video_title}/' folder.")
    print("\n=======================================================================================================================\n")

if __name__ == '__main__':
    # Get the current date, and create a folder 
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    folder_name = f"fig({current_datetime})"
    os.makedirs(folder_name, exist_ok=True)

    # Terminology Reference
    line_width = 78  # Adjust the maximum text length inside the frame
    divider = "+" + "-" * line_width + "--" + "+"
    terminology_content = [
        "[Mentor anchor frames]: Anchor Frames pre-annotated in the mentor's video.",
        "[Convalescent matching result by TALMA]: Frames determined in the convalescent video using our TALMA algorithm, corresponding to the mentor's anchor frames."
    ]
    print("\n" + divider)
    print("| " + "TERMINOLOGY REFERENCE".center(line_width) + " |")
    print(divider)
    # Process each text segment and apply appropriate line wrapping
    for text in terminology_content:
        wrapped_lines = textwrap.wrap(text, width=line_width)
        for line in wrapped_lines:
            print("| " + line.ljust(line_width) + " |")
    print(divider + "\n")


    convalescent = ["DPConvalescent", "DPConvalescentRight", "DPConvalescentLeft"]
    mentor = "DPMentor"
    main_fun(f"{mentor}", f"{convalescent[0]}", folder_name)
    main_fun(f"{mentor}", f"{convalescent[1]}", folder_name)
    main_fun(f"{mentor}", f"{convalescent[2]}", folder_name)