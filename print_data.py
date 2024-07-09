import pickle
import torch
import pandas as pd

# Load the pickle file
file_path = "test_results.pkl"

with open(file_path, "rb") as file:
    data = pickle.load(file)

# Extracting data
conditions = data[0][0]
bounding_boxes = data[1][0]
numeric_attributes_1 = data[2][0]
numeric_attributes_2 = data[3][0]

# Example usage of the data
print("Conditions Tensor:", conditions)
print("Bounding Boxes Tensor:", bounding_boxes)
print("Numeric Attributes Tensor 1:", numeric_attributes_1)
print("Numeric Attributes Tensor 2:", numeric_attributes_2)
print(
    len(conditions),
    len(bounding_boxes),
    len(numeric_attributes_1),
    len(numeric_attributes_2),
)


# blue male, red female, orange asian, black black, brown indian, green white hispanic middle eastern latino
# one category is gender
# one category is race
# tensor 2 is the race becasue there are 4 categories
# tensor 1 must be the gender
def print_gender(tensor):
    male_count = 0
    female_count = 0
    for idx, (female_score, male_score) in enumerate(tensor):
        gender = ""
        if female_score > male_score:
            gender = "Female"
            male_count += 1
        else:
            gender = "Male"
            female_count += 1
        print(f"Image {idx}: {gender}")
    print(f"M: {male_count} | F: {female_count}")
    ratio = float("inf")
    if female_count > 0:
        ratio = male_count / female_count
    print(f"Ratio: {ratio}")


def print_race(tensor):
    score_to_race = {0: "White", 1: "Black", 2: "Indian", 3: "Asian"}
    for idx, scores in enumerate(tensor):
        max_index = torch.argmax(scores).item()
        print(f"Image {idx}: {score_to_race[max_index]}")


print_gender(numeric_attributes_1)
# print_race(numeric_attributes_2)


def generate_finetune_csv(file_path, output_csv):
    # Load the pickle file
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    # Extracting data
    conditions = data[0][0]
    bounding_boxes = data[1][0]
    numeric_attributes_1 = data[2][0]
    numeric_attributes_2 = data[3][0]

    # Gender and race mappings
    gender_scores_to_label = {0: "Female", 1: "Male"}
    race_scores_to_label = {0: "White", 1: "Black", 2: "Indian", 3: "Asian"}

    # Create a list to store data
    data_list = []

    # Iterate through the data to create the list of dictionaries
    for idx in range(len(numeric_attributes_1)):
        # Determine gender
        female_score, male_score = numeric_attributes_1[idx]
        gender = "Female" if female_score > male_score else "Male"

        # Determine race
        race_scores = numeric_attributes_2[idx]
        max_index = torch.argmax(race_scores).item()
        race = race_scores_to_label[max_index]

        # Image file name
        image_file = f"/n/home07/iislasluz/representational-generation/prompt_0/img_{idx}.jpg"  # Assuming image files are named sequentially

        # Append the data as a dictionary
        data_list.append({"file": image_file, "gender": gender, "race": race})

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data_list)

    # Save DataFrame to CSV
    df.to_csv(output_csv, index=False)


generate_finetune_csv("test_results.pkl", "finetune_label_val.csv")


# /n/holylabs/LABS/calmon_lab/Lab/datasets/fairface
# clip  debiasclip  fairface_label_train.csv  fairface_label_val.csv  train  val
