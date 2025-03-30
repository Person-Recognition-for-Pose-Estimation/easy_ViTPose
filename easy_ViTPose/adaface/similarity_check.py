import torch

def find_similar_tests(similarity_matrix, threshold=0.5, num_examples=2):
    
    similar_cases = []
    
    # Number of total rows
    total_rows = similarity_matrix.shape[0]
    
    # For each test case
    for test_idx in range(num_examples, total_rows):
        # Look at similarities with source cases
        for example_idx in range(num_examples):
            similarity = similarity_matrix[test_idx, example_idx].item()
            if similarity > threshold:
                similar_cases.append((test_idx, example_idx, similarity))
    
    return similar_cases
