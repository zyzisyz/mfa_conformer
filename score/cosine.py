import numpy as np

def cosine_score(trials, index_mapping, eval_vectors):
    labels = []
    scores = []
    for item in trials:
        enroll_vector = eval_vectors[index_mapping[item[1]]]
        test_vector = eval_vectors[index_mapping[item[2]]]
        score = enroll_vector.dot(test_vector.T)
        denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
        score = score/denom
        labels.append(int(item[0]))
        scores.append(score)
    return labels, scores

