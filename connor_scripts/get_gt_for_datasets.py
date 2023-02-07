import numpy as np
import os
import sys

query_folder = '../QCR_Lab_1/query/'
ref_folder = '../QCR_Lab_1/ref/'

query_filenames = [filename for filename in sorted(os.listdir(query_folder + 'odo/'))]
ref_filenames = [filename for filename in sorted(os.listdir(ref_folder + 'odo/'))]

query_odos = np.array([np.loadtxt(query_folder+'odo/'+filename, delimiter=',') for filename in query_filenames])
# ref_odos = np.array([np.loadtxt(ref_folder+'odo/'+filename, delimiter=',') for filename in ref_filenames])
ref_odos = np.load(ref_folder+'rectified_odo.npy')
print(query_odos.shape)
print(ref_odos.shape)

ground_truths = []

query_filenames = np.array(query_filenames)
os.makedirs(query_folder+'rectified_distances_to_refs/', exist_ok=True)

for i, qry_odo in enumerate(query_odos):
    # print(query_filenames[i])
    dist = np.linalg.norm(ref_odos-qry_odo, axis=1)
    np.savetxt(query_folder+'rectified_distances_to_refs/'+query_filenames[i],dist,delimiter=',')
    # print(dist.shape)
    # print(np.argmin(dist).shape)
    ground_truths.append(np.argmin(dist))
    # ground_truths.append(np.where(dist==np.min(dist))[0])

print("Ground truth size: {}".format(np.array(ground_truths).shape))

# np.savetxt(query_folder+'rectified_ground_truth_file.csv',ground_truths,delimiter=',')
# np.save(query_folder+'rectified_ground_truth_file.npy',ground_truths)