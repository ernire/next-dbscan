import h5py

# e.g., /data/bremen/bremen.h5
path_to_input_file=" "
# e.g., /data/bremen/bremen.csv
path_to_output_file=" "

f = h5py.File(path_to_input_file)

list(f)
data=f['DBSCAN']

with open(path_to_output_file, 'w') as f:
    for i in range(0,data.shape[0]):
        f.write(str(data[i,0])+' '+str(data[i,1])+' '+str(data[i,2])+'\n')
