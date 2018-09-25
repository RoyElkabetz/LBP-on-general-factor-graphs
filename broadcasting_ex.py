import numpy as np

vec = np.array([1, 2, 3])
mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

matmul = np.matmul(vec, mat)

vec_rep = np.stack([vec] * 3)
vec_rep_t = vec_rep.T

a = mat * vec_rep
a_t = mat * vec_rep_t

b = mat * vec[np.newaxis, :]
b_t = mat * vec[:, np.newaxis]


bc_1 = mat[np.newaxis, :, np.newaxis, :]
bc_2 = np.reshape(mat, [1, mat.shape[0], 1, mat.shape[1]])
node_count = 5
alphabet = [3, 2, 2]
nodes = [1, 3, 4]
factor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])


def broadcasting(factor, nodes, alphabet):
    shape = []
    for i in range(node_count):
        for j in range(len(nodes)):
            if i == nodes[j]:
                shape.append(alphabet[j])
                break
            if j == len(nodes) - 1:
                shape.append(1)
    tensor = np.reshape(factor, shape)
    return tensor


alphabet = np.array([3, 2, 2])
nodes = np.array([1, 3, 4])
factor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])


def broadcasting2(factor, nodes, alphabet):
    new_shape = np.ones(node_count, dtype=np.int)
    new_shape[nodes] = alphabet
    return np.reshape(factor, new_shape)



#tensor = broadcasting(factor, nodes, alphabet)
tensor = broadcasting2(factor, nodes, alphabet)

print('\n')
print('factor shape is ')
print(np.shape(factor))
print('\n')
print('tensor shape is ')
print(np.shape(tensor))
print('\n')
print('nodes = ')
print(nodes)
print('\n')
print('alphabet = ')
print(alphabet)

