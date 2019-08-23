"""
Image module that is used to obtain the "imaged" indices with respect to a some
reference line

In 1D, suppose that the reference is the grid point index im, then the imaged
indices are

phi(im + j), phi(im - j)
phi(in + j), phi(in - j)


-----o-----o-----o-----*-----o-----o------o  (Configuration I)
    im-3  im-2  im-1  im    im+1  im+2  im+3

   -----o-----o-----o--*--o-----o-----o------o (Configuration II)
       zm-3  zm-2  zm-1  zm    zm+1  zm+2   zm+3 

    zm = im + 1/2

    The function 

    image_node(label, lhs, rhs, bounds) 
        images a field defined at the nodes 
        (configuration I)
    image_cell(label, lhs, rhs) images a field defined at the cells
        (configuration II)

"""

#def image_node(label, lhs, rhs, bounds , regions)

def image_node_indices(normal, shape):
    n = len(shape)
    shift = [0]*n
    shift[axis] = shape[axis]
    sign  = [1]*n
    sign[axis] = -1

    # Suppose normal = (-1, 0)
    # then image: phi(i, j) = phi(-i, j)
    # if normal = (1, 0)
    # then image: phi(m - i, j) = phi(m + i, j)

    #   phi(is)  = phi(is - n*(is - shape[0]))

    lhs = lambda x : [xi + si for xi, si in zip (x, shift)] 
    rhs = lambda x : [ni*xi + si for xi, si in zip (sign, x, shift)] 

