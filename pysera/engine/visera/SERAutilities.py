import numpy as np
from pyhull.convex_hull import ConvexHull
from ...processing.synthesize_RoIs import synthesize_coords



def ind2sub(array,indexes=None):
    array_shape = array.shape

    if len(array_shape) == 0:       # to handle 1 voxel
        array = np.atleast_1d(array)
        array_shape = array.shape

    if len(array_shape) == 3:

        ind = np.arange(0,array_shape[0] * array_shape[1] * array_shape[2],1)
        array_flatted = np.reshape(array, newshape= (array_shape[0] * array_shape[1]* array_shape[2]), order='F')

        if indexes is not None:
            array_flatted = array_flatted[indexes]
            ind = ind[indexes]

        slides = np.floor(ind / (array_shape[0] * array_shape[1])).astype(np.int32)
        cols2 = np.floor(ind / array_shape[0]).astype(np.int32) 
        col_minues = slides * array_shape[1]
        cols = cols2 - col_minues
        rows = ind % array_shape[0]
        indexes = []
        for i in range(0,ind.shape[0]):
            if np.isnan(array_flatted[i]):
                indexes.append(i)

        if len(indexes) > 0:
            rows = np.delete(rows,indexes,None)
            cols = np.delete(cols,indexes,0)
            slides = np.delete(slides,indexes,None)

    elif len(array_shape) == 2:

        ind = np.arange(0,array_shape[0] * array_shape[1],1)
        array_flatted = np.reshape(array,newshape= (array_shape[0] * array_shape[1]),order='F')

        if indexes is not None:
            array_flatted = array_flatted[indexes]
            ind = ind[indexes]

        slides = np.floor(ind / (array_shape[0] * array_shape[1])).astype(np.int32)
        cols = np.floor(ind / array_shape[0]).astype(np.int32)
        rows = ind % array_shape[0]
        indexes = []

        for i in range(0,ind.shape[0]):
            if np.isnan(array_flatted[i]):
                indexes.append(i)

        if len(indexes) >0:
            rows = np.delete(rows,indexes,None)
            cols = np.delete(cols,indexes,None)
            slides = np.delete(slides,indexes,None)

    elif len(array_shape) == 1:
        
        ind = np.arange(0,array_shape[0],1)
        array_flatted = np.reshape(array,newshape= array_shape[0],order='F')

        if indexes is not None:
            array_flatted = array_flatted[indexes]
            ind = ind[indexes]

        cols = np.floor(ind / array_shape[0]).astype(np.int32)
        rows = ind % array_shape[0]

        indexes = []
        for i in range(0,ind.shape[0]):
            if np.isnan(array_flatted[i]):
                indexes.append(i)
        
        if len(indexes) >0:
            rows = np.delete(rows,indexes,None)
            cols = np.delete(cols,indexes,None)
        slides = None


    return rows, cols, slides




def sub2ind(siz,v1,v2,v3):

    if np.min(v1) < 0 or np.max(v1) > siz[0]:
        raise Exception('MATLAB:sub2ind:IndexOutOfRange')
    
    ndx = v1
    s = v1.shape
    s2 = v2.shape

    if s != s2:
        raise Exception('MATLAB:sub2ind:SubscriptVectorSize')

    if np.min(v2) < 0 and np.max(v2) > siz[1]:
        raise Exception('MATLAB:sub2ind:IndexOutOfRange')

    ndx = ndx + np.multiply((v2-1),siz[0])
        
    k = np.cumprod(siz)

    s3 = v3.shape

    if s != s3:
        raise Exception('MATLAB:sub2ind:SubscriptVectorSize')

    if np.min(v3) < 0 and np.max(v3) > siz[2]:
        raise Exception('MATLAB:sub2ind:IndexOutOfRange')

    ndx = ndx + np.dot((v3-1),k[1])


    return ndx


# def modifyBit( n, p, b):
# 	mask = 1 << p
# 	return (n & ~mask) | ((b << p) & mask)


# def bitset(num , rng):
#     res = []
#     for t in rng:
#         res.append(int(modifyBit( num, t, 1)))

#     return res

def bitset(cc,idx,bit):
    
    added_num = int(np.float_power(2,bit-1))
    idx[idx != 0] = added_num
    cc = cc + idx

    return cc


def bitget(value, bit_no):
    modeNum = np.float_power(2,bit_no)
    modeNum_min = np.float_power(2,bit_no-1)
    validNum = list(np.arange(modeNum_min,modeNum,1).astype(np.int64))

    res = np.zeros((value.shape[0],1), dtype=np.float32)
    for i in range(0,value.shape[0]):
        if int(np.mod(value[i],modeNum)) in validNum:
            res[i,0] = 1
        else:
            res[i,0] = 0

    return res.flatten(order='F')



def InterpolateVertices(isolevel,p1x,p1y,p1z,p2x,p2y,p2z,valp1,valp2,col1=None,col2=None):

    # valp1[valp1 == 0] = np.nan
    # valp2[valp2 == 0] = np.nan

    if col1 is None and col2 is None:
        p = np.zeros((p1x.shape[0], 3), dtype=np.float32)
    else:
        p = np.zeros((p1x.shape[0], 4), dtype=np.float32)


    mu = np.zeros((p1x.shape[0], 1), dtype=np.float32).flatten(order='F')
    id = np.abs(valp1-valp2) <  np.multiply ((  10 * np.finfo(float).eps   ), (np.abs(valp1) + np.abs(valp2)))
    id = np.multiply(id,1)
    idd = list(np.where(np.array(id) == 1)[0])

    if np.any(idd):
        idd = list(idd)
        p[idd, 0:3] = np.transpose([ p1x[idd], p1y[idd], p1z[idd] ])

        if col1 is not None and col2 is not None:
            p2 = col1[idd]
            p = np.column_stack((p,p2))


    nid = id.copy()
    nid[id == 0] = 1
    nid[id == 1] = 0
    idd = list(np.where(np.array(nid) == 1)[0])
    if np.any(idd):
        idd = list(idd)
        a = np.subtract( isolevel , valp1[idd])
        b = np.subtract(valp2[idd] , valp1[idd])
        # print(a[1940:1990])
        # print(b[1940:1990])
        mu[idd] = np.divide(a ,b )
        a = p1x[idd] + np.multiply(mu[idd] , (p2x[idd] - p1x[idd]))
        b = p1y[idd] + np.multiply(mu[idd] , (p2y[idd] - p1y[idd]))
        c = p1z[idd] + np.multiply(mu[idd] , (p2z[idd] - p1z[idd]))
        p[idd, 0:3] = np.transpose([ a , b , c])

        if col1 is not None and col2 is not None:
            p2 = col1[idd] + np.multiply(mu(idd) , (col2[idd] - col1[idd]))
            p = np.column_stack((p,p2))
        
    
    return p

def MarchingCubes(x,y,z,c,iso,colors):


    PlotFlag = 0
    calc_cols = False
    lindex = 4

    edgeTable, triTable = GetTables()


    if x.ndim != 3 and y.ndim  != 3 and z.ndim  != 3 and c.ndim  != 3:
        raise ValueError('x, y, z, c must be matrices of dim 3')
    

    if x.shape != y.shape or y.shape != z.shape or z.shape != c.shape:
        raise ValueError('x, y, z, c must be the same size')
    

    if np.any(x.shape < (2, 2, 2)):
        raise ValueError('grid size must be at least 2x2x2')
    

    if colors is not None:
        if colors.shape != c.shape:
            raise ValueError('color must be matrix of same size as c')
        
        calc_cols = True
        lindex = 5

    n = np.array(c.shape) - 1

    cc = np.zeros((n[0],n[1],n[2]), dtype=np.float32)
    cc = cc.astype(np.int32)

    newC = c[0:n[0],0:n[1],0:n[2]]              
    idx = newC > iso
    idx = np.multiply(idx, 1)
    cc = bitset(cc,idx,1)

    newC = c[1:n[0]+1,0:n[1],0:n[2]]              
    idx = newC > iso
    idx = np.multiply(idx, 1)
    cc = bitset(cc,idx,2)

    newC = c[1:n[0]+1,1:n[1]+1,0:n[2]]              
    idx = newC > iso
    idx = np.multiply(idx, 1)   
    cc = bitset(cc,idx,3)

    newC = c[0:n[0],1:n[1]+1,0:n[2]]              
    idx = newC > iso
    idx = np.multiply(idx, 1) 
    cc = bitset(cc,idx,4)

    newC = c[0:n[0],0:n[1],1:n[2]+1]              
    idx = newC > iso
    idx = np.multiply(idx, 1)    
    cc = bitset(cc,idx,5)

    newC = c[1:n[0]+1,0:n[1],1:n[2]+1]              
    idx = newC > iso
    idx = np.multiply(idx, 1)
    cc = bitset(cc,idx,6)

    newC = c[1:n[0]+1,1:n[1]+1,1:n[2]+1]              
    idx = newC > iso
    idx = np.multiply(idx, 1)
    cc = bitset(cc,idx,7)

    newC = c[0:n[0],1:n[1]+1,1:n[2]+1]              
    idx = newC > iso
    idx = np.multiply(idx, 1)
    cc = bitset(cc,idx,8)


    
    cc2 = cc+1

    cedge = np.zeros(cc2.shape, dtype=np.float32) - 1
    for i in range(0,cc2.shape[0]):
        for j in range(0,cc2.shape[1]):
            for k in range(0,cc2.shape[2]):
                cedge[i,j,k] = edgeTable[cc2[i,j,k]-1]

    cedgeFlatten = cedge.flatten(order='F')
    id =  np.where(cedgeFlatten>0)[0]   
    
    if len(id) == 0:         
        F = []
        V = []
        col = []
        return
    


    xyz_off = np.array([[1, 1, 1], [2, 1, 1], [2, 2, 1], [1, 2, 1], [1, 1, 2],  [2, 1, 2], [2, 2, 2], [1, 2, 2]])
    edges = np.array([[1, 2], [2, 3], [3, 4], [4 ,1], [5, 6], [6 ,7], [7, 8], [8, 5], [1 ,5], [2, 6], [3, 7], [4, 8]])

    # xyz_off = xyz_off - 1

    offset = sub2ind(c.shape, xyz_off[:,0], xyz_off[:,1], xyz_off[:,2])
    offset = offset -1
    pp = np.zeros((len(id), lindex, 12), dtype=np.float32)
    
    ccedge = np.column_stack((cedgeFlatten[id], np.array(id)))
    ix_offset=0
        
    xFlatten = x.flatten(order='F')
    yFlatten = y.flatten(order='F')
    zFlatten = z.flatten(order='F')
    cFlatten = c.flatten(order='F')

    for jj in range(0,12):     
        id__ = list(bitget(ccedge[:, 0], jj+1))   
        id__ = [int(item) for item in id__]
        idd = list(np.where(np.array(id__) == 1)[0])
        ccedge_id = ccedge[:,1]
        id_ = np.array(ccedge_id[np.array(id__) == 1]).astype(np.int64)
        ix, iy, iz = ind2sub(cc,list(id_))
        id_c = sub2ind(c.shape, ix+1, iy+1, iz+1) - 1
        id1 = list(id_c + offset[edges[jj, 0]-1])
        id2 = list(id_c + offset[edges[jj, 1]-1])

 

        if calc_cols == True:
            colorsFlatten = colors.flatten(order='F')

            interpolate_val = InterpolateVertices(iso, xFlatten[id1], yFlatten[id1], zFlatten[id1],
                xFlatten[id2], yFlatten[id2], zFlatten[id2], cFlatten[id1], cFlatten[id2], colorsFlatten[id1], colorsFlatten[id2])

            nextp = np.transpose( np.arange(1,id_.shape[0]+1,1)) + ix_offset
            nextp = np.expand_dims(nextp,axis=-1).astype(np.float64)
            pp[idd, :, jj] = np.column_stack((interpolate_val, nextp ))

        else:
            interpolate_val = InterpolateVertices(iso, xFlatten[id1], yFlatten[id1], zFlatten[id1],
                xFlatten[id2], yFlatten[id2], zFlatten[id2], cFlatten[id1], cFlatten[id2])
            
            nextp = np.transpose( np.arange(1,id_.shape[0]+1,1)) + ix_offset
            nextp = np.expand_dims(nextp,axis=-1).astype(np.float64)
            pp[idd, :, jj] = np.column_stack((interpolate_val, nextp ))
            # print(pp[:, :, jj])
            
        # pd.DataFrame(pp[:, :, jj]).to_csv('asd_'+str(jj)+'.csv')
        ix_offset = ix_offset + id_.shape[0]
        # print(np.nanmean(pp))
    # print(np.nanmean(pp))
    # F = []
    cc_flatten = cc.flatten(order='F')
    ab = cc_flatten[id] +1 

    tri = np.zeros((ab.shape[0],triTable.shape[1]), dtype=np.float32)
    for i in range(0,tri.shape[0]):
        for j in range(0,tri.shape[1]):
            tri[i,j] = triTable[ab[i]-1,j]

    # tri = triTable[,:]

    pp_flatten = pp.flatten(order='F')

    for jj in range(0,15,3) :
        id_ = np.where(tri[:, jj]>0)[0]
        
        V = np.array(np.column_stack((id_ + 1, lindex*np.ones((id_.shape[0], 1)),tri[id_,jj:jj+3] )))

        if len(V) > 0:
            p1 = list(sub2ind(pp.shape, V[:,0], V[:,1], V[:,2]).astype(np.int64) - 1)
            p2 = list(sub2ind(pp.shape, V[:,0], V[:,1], V[:,3]).astype(np.int64) - 1)
            p3 = list(sub2ind(pp.shape, V[:,0], V[:,1], V[:,4]).astype(np.int64) - 1)

            F2 = np.column_stack((pp_flatten[p1], pp_flatten[p2], pp_flatten[p3]))
            if jj == 0:
                F = F2.copy()
            else:
                F = np.row_stack((F,F2))


    V = []
    col = []
    for jj in range(0,12) :
        idp = list(  np.where(pp[:, lindex-1, jj] > 0)[0]   )
        if np.any(np.array(idp)):
            new_V = list(pp[idp, lindex-1, jj].astype(np.int64))
            # V[new_V, 0:3] = pp[idp, 0:3, jj]
            V.append(pp[idp, 0:3, jj])
            if calc_cols == True:
                new_V = list(pp[idp, lindex-1, jj].astype(np.int64))
                # col[new_V,0] = pp[idp, 3, jj]
                col.append(pp[idp, 3, jj])

    V = np.row_stack(V)
    if len(col) > 0:
        col = np.row_stack(col)

    temp = V.copy()
    temp[np.isnan(V)] = np.inf
    I = np.lexsort((temp[:, 2], temp[:, 1], temp[:, 0])) 
    V = V[I]

    aa = np.diff(V,axis=0)
    bb = np.any(aa>0,axis=1)
    M = list(  np.insert (bb,0,True)    )
    idd = list(np.where(np.array(M) == True)[0])

    V = V[idd,:]
    M = np.multiply(M,1)
    I_rep = np.cumsum(M)
    I2 = np.zeros((I.shape[0],1), dtype=np.float32)
    for i in range(0,I2.shape[0]):
        I2[I[i],0] = I_rep[i]

    F2 = np.zeros((F.shape[0],F.shape[1]), dtype=np.float32)
    for i in range(0,F2.shape[0]):
        F2[i,0] = I2[int(F[i,0])-1]
        F2[i,1] = I2[int(F[i,1])-1]
        F2[i,2] = I2[int(F[i,2])-1]


    return F2,V,col
    


def GetTables():

    edgeTable = [
        0,     265,  515,  778, 1030, 1295, 1541, 1804, 
        2060, 2309, 2575, 2822, 3082, 3331, 3593, 3840, 
        400,   153,  915,  666, 1430, 1183, 1941, 1692, 
        2460, 2197, 2975, 2710, 3482, 3219, 3993, 3728, 
        560,   825,   51,  314, 1590, 1855, 1077, 1340, 
        2620, 2869, 2111, 2358, 3642, 3891, 3129, 3376, 
        928,   681,  419,  170, 1958, 1711, 1445, 1196, 
        2988, 2725, 2479, 2214, 4010, 3747, 3497, 3232, 
        1120, 1385, 1635, 1898,  102,  367,  613,  876, 
        3180, 3429, 3695, 3942, 2154, 2403, 2665, 2912, 
        1520, 1273, 2035, 1786,  502,  255, 1013,  764, 
        3580, 3317, 4095, 3830, 2554, 2291, 3065, 2800, 
        1616, 1881, 1107, 1370,  598,  863,   85,  348, 
        3676, 3925, 3167, 3414, 2650, 2899, 2137, 2384, 
        1984, 1737, 1475, 1226,  966,  719,  453,  204, 
        4044, 3781, 3535, 3270, 3018, 2755, 2505, 2240, 
        2240, 2505, 2755, 3018, 3270, 3535, 3781, 4044, 
        204,   453,  719,  966, 1226, 1475, 1737, 1984, 
        2384, 2137, 2899, 2650, 3414, 3167, 3925, 3676, 
        348,    85,  863,  598, 1370, 1107, 1881, 1616, 
        2800, 3065, 2291, 2554, 3830, 4095, 3317, 3580, 
        764,  1013,  255,  502, 1786, 2035, 1273, 1520, 
        2912, 2665, 2403, 2154, 3942, 3695, 3429, 3180, 
        876,   613,  367,  102, 1898, 1635, 1385, 1120, 
        3232, 3497, 3747, 4010, 2214, 2479, 2725, 2988, 
        1196, 1445, 1711, 1958,  170,  419,  681,  928, 
        3376, 3129, 3891, 3642, 2358, 2111, 2869, 2620, 
        1340, 1077, 1855, 1590,  314,   51,  825,  560, 
        3728, 3993, 3219, 3482, 2710, 2975, 2197, 2460, 
        1692, 1941, 1183, 1430,  666,  915,  153,  400, 
        3840, 3593, 3331, 3082, 2822, 2575, 2309, 2060, 
        1804, 1541, 1295, 1030,  778,  515,  265,    0]

    triTable =[
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
        [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
        [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
        [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
        [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
        [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
        [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
        [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
        [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
        [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
        [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
        [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
        [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
        [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
        [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
        [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
        [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
        [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
        [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
        [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
        [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
        [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
        [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
        [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
        [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
        [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
        [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
        [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
        [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
        [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
        [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
        [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
        [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
        [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
        [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
        [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
        [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
        [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
        [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
        [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
        [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
        [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
        [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
        [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
        [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
        [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
        [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
        [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
        [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
        [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
        [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
        [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
        [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
        [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
        [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
        [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
        [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
        [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
        [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
        [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
        [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
        [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
        [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
        [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
        [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
        [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
        [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
        [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
        [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
        [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
        [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
        [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
        [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
        [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
        [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
        [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
        [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
        [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
        [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
        [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
        [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
        [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
        [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
        [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
        [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
        [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
        [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
        [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
        [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
        [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
        [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
        [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
        [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
        [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
        [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
        [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
        [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
        [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
        [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
        [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
        [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
        [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
        [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
        [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
        [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
        [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
        [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
        [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
        [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
        [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
        [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
        [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
        [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
        [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
        [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
        [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
        [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
        [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
        [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
        [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
        [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
        [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
        [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
        [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
        [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
        [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
        [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
        [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
        [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
        [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
        [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
        [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
        [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
        [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
        [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
        [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
        [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
        [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
        [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
        [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
        [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
        [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
        [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
        [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
        [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
        [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
        [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
        [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
        [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
        [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
        [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
        [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
        [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
        [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
        [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
        [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
        [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
        [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
        [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
        [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
        [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
        [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
        [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
        [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
        [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ]]

    edgeTable = np.array(edgeTable)
    triTable = np.array(triTable) +1 

    return edgeTable, triTable



def multiply_mat(vec, x):

    scale = np.prod(vec)
    if np.isinf(scale):
        for i in range(1, x.shape[1]):
            x[i] = np.prod((x[i], vec))
        
    else:
        x = np.multiply(scale,x)

    return x

def legendre(n,x,normalize = None):

    if n == 0 :
        # if x.ndim == 1:
        #     y = np.ones( (1,x.shape[0]) )
        # else:
        #     y = np.ones(x.shape)
        
        y = 1
        if (normalize is not None) and (normalize == 'norm'):
            y = y/np.sqrt(2)
        

        return y
    
    # if np.isdigit(x):
    sizex = (1,1)
    # else:
        # sizex = x.shape
    # x = np.transpose(x)

    rn = np.arange(0,(2^n),1)
    rootn = np.sqrt(rn)
    s = np.sqrt(   1 -  np.float_power(x,2)    )
    P = np.zeros( (n+3,sizex[0] ), dtype=np.float32)

    # twocot = -2  * np.divide(x ,  s)
    # sn = np.float_power((-s),n)
    # tol = np.sqrt( np.finfo(float).tiny)


    # check plz

    # ind = np.where( s>0 and np.abs(sn) <= tol)[0]
    # if len(ind) > 0:

    #     v = 9.2 - np.divide(np.log(tol)  ,  (n*s[ind]))
    #     w = np.divide(1 , np.log(v))


    #     aa = (1.0058   +  np.multiply( w ,(3.819 - w * 12.173))    )
    #     m1 = 1+  np.multiply(np.multiply(np.multiply( n * s[ind] ,   v ) ,  w ) , aa)
    #     m1 = np.min(n, np.floor(m1))

    #     for k in range (0,m1.shape[0]):
    #         mm1 = m1[k]
    #         col = ind[k]
    #         P[mm1:n+1,col] = 0

    #         tstart = np.finfo(float).eps
    #         P[mm1,col] = np.sign(np.remainder(mm1,2)-0.5)*tstart
    #         if x[col] < 0:
    #             P[mm1,col] = np.sign(np.remainder(n+1,2)-0.5)*tstart
            

    #         sumsq = tol
    #         for m in range (mm1-2,-1,-1):
    #             P[m+1,col] = ((m+1)*twocot[col]*P[m+2,col] - rootn[n+m+3]*rootn[n-m]*P[m+3,col]) / (rootn[n+m+2]*rootn[n-m+1])
    #             sumsq = np.float_power(P[m+1,col],2) + sumsq
            

    #         scale = 1/np.sqrt(2*sumsq - np.float_power(P[1,col],2))
    #         P[1:mm1+1,col] = scale*P[1:mm1+1,col]


    # nind = np.where( x != 1 and np.abs(sn)>=tol)[0]
    # if len(nind) > 0:

    #     d = np.arange(2,(2*n)+1,2)
    #     c = np.prod(   1-  np.divide(1,d)   )

    #     P[n+1,nind] = np.sqrt(c) * sn[nind]

    #     aa = twocot[nind]  *  np.divide(n  ,  rootn[-1])
    #     P[n,nind] = np.multiply(P[n+1,nind] , aa )


    #     for m in range (n-2,-1,-1):

    #         bb = twocot[nind]*(m+1) - P[m+3,nind]*rootn[n+m+3]*rootn[n-m]
    #         P[m+1,nind] = ( np.multiply( P[m+2,nind] , bb )) / (rootn[n+m+2]*rootn[n-m+1])

    # check plz

    y = P[1:n+2,:]

    s0 = np.where(s == 0)[0]
    if len(s0) > 0:

        # print('\ntype y:', type(y))
        # print(f'n: {n}, x: {x}')
        # print("y:", y, '\n')

        # Todo: m-salari: change type y
        # y[0,s0] = np.float_power(x,n)

        y[0, s0] = np.real_if_close(np.float_power(x, n))

    if (normalize is None) or (normalize == 'unnorm'):

        for m in range(0,n-1):
            y[m+1,:] = multiply_mat(rootn[n-m+2:n+m+1],y[m+1,:])


        y[n,:] = multiply_mat(rootn[1:],y[n,:])

    # elif normalize == 'sch':
      
    #     row1 = y[0,:]
    #     y = np.sqrt(2)*y
    #     y[0,:] = row1
    #     const1 = 1
    #     for r in range(1,n+1):
    #         const1 = -const1
    #         y[r,:] = const1*y[r,:]
        
    # elif normalize == 'norm':

    #     y = np.sqrt(n+1/2)*y
    #     const1 = -1
    #     for r in range(0,n+1):
    #         const1 = -const1
    #         y[r,:] = const1*y[r,:]
    # else:
    #     print('MATLAB:legendre:InvalidNormalize', normalize)

    
    if (sizex[0] > 2) or (np.min(sizex) > 1):
        y = np.reshape(y,(n+1 ,sizex),order='F')

    return y




def minrect(x,y,metric, feature_value_mode):


    concat = np.column_stack((x,y))

    # edges = scipy.spatial.ConvexHull(concat)
    # x = x[edges.vertices]
    # y = y[edges.vertices]
    num_points, dim = np.shape(concat)
    if num_points < dim + 1 and feature_value_mode=='APPROXIMATE_VALUE':
        concat = synthesize_coords(concat, num_points, dim, dim+1)

    hull = ConvexHull(concat)

    edges2 = [x for x in hull.vertices if len(x) == 2]
    edges2 = np.array(edges2)

    edges = np.unique(edges2.flatten(order='F'))

    x = x[edges]
    y = y[edges]

    ind = np.arange(0,x.shape[0]-1,1)


    edgeangles = np.arctan2(y[ind+1] - y[ind],x[ind+1] - x[ind])
    edgeangles = np.unique(np.mod(edgeangles,np.pi/2))
    nang = len(edgeangles)
    area = np.inf
    perimeter = np.inf
    met = np.inf
    xy = np.column_stack ((x,y))

    for i in range(0,nang):
        theta= edgeangles[i]
        if theta != 0:
            theta = theta * (-1)
        rot_i =  np.array([[np.cos(theta) ,np.sin(theta)] ,    [-np.sin(theta), np.cos(theta)]])
        rot_i[rot_i == -0] = 0
        xyr = np.dot( xy,rot_i)
        xymin = np.min(xyr,axis=0)
        xymax = np.max(xyr,axis=0)
        A_i = np.prod(xymax - xymin)
        P_i = 2*np.sum(xymax-xymin)


        if metric=='v':
            M_i = A_i
        else: 
            M_i = P_i

        if M_i<met:
            rot = rot_i
            met = M_i
    

    return rot


def edgelist(K):

    e = [K[:,1],K[:,2]]
    e = [e,K[:,2],K[:,3]]
    e = [e,K[:,1],K[:,3]]
    e = [np.min(e,[],2),np.max(e,[],2)]
    ne = e.shape[0]
    e = [e,e[:,2]+ne*e[:,1]]
    nerd,I = np.sort(e[:,3])
    e = e[I[1:2:],:]


    return e


def euler123(v1,v2,nv):

    aa = np.abs(nv[:,0])
    aa [aa > 1] =1
    aa [aa < -1] = - 1
    bb = np.sign(nv[:,0])
    beta = np.arcsin(  np.multiply ( bb , aa ))

    alpha= np.zeros(beta.shape, dtype=np.float32)

    i1 = np.where(nv[:,0]==1)[0]
    lst1 = list(np.arange(0,nv.shape[0],1))
    lst2 = i1
    i2=[]
    for i in lst1:
        if i not in lst2:
            i2.append(i)
    i2 = np.array(i2).astype(np.uint16)
    if len(i1) >0: 
        aa = np.abs(v2[i1,2]) 
        # aa [aa > 1] = 1
        # aa [aa < -1] = - 1
        bb = np.sign(v2[i1,2])
        alpha[i1] = np.arcsin(  np.multiply ( bb , aa ))    

    if len(i2) > 0: 
        a1 = np.abs(   np.divide( nv[i2,2] ,np.cos(beta[i2])  ))
        a1 [a1 > 1] = 1
        a1 [a1 < -1] = - 1
        a2 = np.sign(  np.divide(nv[i2,2] ,np.cos(beta[i2]) ) )
        alpha[i2] = np.arccos(np.multiply(a2,a1) )


        i3 = np.where(np.sign(nv[i2,1]) !=  np.sign(np.multiply (-np.sin(alpha[i2]),np.cos(beta[i2]))))[0]
        indexes = i2[i3]
        
        alpha[indexes] = np.multiply(-1 , alpha[indexes])


    gamma= np.zeros(alpha.shape, dtype=np.float32)

    if len(i2) > 0:

        singamma = np.divide(  np.multiply(v2[i2,0],-1),np.cos(beta[i2]))
        i21 = np.where(v1[i2,0]>=0)[0]

        lst1 = list(np.arange(0,i2.shape[0],1))
        lst2 = i21
        i22=[]
        for i in lst1:
            if i not in lst2:
                i22.append(i)


        aa = np.abs(singamma[i21])
        aa [aa > 1] = 1
        aa [aa < -1] = - 1
        a = np.arcsin(  np.multiply(np.sign(singamma[i21]), aa )  )  
        gamma[i2[i21]] = a 
        

        aa = np.abs(singamma[i22])
        aa [aa > 1] = 1
        aa [aa < -1] = - 1
        q = np.multiply(-1,np.pi)  - np.arcsin(np.multiply(np.sign(singamma[i22]), aa))    
        gamma[i2[i22]] = q
    return alpha,beta,gamma


def Rmat(alpha,beta,gamma):
    return  [np.cos(beta)*np.cos(gamma), -np.cos(beta)*np.sin(gamma), np.sin(beta),
            np.sin(alpha)*np.sin(beta)*np.cos(gamma)+np.cos(alpha)*np.sin(gamma), -np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), -np.sin(alpha)*np.cos(beta),
            -np.cos(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(alpha)*np.sin(gamma), np.cos(alpha)*np.sin(beta)*np.sin(gamma)+np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.cos(beta)]


def checkbox(alpha,beta,gamma,xyz2,metric,d,rotmat,minmax, feature_value_mode='REAL_VALUE'):

    xyz = xyz2 + 1
    
    rot = Rmat(alpha,beta,gamma)
    rot = np.reshape(rot,(3,3))
    xyz_i = np.dot(xyz,rot)                     

    x_i = xyz_i[:,0]
    y_i = xyz_i[:,1]          
    rot2 = minrect(x_i,y_i,metric, feature_value_mode=feature_value_mode)
    aa = np.column_stack ((np.array(rot2),np.array([0,0])))
    bb = np.row_stack ((   aa  ,   np.array([0,0,1])  ))
    bb[bb == -0] = 0
    rot = np.dot(rot , bb)         
    xyz_i = np.dot(xyz,rot )                          

    xyzmin = np.min(xyz_i,axis = 0)
    xyzmax = np.max(xyz_i,axis = 0)
    h = xyzmax-xyzmin

    if metric == 'v':    
        d_i = h[0]*h[1]*h[2]
    elif metric == 's':   
        d_i = h[0]*h[1]+h[1]*h[2]+h[2]*h[0]
    else:                   
        d_i = np.sum(h)
    

    if d_i < d:
        d = d_i
        rotmat = rot
        minmax = np.row_stack((xyzmin,xyzmax))
    

    return d, rotmat, minmax


def ismember(A,B):

    Lia = np.isin(A,B)
    Lia = np.multiply(Lia,1)
    Locb = np.zeros((Lia.shape[0]), dtype=np.float32)
    for a in range(0,Lia.shape[0]):
        if Lia[a] == 1:
            find = np.where(B == A[a])[0][0]
            Locb[a] = find
        else:
            Locb[a] = 0
    return Lia,Locb



def minboundbox(x,y,z,metric=None,level=None, feature_value_mode='REAL_VALUE'):

    if metric is None:
        metric = 'volume'
    metric = metric[0]

    if level is None:
        level = 4

    # x = x.flatten()
    # y = y.flatten()
    # z = z.flatten()

    n1 = x.shape[0]
    n2 = y.shape[0]
    n3 = z.shape[0]
    if n1 != n2 or n1 != n3 or n2 != n3:
        raise ValueError('x, y and z must be the same sizes')


    try:
        
        concat = np.column_stack((x,y,z))

        num_points, dim = np.shape(concat)
        if num_points < dim + 1 and feature_value_mode=='APPROXIMATE_VALUE':
            concat = synthesize_coords(concat, num_points, dim, dim+1)

        hull = ConvexHull(concat)

        K = [x for x in hull.vertices if len(x) == 3]
        K = np.array(K)

        # K2 = scipy.spatial.ConvexHull(concat,qhull_options='Qt')   # Qt
        # K = K2.simplices
        # Vch = K2.volume

        K2 = K.flatten('F')
        
        Ki = np.unique(K2)
        tf,loc = ismember(K2,Ki)
        K = np.reshape(loc,K.shape,order='F')
        x = x[Ki]
        y = y[Ki]
        z = z[Ki]
        n1 = x.shape[0]
    
    except:
        try:
            concat = np.column_stack((x,y))

            num_points, dim = np.shape(concat)
            if num_points < dim + 1 and feature_value_mode=='APPROXIMATE_VALUE':
                concat = synthesize_coords(concat, num_points, dim, dim+1)

            hull = ConvexHull(concat)

            K = [x for x in hull.vertices if len(x) == 3]
            K = np.array(K)

            # K2 = scipy.spatial.ConvexHull(concat)
            # K = K2.simplices
            # Vch = K2.volume

            K2 = K.flatten('F')


            Ki = np.unique(K2)
            tf,loc = ismember(K2,Ki)
            K = np.reshape(loc,K.shape,order='F')
            x = x[Ki]
            y = y[Ki]
            z = z[Ki]
            n1 = x.shape[0]
            K[:,2] = np.ones(K[:,1].shape)

        except:
            raise ValueError('The number and/or distribution of given points does not allow the construction of a convex hull.')
        
    K = K.astype(np.int32)
    fx = np.column_stack((x[K[:,0]],x[K[:,1]],x[K[:,2]]))
    fy = np.column_stack((y[K[:,0]],y[K[:,1]],y[K[:,2]]))
    fz = np.column_stack((z[K[:,0]],z[K[:,1]],z[K[:,2]]))


    v1 = np.column_stack((fx[:,1]-fx[:,0],fy[:,1]-fy[:,0],fz[:,1]-fz[:,0]))
    # v1 = np.multiply(v1,-1)
    v1_sqrt = np.sqrt(np.sum( np.float_power(v1,2),axis=1))
    v1 = np.divide(v1 , np.column_stack((v1_sqrt,v1_sqrt,v1_sqrt)))
    v2 = np.column_stack((fx[:,2]-fx[:,0],fy[:,2]-fy[:,0],fz[:,2]-fz[:,0]))
    ein = np.einsum('ij,ij->i',v1,v2)
    v2_col = np.column_stack((ein,ein,ein))
    v2 = v2  -   np.multiply(v2_col,v1)
    v2_sqrt = np.sqrt(np.sum(  np.float_power(v2,2),axis=1))
    v3 = np.divide(v2 , np.column_stack((v2_sqrt,v2_sqrt,v2_sqrt)))
    v2 = v3
    nv = np.cross(v1,v2,axis=1)

    alpha,beta,gamma=euler123(v1,v2,nv)

    nang = alpha.shape[0]
    d = np.inf
    rotmat=[]
    minmax=[]
    xyz = np.column_stack((x,y,z))

    if level==1 or level==3:    
        for i in range(0,nang):
            d, rotmat, minmax = checkbox(alpha[i],beta[i],gamma[i],xyz,metric,d,rotmat,minmax, feature_value_mode=feature_value_mode)
            # print(i+1, ' ' ,  d)

    # if level>1:
    #     e = edgelist[K]
    #     ne = e.shape[0]


    # if level==2 or level==4:
    #     for i in range(0,ne):
    #         va = xyz[e[i,0],:]-xyz[e[i,1],:]
    #         va = va/ np.linalg.norm(va)
    #         for j in range(i+1,ne):
    #             vb = xyz[e[j,1],:]-xyz[e[j,2],:]
    #             vb = vb - np.dot(va,vb)*va;
    #             nv = np.cross(va,vb)
    #             if np.sum(np.abs(nv))>0:
    #                 vb = vb/np.linalg.norm(vb)
    #                 nv = nv/np.linalg.norm(nv)
    #                 alp,bet,gam=euler123(va,vb,nv)
    #                 d, rotmat, minmax = checkbox(alp,bet,gam,xyz,metric,d,rotmat,minmax)



    # if level==3 or level==4:
    #     for i in range(0,ne):

    #         va = xyz[e[i,1],:]-xyz[e[i,2],:]
    #         vb = [va[2],-va[1],0]
    #         if np.sum(np.abs(vb))==0:
    #             vb = [va[3],0,-va[1]]

    #         va = va/np.linalg.norm(va)
    #         vb = vb/np.linalg.norm(vb)
    #         nv = np.cross(va,vb)

    #         alp,bet,gam=euler123(va,vb,nv)
    #         d, rotmat, minmax = checkbox(alp,bet,gam,xyz,metric,d,rotmat,minmax)
    #         alp,bet,gam=euler123(vb,nv,va)
    #         d, rotmat, minmax = checkbox(alp,bet,gam,xyz,metric,d,rotmat,minmax)
    #         alp,bet,gam=euler123(nv,va,vb)
    #         d, rotmat, minmax = checkbox(alp,bet,gam,xyz,metric,d,rotmat,minmax)


    h = [[minmax[0,0],minmax[0,1],minmax[0,2]],
        [minmax[1,0],minmax[0,1],minmax[0,2]],
        [minmax[1,0],minmax[1,1],minmax[0,2]],
        [minmax[0,0],minmax[1,1],minmax[0,2]],
        [minmax[0,0],minmax[0,1],minmax[1,2]],
        [minmax[1,0],minmax[0,1],minmax[1,2]],
        [minmax[1,0],minmax[1,1],minmax[1,2]],
        [minmax[0,0],minmax[1,1],minmax[1,2]]]



    cornerpoints = np.dot(h, np.linalg.inv(rotmat))
    h = minmax[1,:]-minmax[0,:]
    volume = h[0]*h[1]*h[2]
    surface = 2*(h[0]*h[1]+h[1]*h[2]+h[2]*h[0])
    edgelength = 4*np.sum(h)




    return rotmat,cornerpoints,volume,surface,edgelength



def  minBoundingBox2D(X,Y,dx,dy,dz, feature_value_mode):

    concat = np.column_stack((X,Y))
    num_points, dim = np.shape(concat)
    if num_points < dim + 1 and feature_value_mode=='APPROXIMATE_VALUE':
        concat = synthesize_coords(concat, num_points, dim, dim+1)

    hull = ConvexHull(concat)

    K = [x for x in hull.vertices if len(x) == 2]
    K = np.array(K)

    k = np.unique(K.flatten(order='F'))
    CH = concat[k,:]

    E = np.diff(a=CH,n=1,axis=1);           
    T = np.arctan2(E[:,1],E[:,0])   
    T = np.unique(np.mod(T,np.pi/2))   


    a = np.reshape( np.tile(T,(2,2)),(2*T.shape[0],2)) 
    b = np.tile(    np.array([[0, -np.pi] , [ np.pi, 0]])  / 2  ,     (T.shape[0],1 ) )
    c = a+b
    R = np.cos(c)

    RCH = R*CH

    bsize = np.max(RCH,axis = 1) - np.min(RCH,axis = 1)
    area  = np.prod(   np.reshape(bsize,(2,bsize.shape[0]/2)))
 
    a,i = np.min(area)

    f = np.array([-1, 0])
    Rf    = R[  2*i+f  ,:]   
    bound = Rf * CH           
    bmin  = np.min(bound,axis=1)
    bmax  = np.max(bound,axis=1)

    Rf = np.transpose(Rf)

    bb = np.zeros((2*T.shape[0],2), dtype=np.float32)

    bb[:,3] = bmax[0]*Rf[:,0] + bmin[1]*Rf[:,1]
    bb[:,0] = bmin[0]*Rf[:,0] + bmin[1]*Rf[:,1]
    bb[:,1] = bmin[0]*Rf[:,0] + bmax[1]*Rf[:,1]
    bb[:,2] = bmax[0]*Rf[:,0] + bmax[1]*Rf[:,1]

    Area = a * dx * dy
    Vol =  a * dx * dy * dz


    return bb, Area, Vol
