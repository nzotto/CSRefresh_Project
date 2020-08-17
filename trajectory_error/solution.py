#!/usr/bin/env python
# coding: utf-8

# # Computer Science Refresher Project 1
# 
# ## Description
# 
# The problem given was the following : we want to set up an indoor location system, like an indoor GPS. In this project we will focus on computing the error of detection of the position of a moving object. Thus, we have the trajectory that the object is supposed to follow and the one that the location system says it has followed. The goal is to compute the error of detected trajectory. The given result should be the ratio between the area between the two trajectories, and the length of the theoretical path.
# 
# ## Solution: 
# _Divide and Conquer !_
# Divide the problem when:
# - change of segment in the theoretical trajectory
# - intersection of the theoretical and experimental trajectories
# - reach a _vertex_ in the experimental trajectory
# 
# How to calculate the error:
# 
# __The total area between the two paths is the sum of the areas of the "trapeziums" formed by two points on the theoretical trajectory and two points of the experimental trajectory.__
# When two points from the experimental and theoretical trajectories are equal, the area is a triangle. 
# 
# ## Note on data structure :
# The data structure chosen by the group to model a test case is the following :
# 
# A sequence of ints, only separated by commas, without any bracket, representing the abscissas of the theoretical path
# A sequence of ints, only separated by commas, without any bracket, representing the ordinates of the theoretical path
# A sequence of ints, only separated by commas, without any bracket, representing the abscissas of the experimental path
# A sequence of ints, only separated by commas, without any bracket, representing the ordinates of the experimental path
# The result expected.
# The error tolerated.
# 
# However, ours is :
# Each path is a list of tuples, each tuple representing the coordinates of a point. This explains the function oracle_parser in test.ipynb and test.py.

# In[1]:


#imports

import numpy as np
import numpy.linalg as lin

# In[2]:


def trajectory_error(coord_th,coord_exp):
    """
    Compute the error between the theoretical and the experimental trajectories,
    in the format decided, that is, the area between the two paths divided by
    the length of the theoretical path.
    arg1 coord_th : a list of tuples (x,y). The path which should be followed.
    arg2 coord_exp: a list of tuples (x,y). The points received from the "indoors-gps"
    return: error, the total area difference betweeen the theoretical trajectory
    and the experimental one, divided by the length of coord_th.
    """
    # initialize values:
    area = 0 # the total area between the theoretical and experimental paths
    distance = path_length(coord_th) # total theoretical path length
    j = 0 # iterator over the experimental path
    i = 0 # iterator over the theoretical path
    
    # start iterations:
    while (i+1 < len(coord_th) and j+1 < len(coord_exp)):     
        # we work on a subsegment [coord_th[i], coord_th[i+1]]        
        # search for an intersection:
        intersect_point = intersection(coord_th[i], coord_th[i+1], coord_exp[j], coord_exp[j+1]) 
        # compute orthogonal projections:
        ort_proj1 =  ortogonal_projection(coord_th[i], coord_th[i+1], coord_exp[j])
        ort_proj2 =  ortogonal_projection(coord_th[i], coord_th[i+1], coord_exp[j+1])
                
        if intersect_point:
            
            if intersect_point == coord_th[i+1] == coord_exp[j]:
                # we're outside of the subsegment
                i += 1 # advance along the theoretical path                 
                
            elif point_belongs_to_segment(coord_th[i], coord_th[i+1], ort_proj1)\
            and point_belongs_to_segment(coord_th[i], coord_th[i+1], ort_proj2):
                # Before the intersection:
                # the points form the right triangle -> coord_exp[j], its projection, intersection
                base = seg_length(ort_proj1, intersect_point)
                height =  seg_length(coord_exp[j], ort_proj1)
                area += area_right_triangle(base, height)
                
                # After the intersection:
                # the points form the right triangle -> intersect_point, coord_exp[j+1], its projection
                base = seg_length(intersect_point, ort_proj2)
                height =  seg_length(coord_exp[j+1], ort_proj2)
                area += area_right_triangle(base, height)
                
                j += 1 # advance along the experimental path 
            
            else:
                # Before the intersection:
                # the points form the right triangle -> coord_exp[j], its projection, intersection
                base = seg_length(coord_exp[j], intersect_point)
                height =  seg_length(coord_th[i],\
                                     ortogonal_projection(coord_exp[j], intersect_point, coord_th[i]))
                area += area_right_triangle(base, height)
                
                # After the intersection:
                # the points form the right triangle -> intersect_point, coord_exp[j+1], its projection
                base = seg_length(intersect_point, coord_exp[j+1])
                height =  seg_length(coord_th[i+1],\
                                     ortogonal_projection(coord_exp[j+1], intersect_point, coord_th[i+1]))
                area += area_right_triangle(base, height)
                
                j += 1 # advance along the experimental path 
                
        elif point_belongs_to_segment(coord_th[i], coord_th[i+1], ort_proj1)\
        and point_belongs_to_segment(coord_th[i], coord_th[i+1], ort_proj2):     
            
            area += compute_area(ort_proj1, ort_proj2, coord_exp[j], coord_exp[j+1])
            j += 1 # advance along the experimental path  
        
        else:
            # we're outside of the subsegment
            i += 1 # advance along the theoretical path
            
        ## TODO check for backtracking:
        
        # a) if ort1 or ort2 in in segment [ coord_th[i] , prevOrt1 ] or [ coord_th[i] , prevOrt2 ]
        
        # b) identify the area to remove 

    return area / distance


def seg_length(x_1,x_2):
    """
    Compute the length of a segment.
    arg1 x_1 : A tuple, the coordinates of the first point of the segment
    arg2 x_2 : A tuple, the coordinates of the second point of the segment
    return : A float, the length of the segment made up by x_1 and x_2
    """
    return(np.sqrt((x_1[0]-x_2[0])**2 + (x_1[1]-x_2[1])**2))


def path_length(path):
    """
    Compute the length of a path.
    arg1 path : A table of tuples, each tuple representing the coordinates of a point of the path.
    return : leng. A float, 
    """
    assert len(path) > 1
    leng = 0
    for i in range(len(path)-1):
        leng += seg_length(path[i],path[i+1])
    return(leng)

def point_belongs_to_segment(x_1, x_2, y):
    """
    Check if a point belongs to a segment.
    
    arg1,2 x_i: A tuple, the coordinates of a point on the theoretical trajectory
    arg3 y: A tuple, the coordinates of a point on the experimental trajectory
    return: A boolean, True if y belongs to the segment [x_1,x_2]. False otherwise.
    """
    # A necessary condition is that x_1, x_2 and y are aligned.
    # Otherwise y can't be in the segment.
    # compute the cross product of the points : if it is 0, it means they are not aligned.
    # Given the calculation performed, if the result is a float we consider that
    # if it is smaller than 10**(-10), it is zero up to machine error.
    cross_product = (y[1] - x_1[1]) * (x_2[0] - x_1[0]) - (y[0] - x_1[0]) * (x_2[1] - x_1[1])
    if (isinstance(cross_product, int) and abs(cross_product) != 0)\
    or (isinstance(cross_product,float) and abs(cross_product) > 0.0000000001):
        return False

    # Now, the case when the points are aligned : we compute the dot product
    # between (y-x_1) and (x_2-x1).
    dot_product = (y[0] - x_1[0])*(x_2[0] - x_1[0]) + (y[1] - x_1[1])*(x_2[1] - x_1[1])
    if dot_product < 0:
        # Then y is beyond x_1, not between the two points.
        return False

    squared_segment_length = seg_length(x_1,x_2)**2
    if dot_product > squared_segment_length:
        # Then y is beyond x_2, not between the two points.
        return False

    return True

def intersection(x_1, x_2, y_1, y_2):
    """
    Compute the intersection of two segments, if any.
    arg1,2 x_i: A tuple, the coordinates of a point on the theoretical trajectory
    arg3,4 y_i: A tuple, the coordinates of a point on the experimental trajectory
    return: intersect. A tuple, the coordinates of the intersection.
    Remark : if the intersection is not in the two segments, return None.
    """

    # First case : the two lines are vertical
    if (x_1[0] == x_2[0] and y_1[0] == y_2[0]) :
        intersect = None
    
    # Second case : only the first line is vertical
    elif x_1[0] == x_2[0] :
        a = (y_1[1]-y_2[1])/(y_1[0]-y_2[0])
        b = y_2[1] - a * y_2[0]
        intersect = (x_1[0],a*x_1[0] + b)
    
    # Third case : only the second line is vertical
    elif y_1[0]  == y_2[0] :
        a = (x_1[1]-x_2[1])/(x_1[0]-x_2[0])
        b = x_2[1] - a * x_2[0]
        intersect = (y_1[0],a*y_1[0] + b)
    
    # Last case : general case
    else :
        a_x = (x_1[1] - x_2[1])/(x_1[0] - x_2[0])
        b_x = x_1[1] - a_x * x_1[0]
        a_y = (y_1[1] - y_2[1])/(y_1[0] - y_2[0])
        b_y = y_1[1] - a_y * y_1[0]
        
        if a_x == a_y :
            # Case where the lines are parallel
            intersect = None
        
        else :
            #The equations are
            #y = a_x*x + b_x
            #y = a_y*x + b_y
            # The solution is the inverted matrix
            intersect = ( (b_y-b_x)/(a_x-a_y),\
                          (b_y * a_x - b_x * a_y)/(a_x - a_y) )
    # If a point is found, check if it belongs to the segments
    # and not only to the lines.
    if intersect:
        if not point_belongs_to_segment(x_1, x_2, intersect) and not point_belongs_to_segment(y_1, y_2, intersect):
            intersect = None
    
    return intersect

def ortogonal_projection(x_1, x_2, y):
    """
    Compute the orthogonal projection of y on the line made up of the points x_1 and x_2.
    
    arg1,2 x_i: A tuple, the coordinates of a point on the theoretical trajectory
    arg3 y: A tuple, the coordinates of a point on the experimental trajectory
    return: new_point. A tuple, the coordinates of the orthogonal projection on y on the segment [x_1,x_2]
    """
    
    if point_belongs_to_segment(x_1, x_2, y): # y already belongs to the segment [x_1, x_2]
        new_point = y
        
    # Now we want to find the equation of the line containing x_1 and x_2.
    # We have to tackle the case of the vertical line first, because then
    # we can't find a slope. Then we can tackle the other case,
    # with a slope and intercept.
    
    elif x_1[0] == x_2[0] :
        # First case : the line containing x_1 and x_2 is vertical
        new_point = (x_1[0],y[1])
    
    else :
    # Second case : the line containing x_1 and x_2 is not vertical,
    # so we look for its slope and intercept.
        a = (x_2[1]-x_1[1])/(x_2[0]-x_1[0])
        b = x_1[1] - a * x_1[0]
        # Now we know the line containing x_1 and x_2 has the equation y = a * x + b
        # So a normal vector to this line is (a,-1)
        # This vector is directing the normal line passing by the point y and its equation is - x - a * y + c = 0 :
        # let's find c!
        c = y[0] + a*y[1]
        # Now the intersection point is the point (x,y) which verifies :
        # a*x - y = -b
        # -x - a*y = -c
        # We find it by inverting the matrix.
        mat = np.array([[a,-1],[-1,-a]])
        point = np.dot(lin.inv(mat), np.array([-b,-c]))
        new_point = (point[0],point[1])
        
    return new_point

def area_right_triangle(base, height):
    """
    Compute the area of a right triangle.
    
    arg1: base. A numeral, the length of the basis of the triangle 
    arg2: height. A numeral, the length of the basis of the triangle
    return: the area of the rectangle
    """
    return np.abs((base * height) / 2)
                       
def compute_area(x_1, x_2, y_1, y_2):               
    """ 
    Compute the area of the quadrilateral formed by the four given points.
    To compute it, it splits the quadrilateral into two triangles, with (x_1,y_2)
    as splitting line. Thus, be careful of not giving these two points as successive
    points, otherwise the computed area may be different from what you expect.
    
    arg1,2 x_i: A tuple, the coordinates of a point on the theoretical trajectory
    arg3,4 y_i: A tuple, the coordinates of a point on the experimental trajectory
    return: area. An int, the area of the figure formed by the 4 points
    """
    area = 0
    
    # Let's split into two triangles, on either side of (x_1,y_2)
    
    # first triangle:
    base = seg_length(x_1, y_2)
    height = seg_length(x_2, ortogonal_projection(x_1, y_2, x_2))
    area += area_right_triangle(base, height)
    
    # second triangle:
    base = seg_length(x_1, y_2)
    height =  seg_length(y_1, ortogonal_projection(x_1, y_2, y_1))
    area += area_right_triangle(base, height)

    return area

