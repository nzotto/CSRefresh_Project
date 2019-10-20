#!/usr/bin/env python
# coding: utf-8

# > This is the OOP version of the algorythm presented in the file "solution.ipynb"          
# > The equivalent of the main function of that file (trajectory_error) here belongs to the Trajectory class.

# In[1]:


from math import sqrt
import doctest
import numpy as np
import numpy.linalg as lin

# In[2]:


class Point:
    
    """
    Represents points in the plane, with two attributes :
    x representing the abscissa and y representing the ordinate.
    """
    
    def __init__(self, x,y):
        """
        Constructor.
        arg1 x : the abscissa of the point
        arg2 y : the ordinate of the point
        """
        self.x = x
        self.y = y
    
    def distance(self,point):
        """
        arg1 point : the point to which we want to calculate the distance.
        return : a float, the distance between the two points.
        
        >>> Point(0,0).distance(Point(0,0))
        0.0
        >>> Point(0,0).distance(Point(0,1))
        1.0
        >>> Point(0,0).distance(Point(-1,0))
        1.0
        """
        return sqrt((self.x - point.x)**2+(self.y - point.y)**2)
    
    def __eq__(self,other):
        """
        arg1 other : the point we want to compare to.
        return : a bool, equal to True iff self represents the same point as other,
        ie if it has same abscissa and ordinate.
        """
        return(self.x == other.x and self.y == other.y)

doctest.testmod()

# In[3]:


class Segment:
    
    """
    Represents a segment in the plane.
    Attributes : p1 and p2, of class Point, representing the two extremities of the segment.
    """
    
    def __init__(self,p1,p2):
        """
        Constructor.
        arg1 p1 : the first point of the segment.
        arg2 p2 : the second point of the segment.
        """
        self.p1 = p1
        self.p2 = p2
        
    def length(self):
        """
        Compute the length of a segment.
        return : a float, the length of the segment.
        """
        return((self.p1).distance(self.p2))
    
    def path_length(path):
        """
        Compute the total length of a path.
        arg1 path : a list of Points
        return totalLength : a float, the total length of the path made of the sequence of Points in //path//
        """
        total_length = 0
        for i in range(len(path)-1):
            total_length += Segment(path[i],path[i+1]).length()
        return total_length
    
    def point_belongs_to_segment(self,point):
        """
        Check if a point belongs to a segment.
        arg1 point : A Point
        return: A boolean, True if point belongs to the segment. False otherwise.
        """
        # A necessary condition is that the segment and point are aligned.
        # Otherwise point can't be in the segment.
        # compute the cross product of the points : if it is 0, it means they are not aligned.
        # Given the calculation performed, if the result is a float we consider that
        # if it is smaller than 10**(-10), it is zero up to machine error.
        cross_product = (point.y - self.p1.y) * (self.p2.x - self.p1.x)\
                       - (point.x - self.p1.x) * (self.p2.y - self.p1.y)
        if (isinstance(cross_product, int) and abs(cross_product) != 0)\
        or (isinstance(cross_product,float) and abs(cross_product) > 0.0000000001):
            return False

        # Now, the case when the points are aligned : we compute the dot product
        # between (point - self.p1) and (self.p2 and self.p1).
        dot_product = (point.x - self.p1.x)*(self.p2.x - self.p1.x)\
                     + (point.y - self.p1.y)*(self.p2.y - self.p1.y)
        if dot_product < 0:
            # Then point is beyond self.p1, not between the two points.
            return False

        squared_segment_length = self.length()**2
        if dot_product > squared_segment_length:
            # Then point is beyond self.p2, not between the two points.
            return False

        return True
    
    def intersection(self,other):
        """
        Compute the intersection of two segments, if any.
        arg1 other : a Segment, the one with which we want to compute the intersection
        return intersect : a Point, the intersection of the two segments.
        Remark : if the intersection is not in the two segments, return None.
        """
        # First case : the two lines are vertical
        if (self.p1.x == self.p2.x and other.p1.x  == other.p1.y) :
            intersect = None

        # Second case : only the first line is vertical
        elif self.p1.x == self.p2.x :
            a = (other.p1.y-other.p2.y)/(other.p1.x-other.p1.y)
            b = other.p2.y - a * other.p1.y
            intersect = Point(self.p1.x,a*self.p1.x + b)

        # Third case : only the second line is vertical
        elif other.p1.x  == other.p1.y :
            a = (self.p1.y-self.p2.y)/(self.p1.x-self.p2.x)
            b = self.p2.y - a * self.p2.x
            intersect = Point(other.p1.x,a*other.p1.x + b)

        # Last case : general case
        else :
            a_x = (self.p1.y - self.p2.y)/(self.p1.x - self.p2.x)
            b_x = self.p1.y - a_x * self.p1.x
            a_y = (other.p1.y - other.p2.y)/(other.p1.x - other.p1.y)
            b_y = other.p1.y - a_y * other.p1.x

            if a_x == a_y :
                # Case where the lines are parallel
                intersect = None

            else :
                #The equations are
                #y = a_x*x + b_x
                #y = a_y*x + b_y
                # The solution is the inverted matrix
                intersect = Point( (b_y-b_x)/(a_x-a_y),\
                              (b_y * a_x - b_x * a_y)/(a_x - a_y) )
        # If a point is found, check if it belongs to the segments and not only the lines (self.p1.x<intersect[0]<self.p2.x or...)
        if intersect :
            if not self.point_belongs_to_segment(intersect):
                intersect = None

        return(intersect)
    
    def orthogonal_projection(self,point):
        """
        Compute the orthogonal projection of //point// on the segment.
        arg1 point : a Point.
        return : a Point, the orthogonal projection of point onto self.
        """
        if self.point_belongs_to_segment(point):
            new_point = point

        # Now we want to find the equation of the line containing the segment.
        # We have to tackle the case of the vertical line first, because then
        # we can't find a slope. Then we can tackle the other case,
        # with a slope and intercept.

        elif self.p1.x == self.p2.x :
            # First case : the line containing the segment is vertical
            new_point = Point(self.p1.x,point.y)

        else :
        # Second case : the line containing the segment is not vertical,
        # so we look for its slope and intercept.
            a = (self.p2.y - self.p1.y)/(self.p2.x - self.p1.x)
            b = self.p1.y - a * self.p1.x
            # Now we know the line containing the segment has the equation y = a * x + b
            # So a normal vector to this line is (a,-1)
            # This vector is directing the normal line passing by the point y
            # and its equation is - x - a * y + c = 0 :
            # let's find c!
            c = point.x + a * point.y
            # Now the intersection point is the point (x,y) which verifies :
            # a*x - y = -b
            # -x - a*y = -c
            # We find it by inverting the matrix.
            mat = np.array([[a,-1],[-1,-a]])
            point_vector = np.dot(lin.inv(mat), np.array([-b,-c]))
            new_point = Point(point_vector[0],point_vector[1])

        return new_point
    
    def compute_area(self, other):               
        """ 
        Compute the area of the quadrilateral formed by the two segments self and other.
        To compute it, it splits the quadrilateral into two triangles, with (self.p1,other.p2)
        as splitting line. Thus, be careful of not giving these two points as successive
        points, otherwise the computed area may be different from what you expect.

        arg1 other : a segment of the experimental trajectory.
        return: area. An int, the area of the figure formed by self and other.
        """
        area = 0

        # Let's split into two triangles, on either side of (self.p1,other.p2)

        # first triangle:
        base = Segment(self.p1, other.p2).length()
        height = Segment(self.p2, Segment(self.p1, other.p2).orthogonal_projection(self.p2)).length()
        area += Triangle.area_right_triangle(base, height)

        # second triangle:
        base = Segment(self.p1, other.p2).length()
        # base = self.length(self.p1, other.p2)
        height = Segment(other.p1, Segment(self.p1, other.p2).orthogonal_projection(other.p1)).length()
        # height =  self.length(other.p1, orthogonal_projection(self.p1, other.p2, other.p1))
        area += Triangle.area_right_triangle(base, height)

        return area
    

# In[4]:


class Triangle:
    
    def __init__(self,p1,p2,p3):
        """
        Constructor.
        arg1 p1 : a Point
        arg2 p2 : a Point
        arg3 p3 : a Point
        """
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
    
    def area(self):
        """
        return : the area of the triangle.
        """
        return(0.5*np.linalg.det(np.array([[self.p2.x - self.p1.x,self.p3.x - self.p1.x ],\
                                           [self.p2.y - self.p1.y, self.p3.y - self.p1.y]])))
    
    def area_right_triangle(base, height):
        """
        Compute the area of a right triangle.

        arg1: base. A numeral, the length of the basis of the triangle 
        arg2: height. A numeral, the length of the basis of the triangle
        return: the area of the rectangle
        """
        return np.abs((base * height) / 2)


# In[5]:


class Trajectory:
    
    def __init__(self,theo,expe):
        """
        Constructor.
        arg1 theo : a list of Points, theoretical path
        arg2 expe : a list of Points, experimental path, the real one         
        """
        self.theo = theo
        self.expe = expe
    
    def trajectory_error(self):
        """
        Compute the error between the theoretical and the experimental trajectories,
        in the format decided, that is, the area between the two paths divided by
        the length of the theoretical path.
        arg1 coord_th : a list of Points. The path which should be followed.
        arg2 coord_exp: a list of Points. The points received from the "indoors-gps"
        return: error, the total area difference betweeen the theoretical trajectory
        and the experimental one, divided by the length of the theoretical path.
        """
        # initialize values:
        area = 0 # the total area between the theoretical and experimental paths
        distance = Segment.path_length(self.theo) # total theoretical path length
        j = 0 # iterator over the experimental path
        i = 0 # iterator over the theoretical path

        # start iterations:
        while (i+1 < len(self.theo) and j+1 < len(self.expe)):     
            # we work on a subsegment [coord_th[i], coord_th[i+1]]        
            # search for an intersection:
            intersect_point = Segment(self.theo[i], self.theo[i+1]).intersection(Segment(self.expe[j], self.expe[j+1]))
            # compute orthogonal projections:
            splitting_segment = Segment(self.theo[i], self.theo[i+1])
            ort_proj1 =  splitting_segment.orthogonal_projection(self.expe[j])
            ort_proj2 =  splitting_segment.orthogonal_projection(self.expe[j+1])

            if intersect_point:

                if intersect_point == self.theo[i+1] == self.expe[j]:
                    # we're outside of the subsegment
                    i += 1 # advance along the theoretical path                 

                elif splitting_segment.point_belongs_to_segment(ort_proj1)\
                and splitting_segment.point_belongs_to_segment(ort_proj2):
                    # Before the intersection:
                    # the points form the right triangle -> coord_exp[j], its projection, intersection
                    base = Segment(ort_proj1,intersect_point).length()
                    height =  Segment(self.expe[j], ort_proj1).length()
                    area += Triangle.area_right_triangle(base, height)

                    # After the intersection:
                    # the points form the right triangle -> intersect_point, coord_exp[j+1], its projection
                    base = Segment(intersect_point, ort_proj2).length()
                    height =  Segment(self.expe[j+1], ort_proj2).length()
                    area += Triangle.area_right_triangle(base, height)

                    j += 1 # advance along the experimental path 

                else:
                    # Before the intersection:
                    # the points form the right triangle -> expe[j], its projection, intersect_point
                    base = Segment(self.expe[j], intersect_point).length()
                    height = Segment(self.theo[i],\
                                     Segment(self.expe[j],intersect_point).orthogonal_projection(self.theo[i])).length()
                    area += Triangle.area_right_triangle(base, height)

                    # After the intersection:
                    # the points form the right triangle -> intersect_point, coord_exp[j+1], its projection
                    base = Segment(intersect_point, self.expe[j+1]).length()
                    height = Segment(self.theo[i+1],\
                                    Segment(self.expe[j+1],intersect_point).orthogonal_projection(self.theo[i+1])).length()
                    area += Triangle.area_right_triangle(base, height)

                    j += 1 # advance along the experimental path 

            elif Segment(self.theo[i],self.theo[i+1]).point_belongs_to_segment(ort_proj1)\
            and Segment(self.theo[i],self.theo[i+1]).point_belongs_to_segment(ort_proj2):     

                area += Segment(ort_proj1, ort_proj2).compute_area(Segment(self.expe[j], self.expe[j+1]))
                j += 1 # advance along the experimental path  

            else:
                # we're outside of the subsegment
                i += 1 # advance along the theoretical path

        return area / distance
    

# In[ ]:



