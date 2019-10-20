#!/usr/bin/env python
# coding: utf-8

# In this file, we will perform unit testing for the object-oriented version of the solution.
# 
# ## Format of the paths: 
# p = [P1,P2,...] where P1, P2 are Points.

# In[1]:


# Imports

import matplotlib.pyplot as plt
import numpy as np
import unittest

import ../oop_solutions as sl

# In[2]:


class test_point(unittest.TestCase):
    """
    Unit testing for the Point object.
    """
    # equality function:
    def test_equal(self):
        
        self.assertEqual(sl.Point(0,0), sl.Point(0,0))
        
    def test_not_equal(self):
        
        self.assertFalse(sl.Point(0,0) == sl.Point(1,1))
    
    # distance function:
    def test_distance_xaxis(self):
        
        self.assertEqual(1, sl.Point(0,0).distance(sl.Point(1,0)))
        
    def test_distance_xaxis_negative(self):
        
        self.assertEqual(1, sl.Point(0,0).distance(sl.Point(-1,0)))
    
    def test_distance_yaxis(self):
        
        self.assertEqual(1, sl.Point(0,0).distance(sl.Point(0,1)))
        
    def test_distance_yaxis_negative(self):
        
        self.assertEqual(1, sl.Point(0,0).distance(sl.Point(0,-1)))
        
    def test_distance_diagonal(self):
        
        np.testing.assert_almost_equal(1.41421, sl.Point(0,0).distance(sl.Point(1,1)), 4)
        
    def test_distance_diagonal_negative(self):
        
        np.testing.assert_almost_equal(1.41421, sl.Point(0,0).distance(sl.Point(-1,-1)), 4)

# In[3]:


class test_segment(unittest.TestCase):
    """
    Unit testing for the Segment object.
    """
    # length function:
    def test_null_segment(self):
        a = sl.Point(5,5)
        b = sl.Point(5,5)
        
        self.assertEqual(0, sl.Segment(a,b).length())
    
    def test_simplest_segment(self):
        a = sl.Point(0,0)
        b = sl.Point(1,0)
        
        self.assertEqual(1, sl.Segment(a,b).length())
        
    def test_simplest_segment_inverted(self):
        a = sl.Point(0,0)
        b = sl.Point(1,0)
        
        self.assertEqual(1, sl.Segment(a,b).length())

    # path_length function:
    def test_simplest_path(self):
        path = [sl.Point(0,0),sl.Point(0,1),sl.Point(1,1)]
        
        self.assertEqual(2, sl.Segment.path_length(path))
        
    def test_simplest_path_inverted(self):
        invPath = [sl.Point(0,0),sl.Point(0,-1),sl.Point(-1,-1)]
        
        self.assertEqual(2, sl.Segment.path_length(invPath))
        
    # intersection function:
    def test_simplest_intersection(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(2,2)
        segB_1 = sl.Point(2,0)
        segB_2 = sl.Point(0,2)
        
        self.assertEqual(sl.Point(1.0,1.0), sl.Segment(segA_1,segA_2).intersection(sl.Segment(segB_1,segB_2)))
        
    #def test_no_intersection(self):
    #    segA_1 = sl.Point(0,0)
    #    segA_2 = sl.Point(0,1)
    #    segB_1 = sl.Point(1,0)
    #    segB_2 = sl.Point(1,1)
    #    
    #    self.assertEqual(None, sl.Segment(segA_1,segA_2).intersection(sl.Segment(segB_1,segB_2)))
        
    def test_common_point(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(2,2)
        segB_1 = sl.Point(0,0)
        segB_2 = sl.Point(0,-1)
        
        self.assertEqual(sl.Point(0.0,0.0), sl.Segment(segA_1,segA_2).intersection(sl.Segment(segB_1,segB_2)))
        
    def test_permutation_order_one(self):
         ## same segments as test_simplestIntersection but given in a different order
        segA_1 = sl.Point(2,2)
        segA_2 = sl.Point(0,0)
        segB_1 = sl.Point(2,0)
        segB_2 = sl.Point(0,2)
        
        self.assertEqual(sl.Point(1.0,1.0), sl.Segment(segA_1,segA_2).intersection(sl.Segment(segB_1,segB_2)))
        
    def test_permutation_order_two(self):
         ## same segments as test_simplestIntersection but given in a different order
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(2,2)
        segB_1 = sl.Point(0,2)
        segB_2 = sl.Point(2,0)
        
        self.assertEqual(sl.Point(1.0,1.0), sl.Segment(segA_1,segA_2).intersection(sl.Segment(segB_1,segB_2)))
        
    def test_permutation_order_three(self):
         ## same segments as test_simplestIntersection but given in a different order
        segA_1 = sl.Point(2,2)
        segA_2 = sl.Point(0,0)
        segB_1 = sl.Point(0,2)
        segB_2 = sl.Point(2,0)
        
        self.assertEqual(sl.Point(1.0,1.0), sl.Segment(segA_1,segA_2).intersection(sl.Segment(segB_1,segB_2)))
       
    # orthogonal_projection function:
    def test_projection_not_in_segment(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(0,2)
        y = sl.Point(-1,-1)
        
        self.assertEqual(sl.Point(0,-1), sl.Segment(segA_1, segA_2).orthogonal_projection(y))

    def test_projection_on_vertical_line_from_right(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(0,2)
        y = sl.Point(1,1)
        
        self.assertEqual(sl.Point(0,1), sl.Segment(segA_1, segA_2).orthogonal_projection(y))
        
    def test_projection_on_vertical_line_from_left(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(0,2)
        y = sl.Point(-1,1)
        
        self.assertEqual(sl.Point(0,1), sl.Segment(segA_1, segA_2).orthogonal_projection(y))
        
    def test_projection_on_horizontal_line_from_above(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(2,0)
        y = sl.Point(1,1)
        
        self.assertEqual(sl.Point(1,0), sl.Segment(segA_1, segA_2).orthogonal_projection(y))
    
    def test_projection_on_horizontal_line_from_below(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(2,0)
        y = sl.Point(1,-1)
        
        self.assertEqual(sl.Point(1,0), sl.Segment(segA_1, segA_2).orthogonal_projection(y))
    
    def test_already_in_segment(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(0,2)
        y = sl.Point(0,1)
        
        self.assertEqual(sl.Point(0,1), sl.Segment(segA_1, segA_2).orthogonal_projection(y))
        
    def test_vertically_aligned_above_segment(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(0,1)
        y = sl.Point(0,2)
        
        self.assertEqual(sl.Point(0,2), sl.Segment(segA_1, segA_2).orthogonal_projection(y))
        
    def test_vertically_aligned_under_segment(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(0,1)
        y = sl.Point(0,-1)
        
        self.assertEqual(sl.Point(0,-1), sl.Segment(segA_1, segA_2).orthogonal_projection(y))
    
    def test_diagonally_aligned_above_segment(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(1,1)
        y = sl.Point(2,2)
        
        self.assertEqual(sl.Point(2,2), sl.Segment(segA_1, segA_2).orthogonal_projection(y))
    
    # point_belongs_to_segment function:
    def test_clearly_belongs_to_segment(self):
        seg_A = sl.Point(0,0)
        seg_B = sl.Point(2,2)
        point = sl.Point(1,1)
        
        self.assertTrue(sl.Segment(seg_A, seg_B).point_belongs_to_segment(point))
    
    def test_inside_below(self):
        seg_A = sl.Point(0,0)
        seg_B = sl.Point(2,2)
        point = sl.Point(1,0.5)
        
        self.assertFalse(sl.Segment(seg_A, seg_B).point_belongs_to_segment(point))
        
    def test_inside_above(self):
        seg_A = sl.Point(0,0)
        seg_B = sl.Point(2,2)
        point = sl.Point(1,1.5)
        
        self.assertFalse(sl.Segment(seg_A, seg_B).point_belongs_to_segment(point))
        
    def test_left(self):
        seg_A = sl.Point(0,0)
        seg_B = sl.Point(2,2)
        point = sl.Point(-1,1)
        
        self.assertFalse(sl.Segment(seg_A, seg_B).point_belongs_to_segment(point))
        
    def test_right(self):
        seg_A = sl.Point(0,0)
        seg_B = sl.Point(2,2)
        point = sl.Point(4,1)
        
        self.assertFalse(sl.Segment(seg_A, seg_B).point_belongs_to_segment(point))
        
    def test_above(self):
        seg_A = sl.Point(0,0)
        seg_B = sl.Point(2,2)
        point = sl.Point(1,4)
        
        self.assertFalse(sl.Segment(seg_A, seg_B).point_belongs_to_segment(point))
        
    def test_below(self):
        seg_A = sl.Point(0,0)
        seg_B = sl.Point(2,2)
        point = sl.Point(1,-2)
        
        self.assertFalse(sl.Segment(seg_A, seg_B).point_belongs_to_segment(point))
        
    # compute_area function
    def test_c_a_identical_segments(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(2,2)
        segB_1 = sl.Point(0,0)
        segB_2 = sl.Point(2,2)
        
        self.assertEqual(0, sl.Segment(segA_1, segA_2).compute_area(sl.Segment(segB_1, segB_2)))
        
    def test_c_a_square(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(0,1)
        segB_1 = sl.Point(1,1)
        segB_2 = sl.Point(1,0)
        
        self.assertEqual(1, sl.Segment(segA_1, segA_2).compute_area(sl.Segment(segB_1, segB_2)))
                         
    def test_c_a_trapezium_simplest(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(0,1)
        segB_1 = sl.Point(1,0)
        segB_2 = sl.Point(2,1)
        
        np.testing.assert_almost_equal(1.5, sl.Segment(segA_1, segA_2).compute_area(sl.Segment(segB_1, segB_2)), 5)
        
    def test_c_a_trapezium_simplest_negative(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(0,-1)
        segB_1 = sl.Point(-1,0)
        segB_2 = sl.Point(-2,-1)
        
        np.testing.assert_almost_equal(1.5, sl.Segment(segA_1, segA_2).compute_area(sl.Segment(segB_1, segB_2)), 5)
        
    def test_c_a_one_obtuse_angle(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(2,0)
        segB_1 = sl.Point(-1,1)
        segB_2 = sl.Point(2,1)
        
        self.assertEqual(2.5, sl.Segment(segA_1, segA_2).compute_area(sl.Segment(segB_1, segB_2)))
    
    def test_c_a_another_obtuse_angle(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(2,0)
        segB_1 = sl.Point(0,1)
        segB_2 = sl.Point(3,1)
        
        self.assertEqual(2.5, sl.Segment(segA_1, segA_2).compute_area(sl.Segment(segB_1, segB_2)))
    
    def test_c_a_two_obtuse_angles(self):
        segA_1 = sl.Point(0,0)
        segA_2 = sl.Point(2,0)
        segB_1 = sl.Point(-1,1)
        segB_2 = sl.Point(3,1)
        
        self.assertEqual(3, sl.Segment(segA_1, segA_2).compute_area(sl.Segment(segB_1, segB_2)))

# In[4]:


class test_triangle(unittest.TestCase):
    """
    Unit testing for the Triangle object.
    """
    def test_flat_triangle(self):
        base = 0
        height = 3
        
        self.assertEqual(0, sl.Triangle.area_right_triangle(base,height))
    
    def test_other_flat_triangle(self):
        base = 3
        height = 0
        
        self.assertEqual(0, sl.Triangle.area_right_triangle(base,height))
    
    def test_normal_triangle(self):
        base = 3
        height = 5
        
        self.assertEqual(7.5, sl.Triangle.area_right_triangle(base,height))
    
    def test_negative_distance(self):
        base = -3
        height = 5
        
        self.assertEqual(7.5, sl.Triangle.area_right_triangle(base,height))
        
    def test_other_negative_distance(self):
        base = 3
        height = -5
        
        self.assertEqual(7.5, sl.Triangle.area_right_triangle(base,height))

# In[5]:


class test_trajectory(unittest.TestCase):
    """
    Unit testing for the Trajectory object.
    """
    def test_equal_paths(self):
        theo = [sl.Point(0,x) for x in range(5)]
        expe = [sl.Point(0,x) for x in range(5)]
        
        self.assertEqual(0, sl.Trajectory(theo, expe).trajectory_error())
    
    def test_equal_paths_with_u_turn(self):
        theo = [sl.Point(0,2), sl.Point(1,2), sl.Point(1,0), sl.Point(0,0)]
        expe = [sl.Point(0,2), sl.Point(1,2), sl.Point(1,0), sl.Point(0,0)]
        
        self.assertEqual(0, sl.Trajectory(theo, expe).trajectory_error())
        
    def test_straight_parallel(self):
        theo = [sl.Point(0,x) for x in range(5)]
        expe = [sl.Point(1,x) for x in range(5)]
        
        np.testing.assert_almost_equal(1, sl.Trajectory(theo, expe).trajectory_error(), 5)
        
    def test_straight_crossing(self):
        theo = [sl.Point(0,0),sl.Point(0,1)]
        expe = [sl.Point(1,0),sl.Point(-1,1)]
        
        self.assertEqual(0.5, sl.Trajectory(theo, expe).trajectory_error())
        
    def test_different_length(self):
        theo = [sl.Point(0,0),sl.Point(5,0)]
        expe = [sl.Point(0,0)] + [sl.Point(x,1) for x in range(6)] + [sl.Point(5,0)]
        
        np.testing.assert_almost_equal(1, sl.Trajectory(theo, expe).trajectory_error(), 5)
        
    def test_different_length_crossing(self):
        theo = [sl.Point(0,0),sl.Point(0,2)]
        expe = [sl.Point(0,0),sl.Point(1,0),sl.Point(-1,1),sl.Point(1,2),sl.Point(0,2)]
        
        self.assertEqual(0.5, sl.Trajectory(theo, expe).trajectory_error())

# In[6]:


# initialisation
loader = unittest.TestLoader()
myTestSuite = unittest.TestSuite()
# add tests
myTestSuite.addTests(loader.loadTestsFromTestCase(test_point))
myTestSuite.addTests(loader.loadTestsFromTestCase(test_segment))
myTestSuite.addTests(loader.loadTestsFromTestCase(test_triangle))
myTestSuite.addTests(loader.loadTestsFromTestCase(test_trajectory))
# run!
runner = unittest.TextTestRunner()
runner.run(myTestSuite)

# In[ ]:




# In[ ]:



