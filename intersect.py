'''
From http://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
A simple program to detect line intersections.
'''

def ccw(A,B,C):
	return (C[0]-A[0])*(B[1]-A[1]) > (B[0]-A[0])*(C[1]-A[1])

def intersects(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
