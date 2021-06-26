import numpy as np

# Idea:  
# 1) Draw a horizontal line to the right of each point and extend it to infinity

# 2) Count the number of times the line intersects with polygon edges.

# 3) A point is inside the polygon if either count of intersections is odd or
#    point lies on an edge of polygon.  If none of the conditions is true, then 
#    point lies outside.

# Given three colinear points p, q, r, the function checks if 
# point q lies on line segment 'pr' 
def onSegment(p, q, r):
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True 
    return False 


# To find orientation of ordered triplet (p, q, r). 
# The function returns following values 
# 0 --> p, q and r are colinear 
# 1 --> Clockwise 
# 2 --> Counterclockwise 
def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
  
  	# colinear 
    if (val == 0):
    	return 0  			

   	# clock or counterclock wise 
    if (val > 0):
    	return 1
    else:
    	return 2

def is_intersect(p1, q1, p2, q2):
	# Find the four orientations needed for general and special cases 
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
  
    # General case 
    if (o1 != o2 and o3 != o4):
        return True 
  
    # Special Cases 
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1 
    if (o1 == 0 and onSegment(p1, p2, q1)):
    	return True
  
    # p1, q1 and p2 are colinear and q2 lies on segment p1q1 
    if (o2 == 0 and onSegment(p1, q2, q1)):
    	return True
  
    # p2, q2 and p1 are colinear and p1 lies on segment p2q2 
    if (o3 == 0 and onSegment(p2, p1, q2)):
    	return True 
  
    # p2, q2 and q1 are colinear and q1 lies on segment p2q2 
    if (o4 == 0 and onSegment(p2, q1, q2)):
    	return True
  
    return False # Doesn't fall in any of the above cases

def is_point_in_polygon(polygon, point):
	# Create a point for line segment from p to infinite 
	extreme = [point[0], 1e9]

	# Count intersections of the above line with sides of polygon 
	count = 0
	i = 0

	while True:
		j = (i+1) % len(polygon)

		# Check if the line segment from 'p' to 'extreme' intersects 
        # with the line segment from 'polygon[i]' to 'polygon[j]'
		if is_intersect(polygon[i], polygon[j], point, extreme):
			# If the point 'p' is colinear with line segment 'i-j', 
			# then check if it lies on segment. If it lies, return true, 
			# otherwise false 
			if orientation(polygon[i], point, polygon[j])==0:
				return onSegment(polygon[i], point, polygon[j])
			count = count + 1

		i = j
		if i==0:
			break
	
	return count % 2 == 1

# use this function to check if a bounding box is inside the polygon 
def is_bounding_box_intersect(bounding_box, polygon):
	for i in range(len(bounding_box)):
		if is_point_in_polygon(polygon, bounding_box[i]):
			return True
	return False

def check_bbox_intersect_polygon(polygon, bbox):
	"""

	Args:
	polygon: List of points (x,y)
	bbox: A tuple (xmin, ymin, xmax, ymax)

	Returns:
	True if the bbox intersect the polygon
	"""
	x1, y1, x2, y2 = bbox
	bb = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
	return is_bounding_box_intersect(bb, polygon)


def cosin_similarity(a2d, b2d, debug=False):
	a = np.array((a2d[1][0] - a2d[0][0], a2d[1][1] - a2d[0][1])).astype(float)
	b = np.array((b2d[1][0] - b2d[0][0], b2d[1][1] - b2d[0][1])).astype(float)
	tmp1 = np.dot(a, b)
	tmp2 = (np.linalg.norm(a)*np.linalg.norm(b*1.0))
	if debug:
		print(a, b, tmp1, tmp2)
	return tmp1/tmp2