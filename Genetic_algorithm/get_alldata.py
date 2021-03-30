# borrowed idea from: http://www.randalolson.com/2015/03/10/computing-the-optimal-road-trip-across-europe/#ixzz3UlIoHfKZ
import pandas as pd
import numpy as np
import time
from itertools import combinations
import math

import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

# Requirement
# pip install tqdm
from tqdm import tqdm

# Requirement
# pip install googlemaps

# Enable the "Google Maps Distance Matrix API" on your Google account. Google explains how to do that here.
# https://github.com/googlemaps/google-maps-services-python#api-keys
# https://developers.google.com/maps/documentation/distance-matrix/intro?hl=th

import googlemaps

HEAD_COLUMNS = ["waypoint1", "waypoint2", "distance_m", "duration_s"]

def get_waypoints(num_province=10, file_tosave="my-waypoints-dist-dur.csv"):	
	df_waypoints = pd.read_csv('province_thai.csv', usecols=['province'], encoding="utf-8")
	#df_sample = df_waypoints.sample(num_province)
	df_sample = df_waypoints[0:num_province]
	all_waypoints = df_sample.values.ravel()
	assert len(all_waypoints) == num_province

	all_row = [ [waypoint1, waypoint2] for (waypoint1, waypoint2) in combinations(all_waypoints, 2)]
	# total of (waypoint1, waypoint2) = n!/(2*(n-2)!)
	assert len(all_row) == math.factorial(num_province)/ (2* math.factorial(num_province-2))
	print("all waypoint:", len(all_row))
	#df = pd.DataFrame(data=all_row, columns=["waypoint1", "waypoint2"])
	#df.to_csv("all_waypoint.csv", index=False,)
	
	# call Google Map API
	#gmaps = googlemaps.Client(key="AIzaSyCcwCIrvrb8WtYe4oCI8-AH3vHMGCzv8Nc")
	gmaps = googlemaps.Client(key="AIzaSyAK3RgqSLy1toc4lkh2JVFQ5ipuRB106vU")

	all_data = []
	for i in tqdm(range(len(all_row)) ,ascii=True, desc='get distance and duration'):
		waypoint1, waypoint2 = all_row[i]
		try:
			route = gmaps.distance_matrix(origins=[waypoint1],
										destinations=[waypoint2],
										mode="driving", # Change this to "walking" for walking directions,
														# "bicycling" for biking directions, etc.										
										language="Thai", # Change this to "English", etc
										units="metric")
			# "distance" is in meters
			distance = route["rows"][0]["elements"][0]["distance"]["value"]
			# "duration" is in seconds
			duration = route["rows"][0]["elements"][0]["duration"]["value"]

			all_data.append( [waypoint1, waypoint2 , distance, duration] )
			time.sleep(1)	# waiting 1 seconds
		except Exception as e:
			logging.error("Error with finding the route between %s and %s." % (waypoint1, waypoint2))
			file = open('error.txt','a')
			file.write("Error with finding the route between %s and %s.\n" % (waypoint1, waypoint2))
			file.close()
	
	df = pd.DataFrame(all_data, columns=HEAD_COLUMNS)	
	#assert len(df) == len(all_row)		
	df.to_csv(file_tosave, index=False, encoding="utf-8")

# limitations of Google API, the maximum allowed waypoints is 23 plus the origin and destination.
# routing possibility 20! = 2432902008176640000
# all waypoiint = 20!/(2* (20-2)!) = 190
get_waypoints(num_province=20)
logging.info("=================Finish==================")
