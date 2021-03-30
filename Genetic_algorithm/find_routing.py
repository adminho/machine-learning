import pandas as pd
import numpy as np
import webbrowser
import os.path

from optimal_road_trip import run_genetic_algorithm

if __name__ == "__main__":
	#all_route = run_genetic_algorithm(generations=5000, population_size=100)
	all_route = run_genetic_algorithm(generations=500, population_size=100)
	df = pd.DataFrame(data=all_route)
	df.to_csv("debug_all_route.csv")

	def list2String(list_data):
		return "['" + "','".join(map(str, list_data )) + "']"	

	f = open("show_routing.template","r" ,encoding="utf-8")
	html_template = f.read()
	f.close()

	route_str ='['
	for i in range(0, len(all_route)-1):
		route_str += list2String(all_route[i]) +','
	route_str += list2String(all_route[len(all_route)-1]) + ']'

	start_str = list2String(df[0].values)
	last_col = np.shape(all_route)[1]- 1
	end_str= list2String(df[last_col].values)

	html_code = html_template % (start_str, end_str, route_str)

	file_name = "show_routing.html"
	f = open(file_name, "w", encoding="utf-8")
	f.write(html_code)
	f.close()

	# If your system set IE is default, it may be don't show html on your webbrowser		
	webbrowser.open('file://' + os.path.realpath(file_name))
