import collections


regular_angle_descriptors: collections.defaultdict(list) = {
			"r_armpit" 	: [ ["RIGHT", "elbow"],["RIGHT", "shoulder"], ["RIGHT", "hip"]	 	],
			"l_armpit" 	: [ ["LEFT", "elbow"], ["LEFT", "shoulder"],  ["LEFT", "hip"] 	 	],
			"r_elbow" 	: [ ["RIGHT","wrist"], ["RIGHT","elbow"],     ["RIGHT","shoulder"] 	],
			"l_elbow" 	: [ ["LEFT","wrist"],  ["LEFT","elbow"],      ["LEFT","shoulder"]  	],
			"r_knee" 	: [ ["RIGHT","hip"],  ["RIGHT","knee"], 	  ["RIGHT","ankle"]	 	],
			"l_knee" 	: [ ["LEFT","hip"],    ["LEFT","knee"], 	  ["LEFT","ankle"] 	    ]
		}

for key in regular_angle_descriptors:
    print(regular_angle_descriptors[key])