"""
Very NON-ROBUST way of fixing mesh files
so that IGL will read them.

THIS IS NOT READY FOR GENERAL USE. USE WITH 
EXTREME CARE!
"""

import sys

filename = sys.argv[1]

content = ""
with open(filename) as f:
    edgesFound = False
    tetrahedraFound = False
    for line in f:
        
        line = line.strip()
        if line == "MeshVersionFormatted 2":
            line = "MeshVersionFormatted 1"
        if line == "Tetrahedra":
            tetrahedraFound = True
            
        if edgesFound and line == "Triangles":
            edgesFound = False
        if not tetrahedraFound and line == "End":
            content +="Tetrahedra\n0\nEnd\n"
            break
        if line == "Edges":
            edgesFound = True
        if not edgesFound:
            content += line + "\n"
with open(filename, "w") as f:
    f.write(content)
        
