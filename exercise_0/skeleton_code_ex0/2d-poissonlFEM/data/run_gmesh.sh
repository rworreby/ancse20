#!/bin/bash



square_file="square.geo"

gmsh -2 -o square_0.mesh $square_file
gmsh -2 -refine -o square_0.mesh square_0.mesh

for i in `seq 1 9`
do
    gmsh -2 -refine -o square_$i.mesh square_$((i-1)).mesh
    
done

for i in `seq 0 9`
do

    python fix_mesh.py square_$i.mesh
done


L_file="Lshape.geo"

gmsh -2 -o Lshape_0.mesh $L_file
gmsh -2 -refine -o Lshape_0.mesh Lshape_0.mesh

for i in `seq 1 9`
do
    gmsh -2 -refine -o Lshape_$i.mesh Lshape_$((i-1)).mesh;
    
done

for i in `seq 0 9`
do
    python fix_mesh.py Lshape_$i.mesh;
    
done
