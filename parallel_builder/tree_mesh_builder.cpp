/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  FULL NAME <xkuzni04@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"



TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::decomp(const ParametricScalarField &field, Vec3_t<float> &ofs, size_t grSize)
{
    if ( 1 >= grSize)
    {
        return buildCube(ofs, field);
    }
    
    const Vec3_t<float> m(
        ofs.x * BaseMeshBuilder::mGridResolution + (float(grSize) * BaseMeshBuilder::mGridSize) / 2.0,
        ofs.y * BaseMeshBuilder::mGridResolution + (float(grSize) * BaseMeshBuilder::mGridSize) / 2.0,
        ofs.z * BaseMeshBuilder::mGridResolution + (float(grSize) * BaseMeshBuilder::mGridSize) / 2.0
    ); 
    bool a = evaluateFieldAt(m, field) > BaseMeshBuilder::mIsoLevel + (sqrtf(3.0)/2.0) * (float(grSize) * BaseMeshBuilder::mGridSize);

    if (a) {
        return 0;
    }
    
    unsigned crit = 0; 

    char codeX[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    char codeY[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    char codeZ[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    for (char i = 0; i < 8; i++) {
        #pragma omp task shared(crit)
        {
            Vec3_t<float> offNew(
                ofs.x + codeX[i] * (float)(grSize / 2.0),
                ofs.y + codeY[i] * (float)(grSize / 2.0),
                ofs.z + codeZ[i] * (float)(grSize / 2.0)
            );

            unsigned tc = decomp(field, offNew, grSize/2);
            #pragma omp critical
            crit =  crit + tc;
        }
    }

    // wait for the children tasks to finish
    #pragma omp taskwait
    return crit;


}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    Vec3_t<float> init(0.0, 0.0, 0.0);
    unsigned res = 0;
    #pragma omp parallel
    #pragma omp single
    res = decomp(field, init, BaseMeshBuilder::mGridSize); 

    return res;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        value = std::min(value, distanceSquared);
    }
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical
    mTriangles.push_back(triangle);
}
