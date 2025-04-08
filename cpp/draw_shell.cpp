#include <vcg/complex/algorithms/create/platonic.h>
#include <vcg/complex/algorithms/update/position.h>
#include <vcg/complex/append.h>
#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export.h>
#include <cmath>
#include <iostream>

class MyVertex; class MyEdge; class MyFace;
struct MyUsedTypes : public vcg::UsedTypes<vcg::Use<MyVertex>   ::AsVertexType,
                                           vcg::Use<MyEdge>     ::AsEdgeType,
                                           vcg::Use<MyFace>     ::AsFaceType>{};

class MyVertex  : public vcg::Vertex< MyUsedTypes, vcg::vertex::Coord3d, vcg::vertex::Normal3d, vcg::vertex::BitFlags >{};
class MyFace    : public vcg::Face<   MyUsedTypes, vcg::face::FFAdj,  vcg::face::VertexRef, vcg::face::BitFlags > {};
class MyEdge    : public vcg::Edge<   MyUsedTypes, vcg::edge::EVAdj> {};

class MyMesh    : public vcg::tri::TriMesh< std::vector<MyVertex>, std::vector<MyFace> , std::vector<MyEdge>  > {};

// Draws a gridshell node as a sphere, given the center and the radius.
void draw_node(MyMesh &m, vcg::Point3d center, double radius)
{
    vcg::tri::Sphere(m, 2);
    vcg::tri::UpdatePosition<MyMesh>::Scale(m, radius);
    vcg::tri::UpdatePosition<MyMesh>::Translate(m, center);
}

// Draws a gridshell beam as a cylinder, given the origin and the end.
void draw_beam(MyMesh &m, vcg::Point3d origin, vcg::Point3d end, double radius)
{
    vcg::tri::OrientedCylinder(m, origin, end, radius, false, 6, 1);
}

int main(int argc, char** argv){
    MyMesh source, target, target_aux;

    if (argc < 3) 
    {
        printf("Usage draw_shell <filename.ply> <outputname.ply>.\n");
        return -1;
    }

    if (vcg::tri::io::ImporterPLY<MyMesh>::Open(source, argv[1]) != 0) 
    {
        printf("Error reading file %s.\n", argv[1]);
        return -1;
    }
    vcg::tri::UpdateTopology<MyMesh>::AllocateEdge(source);

    // double beam_cross_section_area = 1e-3;
    // double radius = sqrt(beam_cross_section_area / M_PI);
    double radius = 0.08;

    if (argc >= 4)
    {
        radius = atof(argv[3]);
    }

    int i = 0;
    std::vector<bool> visited(source.VN(), false);
    for (i = 0; i < source.EN(); i++)
    {
        MyMesh::EdgePointer edge_p = &(source.edge[i]);

        int endpoint_0_idx = vcg::tri::Index(source, edge_p->V(0));
        int endpoint_1_idx = vcg::tri::Index(source, edge_p->V(1));

        MyMesh::VertexPointer endpoint_0_p = &(source.vert[endpoint_0_idx]);
        MyMesh::VertexPointer endpoint_1_p = &(source.vert[endpoint_1_idx]);

        vcg::Point3d endpoint_0 = endpoint_0_p->cP();
        vcg::Point3d endpoint_1 = endpoint_1_p->cP();

        // Drawing the beam belonging to the current edge.
        // vcg::Point3d beam_origin = endpoint_0 + 2 * radius * (endpoint_1 - endpoint_0).normalized();
        // vcg::Point3d beam_end = endpoint_1 + 2 * radius * (endpoint_0 - endpoint_1).normalized();
        draw_beam(target_aux, endpoint_0, endpoint_1, radius);
        vcg::tri::Append<MyMesh, MyMesh>::Mesh(target, target_aux);
        target_aux.Clear();

        // Drawing endpoint nodes, if such vertices were not visited already.
        if (! visited[endpoint_0_idx])
        {
            draw_node(target_aux, endpoint_0, 1.5 * radius);
            vcg::tri::Append<MyMesh, MyMesh>::Mesh(target, target_aux);
            target_aux.Clear();
            visited[endpoint_0_idx] = true;
        }
        if (! visited[endpoint_1_idx])
        {
            draw_node(target_aux, endpoint_1, 1.5 * radius);
            vcg::tri::Append<MyMesh, MyMesh>::Mesh(target, target_aux);
            target_aux.Clear();
            visited[endpoint_1_idx] = true;
        }

    }

    vcg::tri::io::ExporterPLY<MyMesh>::Save(target, argv[2]);

    return 0;
}