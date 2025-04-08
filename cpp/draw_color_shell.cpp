#include <vcg/complex/algorithms/create/platonic.h>
#include <vcg/complex/algorithms/update/position.h>
#include <vcg/complex/algorithms/update/color.h>
#include <vcg/complex/append.h>
#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export.h>
#include <cmath>
#include <map>
#include <iostream>
#include <string>
#include <string.h>
#include <vector>
#include <fstream>
#include <sstream>

class MyVertex; class MyEdge; class MyFace;
struct MyUsedTypes : public vcg::UsedTypes<vcg::Use<MyVertex>   ::AsVertexType,
                                           vcg::Use<MyEdge>     ::AsEdgeType,
                                           vcg::Use<MyFace>     ::AsFaceType>{};

class MyVertex  : public vcg::Vertex< MyUsedTypes, vcg::vertex::Coord3d, vcg::vertex::Normal3d, vcg::vertex::BitFlags, vcg::vertex::Color4b >{};
class MyFace    : public vcg::Face<   MyUsedTypes, vcg::face::FFAdj,  vcg::face::VertexRef, vcg::face::BitFlags > {};
class MyEdge    : public vcg::Edge<   MyUsedTypes, vcg::edge::EVAdj> {};

class MyMesh    : public vcg::tri::TriMesh< std::vector<MyVertex>, std::vector<MyFace> , std::vector<MyEdge>  > {};

// Reads the content of a .csv file and stores it inside a vector.
int readcsv(std::vector< std::vector<int> > &container, std::string filename)
{
    std::vector<int> row;
    std::string line, word;
 
    std::fstream file (filename, std::ios::in);
    if(file.is_open())
    {
        while(getline(file, line))
        {
            row.clear();
            std::stringstream str(line);
            while(getline(str, word, ','))
                row.push_back(stoi(word));         
            container.push_back(row);
        }
    }
    else 
    {
        std::cout << "Could not open the file." << std::endl;
        return -1;
    }
    return 0;
}

// Builds an hash table mapping the edge index vector to the corresponding color.
void build_edge_to_color_map(std::map< std::vector<int>, vcg::Color4b > &edge_colors, std::vector< std::vector<int> > edges, std::vector< std::vector<int> > colors)
{
    int i;
    for (i = 0; i < edges.size(); i++)
    {
        vcg::Color4b color(colors[i][0], colors[i][1], colors[i][2], colors[i][3]);
        edge_colors[edges[i]] = color;
    }
}

// Draws a gridshell node as a sphere, given the center and the radius.
void draw_node(MyMesh &m, vcg::Point3d center, double radius)
{
    vcg::tri::Sphere(m, 1);
    vcg::tri::UpdatePosition<MyMesh>::Scale(m, radius);
    vcg::tri::UpdatePosition<MyMesh>::Translate(m, center);
    vcg::tri::UpdateColor<MyMesh>::PerVertexConstant(m, vcg::Color4b(64, 64, 64, 255));
}

// Draws a gridshell beam as a cylinder, given the origin and the end.
void draw_beam(MyMesh &m, vcg::Point3d origin, vcg::Point3d end, double radius, vcg::Color4b color)
{
    vcg::tri::OrientedCylinder(m, origin, end, radius, false, 4, 1);
    vcg::tri::UpdateColor<MyMesh>::PerVertexConstant(m, color);
}

int main(int argc, char** argv){
    MyMesh source, target, tmp;
    std::vector< std::vector<int> > edges;
    std::vector< std::vector<int> > colors;
    std::map< std::vector<int>, vcg::Color4b > edge_to_color;

    if (argc < 2) 
    {
        printf("Usage draw_color_shell <filename.ply> <edges.csv> <energy_rgba.csv> <outputname.ply>.\n");
        return -1;
    }

    if (vcg::tri::io::ImporterPLY<MyMesh>::Open(source, argv[1]) != 0) 
    {
        printf("Error reading file %s.\n", argv[1]);
        return -1;   
    }
    vcg::tri::UpdateTopology<MyMesh>::AllocateEdge(source);

    if (readcsv(edges, argv[2]) != 0)
    {
        printf("Error reading file %s.\n", argv[2]);
        return -1;  
    }

    if (readcsv(colors, argv[3]) != 0)
    {
        printf("Error reading file %s.\n", argv[3]);
        return -1; 
    }

    build_edge_to_color_map(edge_to_color, edges, colors);

    double beam_cross_section_area = 1e-3;
    double radius = sqrt(beam_cross_section_area / M_PI);
    radius = 0.12;

    int i = 0;
    std::vector<bool> visited(source.VN(), false);
    for (i = 0; i < source.EN(); i++)
    {
        MyMesh::EdgePointer edge_p = &(source.edge[i]);

        std::vector<int> endpoints_idx = {int(vcg::tri::Index(source, edge_p->V(0))), int(vcg::tri::Index(source, edge_p->V(1)))};
        std::vector< MyMesh::VertexPointer > endpoints_p = {&(source.vert[endpoints_idx[0]]), &(source.vert[endpoints_idx[1]])};
        std::vector< vcg::Point3d > endpoints = {endpoints_p[0]->cP(), endpoints_p[1]->cP()};

        // Drawing the beam belonging to the current edge.
        // vcg::Point3d beam_origin = endpoints[0] + 2 * radius * (endpoints[1] - endpoints[0]).normalized();
        // vcg::Point3d beam_end = endpoints[1] + 2 * radius * (endpoints[0] - endpoints[1]).normalized();
        draw_beam(tmp, endpoints[0], endpoints[1], radius, edge_to_color[endpoints_idx]);
        vcg::tri::Append<MyMesh, MyMesh>::Mesh(target, tmp);
        tmp.Clear();

        // Drawing endpoint nodes, if such vertices were not visited already.
        int j = 0;
        for (j = 0; j < 2; j++)
        {
            if (! visited[endpoints_idx[j]])
            {
                draw_node(tmp, endpoints[j], 1.5 * radius);
                vcg::tri::Append<MyMesh, MyMesh>::Mesh(target, tmp);
                tmp.Clear();
                visited[endpoints_idx[j]] = true;
            }    
        }
    }

    vcg::tri::io::ExporterPLY<MyMesh>::Save(target, argv[4], vcg::tri::io::Mask::IOM_VERTCOLOR);
    std::cout << "Done!" << std::endl;
    return 0;
}