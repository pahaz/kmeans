#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>

using namespace std;

typedef vector<double> Point;

// Pretty good uniform distribution on [0..1]
double RandUniform01() {
    int rand1 = rand();
    double div = RAND_MAX;
    while (rand1 == RAND_MAX) {
        rand1 = rand();
    }
    return ((double)rand1 + rand() / div) / div;
}

// Box-Muller transform
double RandNormal(double mean, double sigma) {
    double x, y, r;
    do {  
        x = 2 * RandUniform01() - 1;
        y = 2 * RandUniform01() - 1;
        r = x * x + y * y;
    } while (r == 0.0 || r > 1.0);
    return sigma * x * sqrt(-2 * log(r) / r) + mean;
}

struct ClusterParams {
    Point mean;
    double var;
};

Point RandomPointGauss(ClusterParams params) {
    size_t dimensions = params.mean.size();
    Point coord(dimensions);
    for (size_t i = 0; i < dimensions; ++i) {
        coord[i] = RandNormal(params.mean[i], params.var);
    }
    return coord;
}

Point RandomPointUniform(size_t dimensions, double space_size) {
    Point coord(dimensions);
    for (size_t i = 0; i < dimensions; ++i) {
        coord[i] = RandUniform01() * space_size;
    }
    return coord;
}

void WritePoint(Point point, ofstream& out) {
    for (size_t i = 0; i < point.size()-1; ++i) {
        out << point[i] << " ";
    }
    out << point[point.size()-1] << endl;
}

int main(int argc , char** argv) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " dimensions number_of_points number_of_clusters output_file" << endl;
        return 1;
    }
    size_t dimensions = atoi(argv[1]);
    size_t number_of_points = atoi(argv[2]);
    size_t number_of_clusters = atoi(argv[3]); // 0..32767
    
    string output_file = argv[4];
    ofstream output(output_file);
    if(!output.is_open()) {
        cerr << "Error: output file could not be opened" << endl;
        return 1;
    }

    // advanced params
    double space_size = 100;
    double cluster_size = 5;
    int random_point_pct = 20;

    srand((unsigned)time(0));

    vector<ClusterParams> cluster_params(number_of_clusters);
    for (size_t i = 0; i < number_of_clusters; ++i) {
        cluster_params[i].mean = RandomPointUniform(dimensions, space_size);
        cluster_params[i].var = cluster_size / 2 + RandUniform01() * cluster_size;
    }

    output << number_of_points << " " << dimensions << endl;

    for (size_t i = 0; i < number_of_points; ++i) {
        bool in_cluster = (rand() % 100) >= random_point_pct;
        if (in_cluster) {
            size_t cluster = rand() % number_of_clusters;
            WritePoint(RandomPointGauss(cluster_params[cluster]), output);
        } else {
            WritePoint(RandomPointUniform(dimensions, space_size), output);
        }
    }

    output.close();  
    return 0;
}