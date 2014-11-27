#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <omp.h>

using namespace std;

struct timeval start, stop;

typedef vector<double> Point;
typedef vector<Point> Points;


// get random [0..max_value]
unsigned int UniformRandom(unsigned int max_value) {
    unsigned int rnd = ((static_cast<unsigned int>(rand()) % 32768) << 17) |
                       ((static_cast<unsigned int>(rand()) % 32768) << 2) |
                       rand() % 4;
    return ((max_value + 1 == 0) ? rnd : rnd % (max_value + 1));
}

double Distance(const Point& point1, const Point& point2) {
    double distance_sqr = 0;
    size_t dim = point1.size();
    for (size_t i = 0; i < dim; ++i) {
        distance_sqr += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return sqrt(distance_sqr);
}

size_t FindNearestCentroid(const Points& centroids, const Point& point) {
    double min_distance = Distance(point, centroids[0]);
    size_t centroid_index = 0;
    for (size_t i = 1; i < centroids.size(); ++i) {
        double distance = Distance(point, centroids[i]);
        if (distance < min_distance) {
            min_distance = distance;
            centroid_index = i;
        }
    }
    return centroid_index;
}

// Calculates new centroid position as mean of positions of 3 random centroids
Point GetRandomPosition(const Points& centroids) {
    size_t K = centroids.size();
    int c1 = rand() % K;
    int c2 = rand() % K;
    int c3 = rand() % K;
    size_t dim = centroids[0].size();
    Point new_position(dim);
    for (size_t d = 0; d < dim; ++d) {
        new_position[d] = (centroids[c1][d] + centroids[c2][d] + centroids[c3][d]) / 3;
    }
    return new_position;
}

vector<size_t> KMeans(const Points& data, size_t K) {
    size_t data_size = data.size();
    size_t dim = data[0].size();
    vector<size_t> clusters(data_size);

    // Initialize randomly
    Points centroids(K);
    for (size_t i = 0; i < K; ++i) {
        centroids[i] = data[UniformRandom(data_size - 1)];
    }
    
    int nthreads = omp_get_max_threads();
    
    /*
    Each thread calculates new centroids using a private space,
    then thread 0 does an array reduction on them. */
    vector<Point> local_new_cluster_size(nthreads);
    local_new_cluster_size.assign(nthreads, Point(K));
    
    vector<Points> local_new_centroids(nthreads);
    local_new_centroids.assign(nthreads, Points(K));
    for (int i = 0; i < nthreads; i++)
    	local_new_centroids[i].assign(K, Point(dim));
    
    vector<size_t> new_cluster_size(K);
    
    Points new_centroids(K);
    new_centroids.assign(K, Point(dim));
    
    int points_moved;
    do {
    	points_moved = 0;
    	
    	#pragma omp parallel shared(data,clusters,centroids,local_new_centroids,local_new_cluster_size)
		{
			int tid = omp_get_thread_num();
			
			#pragma omp for firstprivate(data_size,K,dim) schedule(static) reduction(+:points_moved)
			for (size_t i = 0; i < data_size; ++i) 
			{
				size_t nearest_cluster = FindNearestCentroid(centroids, data[i]);
				if (clusters[i] != nearest_cluster) 
				{
                	clusters[i] = nearest_cluster;
                	points_moved++;
				}
				
				local_new_cluster_size[tid][nearest_cluster]++;
				for (size_t d = 0; d < dim; ++d)
					local_new_centroids[tid][nearest_cluster][d] += data[i][d];
					
					
            }
		}
		
		/* let the main thread perform the array reduction */
		for (size_t i = 0; i < K; i++) {
			for (int j = 0; j < nthreads; j++) {
				new_cluster_size[i] += local_new_cluster_size[j][i];
				local_new_cluster_size[j][i] = 0.0;
				for (size_t d = 0; d < dim; d++) {
					new_centroids[i][d] += local_new_centroids[j][i][d];
					local_new_centroids[j][i][d] = 0.0;
				}
			}
		}

		/* average the sum and replace old centroids with newCentroids */
		for (size_t i = 0; i < K; ++i) 
		{
            if (new_cluster_size[i] > 1)
            {
                for (size_t d = 0; d < dim; ++d)
                    centroids[i][d] = new_centroids[i][d] / new_cluster_size[i];
            } 
            else 
            {
                centroids[i] = GetRandomPosition(centroids);
            }
        }
        new_cluster_size.assign(K, 0);
        new_centroids.assign(K, Point(dim));
	
    } while (points_moved > 0);

    return clusters;
}

void ReadPoints(Points* data, ifstream& input) {
    size_t data_size;
    size_t dim;
    input >> data_size >> dim;
    data->assign(data_size, Point(dim));
    for (size_t i = 0; i < data_size; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            double coord;
            input >> coord;
            (*data)[i][d] = coord;
        }
    }
}

void WriteOutput(const vector<size_t>& clusters, ofstream& output) {
    for (size_t i = 0; i < clusters.size(); ++i) {
        output << clusters[i] << endl;
    }
}

int main(int argc , char** argv) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << "number_of_clusters in_file out_file" << endl;
        return 1;
    }
    size_t K = atoi(argv[1]);

    char* input_file = argv[2];

    ifstream input;
    input.open(input_file, ifstream::in);
    if(!input) {
        cerr << "Error: open(in_file) error" << endl;
        return 1;
    }
    
    gettimeofday(&start, NULL);

    Points data;
    ReadPoints(&data, input);
    input.close();

    char* output_file = argv[3];
    ofstream output;
    output.open(output_file, ifstream::out);
    if(!output) {
        cerr << "Error: open(out_file) error" << endl;
        return 1;
    }

    srand(228); // for reproducible results

    vector<size_t> clusters = KMeans(data, K);

    WriteOutput(clusters, output);
    output.close();
    
    gettimeofday(&stop, NULL);
    
    double duration = (double)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/1000000.0);
    
    printf("Duration: %f seconds\n", duration);

    return 0;
}