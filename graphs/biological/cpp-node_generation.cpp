#include <stdio.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#define IB_X 2
#define IB_Y 1
#define IB_Z 0


static long nentries;
static long sheet_size;
static long row_size;



static double threshold = 0.5;


static void IndexToIndicies(long iv, long &ix, long &iy, long &iz)
{
    iz = iv / sheet_size;
    iy = (iv - iz * sheet_size) / row_size;
    ix = iv % row_size;
}



static long IndiciesToIndex(long ix, long iy, long iz)
{
    return iz * sheet_size + iy * row_size + ix;
}



struct Singleton {
    Singleton(void) { xy_indices = NULL; nvoxels = 0; iz = -1; }
    Singleton(long nelements) { xy_indices = new long[nelements]; nvoxels = 0; iz = -1; }
    ~Singleton() {  }

    long *xy_indices;
    long nvoxels;
    long iz;
};



long *CppFindZSingletons(long *segmentation, long grid_size[3])
{
    // set useful global variables
    nentries = grid_size[IB_Z] * grid_size[IB_Y] * grid_size[IB_X];
    sheet_size = grid_size[IB_Y] * grid_size[IB_X];
    row_size = grid_size[IB_X];

    // get the maximum label
    long maximum_label = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (segmentation[iv] > maximum_label) maximum_label = segmentation[iv];
    }
    ++maximum_label;

    // count the number of voxels per segment
    long *nvoxels_per_segment = new long[maximum_label];
    for (long iv = 0; iv < maximum_label; ++iv)
        nvoxels_per_segment[iv] = 0;

    std::unordered_set<long> *zslices_per_label = new std::unordered_set<long>[maximum_label];
    for (long is = 0; is < maximum_label; ++is) {
        zslices_per_label[is] = std::unordered_set<long>();
    }

    // add each corresponding z slice for each segment
    for (long iv = 0; iv < nentries; ++iv) {
        long ix, iy, iz;
        IndexToIndicies(iv, ix, iy, iz);

        long segment = segmentation[iv];

        zslices_per_label[segment].insert(iz);

        nvoxels_per_segment[segment] += 1;
    }

    // count the number of segments that have only one z slice
    bool *is_singleton = new bool[maximum_label];
    for (long is = 0; is < maximum_label; ++is) {
        if (zslices_per_label[is].size() == 1) is_singleton[is] = true;
    }

    // free memory  
    delete[] zslices_per_label;

    std::unordered_map<long, long> voxel_matches = std::unordered_map<long, long>();

    for (long iz = 0; iz < grid_size[IB_Z] - 1; ++iz) {
        for (long iy = 0; iy < grid_size[IB_Y]; ++iy) {
            for (long ix = 0; ix <grid_size[IB_X]; ++ix) {
                long segment = segmentation[IndiciesToIndex(ix, iy, iz)];
                long neighbor_segment = segmentation[IndiciesToIndex(ix, iy, iz + 1)];

                // skip over non singleton elements
                if (not is_singleton[segment] or not is_singleton[neighbor_segment]) continue;

                if (neighbor_segment < segment) {
                    long tmp = neighbor_segment;
                    neighbor_segment = segment;
                    segment = tmp;
                }

                voxel_matches[segment * maximum_label + neighbor_segment]++;
            }
        }
    }

    std::vector<long> pairs = std::vector<long>();

    std::unordered_map<long, long>::iterator it;
    for (it = voxel_matches.begin(); it != voxel_matches.end(); ++it) {
        long indicies = it->first;
        long first_index = indicies / maximum_label;
        long second_index = indicies % maximum_label;

        long nvoxel_matches = it->second;

        long total_voxels = nvoxels_per_segment[first_index] + nvoxels_per_segment[second_index] - nvoxel_matches;

        double overlap = nvoxel_matches / (double)(total_voxels);

        if (overlap > threshold) {
            pairs.push_back(first_index);
            pairs.push_back(second_index);
        }
    }

    long *matches = new long[pairs.size() + 1];
    matches[0] = pairs.size();
    for (long iv = 0; iv < pairs.size(); ++iv) {
        matches[iv + 1] = pairs[iv];
    }

    delete[] nvoxels_per_segment;

    return matches;
}