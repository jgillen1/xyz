
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"
#include <vector>
#define NUM_THREADS 256

using namespace std;

extern double size;
//
//  benchmarking program
//

#define MAX 100

__global__ void clear_gpu(int *numInBlock, int row) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= row*row) return;

        numInBlock[tid] = 0;


}


__global__ void bin_gpu(particle_t *p,particle_t **bins,int n, int row, int *numInBlock, int bin_size) {

        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= n) return;
	//printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);

        int idx = floor(p[tid].x/0.01) + row *floor(p[tid].y/0.01);
	//printf("asd");

        int num = atomicAdd(&numInBlock[idx],1);
        bins[idx*MAX + num] = p + tid;

	//printf("Hello from block %d, thread %d  Number of particles in bin %d is %d\n", blockIdx.x, threadIdx.x,idx,numInBlock[idx]);


	//printf("Number of particels in bin 10\n");


}






__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor) {
        double dx = neighbor.x - particle.x;
        double dy = neighbor.y - particle.y;
        double r2 = dx * dx + dy * dy;
        if (r2 > cutoff*cutoff)
                return;
        //r2 = fmax( r2, min_r*min_r );
        r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
        double r = sqrt(r2);

        //
        //  very simple short-range repulsive force
        //
        double coef = (1 - cutoff / r) / r2 / mass;
        particle.ax += coef * dx;
        particle.ay += coef * dy;

}

__global__ void compute_forces_gpu(particle_t * particles, int n, particle_t **bins, int *numInBlock, int row) {
        // Get thread (particle) ID
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= n) return;

        //particles[tid].ax = particles[tid].ay = 0;
        //for (int j = 0; j < n; j++)
        //      apply_force_gpu(particles[tid], particles[j]);
				/*
        int chunk = blockDim.x * gridDim.x;
        for (int j = 0; j < row; j+=chunk) {
                for (int i = 0; i < row; i++) {
                        for(int k = 0; k < numInBlock[i + j*row]; k++) {
                                bins[(i + j*row)*n + k]->ax = 0;
                                bins[(i + j*row)*n + k]->ay = 0;
                                for (int ii = -1; ii < 2; ii++) {
                                        int idx = i + ii;
                                        if (idx >=0 && idx < row) {
                                                for (int jj = -1; jj < 2; jj++) {
                                                        int idy = j + jj;
                                                        if(idy >=0 && idy < row ) {
                                                                for (int kk = 0; kk < numInBlock[idx + row*idy]; kk++ ) {
                                                                        apply_force_gpu(*(bins[(i + j*row)*n + k]), *(bins[n*(idx + idy*row)+kk]));
                                                                }
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }

*/


particle_t  p = particles[tid];

int binx = floor(p.x/0.01);
int biny = floor(p.y/0.01);

particles[tid].ax = 0;
particles[tid].ay = 0;

/*
int li = -1;
int hi = 1;
int lj = -1;
int hj = 1;

if(binx + biny < row) lj = 0;
if((binx + biny) % row == 0) li = 0;
if((binx + biny) % row == (row-1)) hi = 0;
if((binx + biny) >= row * (row-1)) hj = 0;




for (int ii = li; ii <= hi; ii++) {
	for (int jj = lj; jj <= hj; jj++) {
		int nbin = binx + biny + ii + jj*row;
		for (int kk = 0; kk < numInBlock[nbin]; kk++ ) {
                 //printf("here\n" );
		apply_force_gpu(p, *(bins[n*nbin+kk]));
		}
	}
}


particles[tid].ax = p.ax;
particles[tid].ay = p.ay;

*/


for (int ii = -1; ii < 2; ii++) {
	int idx = binx + ii;
	if (idx >=0 && idx < row) {
		for (int jj = -1; jj < 2; jj++) {
			int idy = biny + jj;
			if(idy >=0 && idy < row ) {
				for (int kk = 0; kk < numInBlock[idx + idy*row]; kk++ ) {
                                   //printf("here\n" );
				apply_force_gpu(particles[tid], *(bins[MAX*(idx + idy*row)+kk]));
				}
			}
		}
	}
}
}

__global__ void move_gpu(particle_t * particles, int n, double size) {

        // Get thread (particle) ID
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= n) return;

        particle_t * p = &particles[tid];
        //
        //  slightly simplified Velocity Verlet integration
        //  conserves energy better than explicit Euler method
        //
        p->vx += p->ax * dt;
        p->vy += p->ay * dt;
        p->x += p->vx * dt;
        p->y += p->vy * dt;

        //
        //  bounce from walls
        //
        while (p->x < 0 || p->x > size) {
                p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
                p->vx = -(p->vx);
        }
        while (p->y < 0 || p->y > size) {
                p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
                p->vy = -(p->vy);
        }

}



int main(int argc, char **argv) {
        // This takes a few seconds to initialize the runtime
        cudaThreadSynchronize();

        if (find_option(argc, argv, "-h") >= 0) {
                printf("Options:\n");
                printf("-h to see this help\n");
                printf("-n <int> to set the number of particles\n");
                printf("-o <filename> to specify the output file name\n");
                return 0;
        }

        int n = read_int(argc, argv, "-n", 1000);

        char *savename = read_string(argc, argv, "-o", NULL);

        char *sumname = read_string( argc, argv, "-s", NULL );
         FILE *fsum = sumname ? fopen(sumname,"a") : NULL;

        FILE *fsave = savename ? fopen(savename, "w") : NULL;
        particle_t *particles = (particle_t*)malloc(n * sizeof(particle_t));

        // GPU particle data structure
        particle_t * d_particles;
        cudaMalloc((void **)&d_particles, n * sizeof(particle_t));




        // number of bins per row
        // row * row is total number of bins
				double sz = sqrt(0.0005 * n);
        int row = ceil(sz/0.01);

        particle_t ** bins;
        cudaMalloc((void **)&bins, row*row*MAX *sizeof(particle_t*));

        //for (int i = 0; i < row*row; i++) {
        //      cudaMalloc((void **)&bins[i], n*sizeof(particle_t*));
        //}
        int * numInBlock;
        cudaMalloc((void **)&numInBlock,row*row*sizeof(int));
        set_size(n);
				int bin_size = size / row;
        init_particles(n, particles);

        cudaThreadSynchronize();
        double copy_time = read_timer();

        // Copy the particles to the GPU
        cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

        cudaThreadSynchronize();
        copy_time = read_timer() - copy_time;

        //
        //  simulate a number of time steps
        //
        cudaThreadSynchronize();
        double simulation_time = read_timer();


        //vector<particle_t*> *binTest = new vector<particle_t*>[row*row];





        for (int step = 0; step < NSTEPS; step++) {
                //
                //  compute forces
                //









                int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
              //  clear_gpu<<< blks, NUM_THREADS >>> (numInBlock,row);
							cudaMemset(numInBlock,0,row*row*sizeof(int));
                bin_gpu <<< blks, NUM_THREADS >>> (d_particles, bins, n, row, numInBlock,bin_size);

                compute_forces_gpu << < blks, NUM_THREADS >> > (d_particles, n, bins, numInBlock, row);


                //
                //  move particles
                //
                move_gpu << < blks, NUM_THREADS >> > (d_particles, n, size);
              

                //
                //  save if necessary
                //
              /*  if (fsave && (step%SAVEFREQ) == 0) {
                        // Copy the particles back to the CPU
                        cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
                        save(fsave, n, particles);
                }*/
        }
        cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
        if(fsave)
          save(fsave, n, particles);
        cudaThreadSynchronize();
        simulation_time = read_timer() - simulation_time;

        printf("CPU-GPU copy time = %g seconds\n", copy_time);
        printf("n = %d, simulation time = %g seconds\n", n, simulation_time);

        free(particles);
        cudaFree(d_particles);
        cudaFree(bins);
        cudaFree(numInBlock);

        if (fsum)
fprintf(fsum,"%d %lf \n",n,simulation_time);

  if (fsum)
fclose( fsum );
        if (fsave)
                fclose(fsave);

        return 0;
}
