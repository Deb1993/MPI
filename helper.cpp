/* 
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */
//---
#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>
#include "cblock.h"
#include "arblock.h"

using namespace std;

#ifdef _MPI_
#include <mpi.h>
#endif

#define E_prev(ii,jj) E_prev[(ii)*(n+2)+jj]
#define R(ii,jj) R[(ii)*(n+2)+jj]
#define E(ii,jj) E[(ii)*(n+2)+jj]
#define E_prev_copy(ii,jj) E_prev_copy[(ii)*(n+2)+jj]
#define R_copy(ii,jj) R_copy[(ii)*(n+2)+jj]
#define E_copy(ii,jj) E_copy[(ii)*(n+2)+jj]

extern control_block cb;

void printMat(const char mesg[], double *E, int m, int n);
array_chunk ar;


//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init (double *E,double *E_prev,double *R,int m,int n){
	int i;
	int nprocs,myrank;

	for (i=0; i < (m+2)*(n+2); i++)
		E_prev[i] = R[i] = 0;

	for (i = (n+2); i < (m+1)*(n+2); i++) {
		int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

		// Need to compute (n+1)/2 rather than n/2 to work with odd numbers
		if(colIndex == 0 || colIndex == (n+1) || colIndex < ((n+1)/2+1))
			continue;

		E_prev[i] = 1.0;
	}

	for (i = 0; i < (m+2)*(n+2); i++) {
		int rowIndex = i / (n+2);		// gives the current row number in 2D array representation
		int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

		// Need to compute (m+1)/2 rather than m/2 to work with odd numbers
		if(colIndex == 0 || colIndex == (n+1) || rowIndex < ((m+1)/2+1))
			continue;

		R[i] = 1.0;
	}
	// We only print the meshes if they are small enough
#if 0
	printMat("E_prev",E_prev,m,n);
	printMat("R",R,m,n);
#endif

#ifdef _MPI_
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	double *E_copy, *E_prev_copy, *R_copy;
		
	int rows1,cols1;
	rows1 = n/cb.py;
	cols1 = n/cb.px;
	int incr_row1 = n%(cb.py);
	int incr_col1 = n%(cb.px);
	if(myrank/cb.px < incr_row1) {
		rows1++;
	}
	if(myrank%cb.px < incr_col1) {
		cols1++;
	}
	ar.m = rows1 + 2;
	ar.n = cols1 + 2;

	ar.E_prev = (double*)malloc(ar.m*ar.n*sizeof(double)); 
	ar.R = (double*)malloc(ar.m*ar.n*sizeof(double)); 
	ar.E = (double*)malloc(ar.m*ar.n*sizeof(double)); 

	//cout<<"my rank = "<<myrank<<endl;
	//cout<<"m = "<<ar.m-2<<" "<<"n = "<<ar.n-2<<endl;
	//if(myrank == 0)
	//printMat("E_prev",E_prev,n,n);


	int rows,cols,incr_row,incr_col,incr_px,incr_py;
	E_copy = E + (n+2) + 1;
	E_prev_copy = E_prev + (n+2) + 1;
	R_copy = R + (n+2) + 1;
	incr_px = cb.px;
	incr_py = cb.py;
	rows = n/cb.py;
	cols = n/cb.px;
	incr_row = n%(cb.py);
	incr_col = n%(cb.px);
	incr_px--;
	if(incr_row > cb.py - incr_py) {
		rows++;
	}
	if(incr_col > cb.px - incr_px) {
		cols++;
	}
	
	if(myrank == 0) {
	//cout<<"my rank = "<<myrank<<endl;
	//cout<<"m = "<<rows1<<" "<<"n = "<<cols1<<endl;
			for(int ii = 0 ; ii < rows1 ; ii+=1) {
				for(int jj = 0 ; jj < cols1 ; jj+=1) {
					//cout << "ii = " << ii << " " << "jj = " << jj <<endl;
					ar.E_prev[(ii+1)*ar.n+jj+1] = E_prev_copy(ii,jj);
					ar.E[(ii+1)*ar.n+jj+1] = E_copy(ii,jj);
					ar.R[(ii+1)*ar.n+jj+1] = R_copy(ii,jj);
				}
			}
	//cout<<"Rank 0 = "<<myrank<<endl;
	//printMat("ar.E_prev",ar.E_prev,rows,cols);			
	//printMat("ar.E",ar.E,rows,cols);			
	//printMat("ar.R",ar.R,rows,cols);			
	}

	if(incr_px != 0) {
	E_prev = E_prev_copy + cols;
	R = R_copy + cols;
	E = E_copy + cols;
	}
	else { 
	incr_py--;
	if(incr_py != 0) {
		E_copy = E_copy + rows*(n+2);
		E_prev_copy = E_prev_copy + rows*(n+2);
		R_copy = R_copy + rows*(n+2);
		E_prev = E_prev_copy;
		R = R_copy;
		E = E_copy;
		}
	else {
		return;
		}	
	}

	
      if(myrank == 0) {
		
		for( int rank = 1 ; rank < nprocs ; rank++) {
			if(rank != 1) {
				//cout<<"Rank Now = "<<rank<<endl;
				//cout<<"Rows Now = "<<rows<<"Cols now = "<<cols<<endl; 
				incr_px--;
				//cout << "incr_px = "<<incr_px<<endl;
				if(incr_px != 0) {
				E_prev = E_prev + cols;
				R = R + cols;
				E = E + cols;
				}
				else {
				incr_px = cb.px; 
				incr_py--;
				//cout << "incr_py = "<<incr_py<<endl;
				if(incr_py != 0) {
					E_copy = E_copy + rows*(n+2);
					E_prev_copy = E_prev_copy + rows*(n+2);
					R_copy = R_copy + rows*(n+2);
					E_prev = E_prev_copy;
					R = R_copy;
					E = E_copy;
					}
				else {
					//cout << "Break"<<endl;
					break;
					}	
				}
			}
			rows = n/cb.py;
			cols = n/cb.px;
			if(incr_row > cb.py - incr_py) {
				rows++;
				//incr_row--;
			}
			if(incr_col > cb.px - incr_px) {
				cols++;
				//incr_col--;
			}
			//cout<<"Rank while sending = "<<rank<<endl;
			//cout<<"my rank = "<<rank<<endl;
			//if(rank == 3)
			//cout<<"m = "<<rows<<" "<<"n = "<<cols<<endl;
			double* buffer_E_prev = (double*)malloc(rows*cols*sizeof(double));	
			double* buffer_R = (double*)malloc(rows*cols*sizeof(double));	
			double* buffer_E = (double*)malloc(rows*cols*sizeof(double));	
			for(int ii = 0 ; ii < rows ; ii+=1) {
				for(int jj = 0 ; jj < cols ; jj+=1) {
					//cout << "ii = " << ii << " " << "jj = " << jj <<endl;
					buffer_E_prev[ii*cols+jj] = E_prev(ii,jj);
					buffer_R[ii*cols+jj] = R(ii,jj);
					buffer_E[ii*cols+jj] = E(ii,jj);
					//if(rank == 3 ) {
					//cout << "buffer_E_prev = "<<buffer_E_prev[ii*cols+jj]<<endl;
					//cout << "E_prev small = "<<E_prev(ii,jj)<<endl;
					//}
				}
			}
			//cout<< "MPI_SEND"<<endl;
			//cout<< "Rows while sending = "<<rows<<"Cols while sending = "<<cols<<endl;
      		MPI_Send(buffer_E_prev,rows*cols,MPI_DOUBLE,rank,0,MPI_COMM_WORLD);
      		MPI_Send(buffer_R,rows*cols,MPI_DOUBLE,rank,1,MPI_COMM_WORLD);
      		MPI_Send(buffer_E,rows*cols,MPI_DOUBLE,rank,2,MPI_COMM_WORLD);
      	//cout<< "rank = "<<rank<<endl;
		}
      }
      else {
		double* buffer_E_prev_tmp = (double*)malloc(rows1*cols1*sizeof(double));	
		double* buffer_R_tmp = (double*)malloc(rows1*cols1*sizeof(double));	
		double* buffer_E_tmp = (double*)malloc(rows1*cols1*sizeof(double));	
		MPI_Recv(buffer_E_prev_tmp,rows1*cols1,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		MPI_Recv(buffer_R_tmp,rows1*cols1,MPI_DOUBLE,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		MPI_Recv(buffer_E_tmp,rows1*cols1,MPI_DOUBLE,0,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

			for(int ii = 0 ; ii < rows1 ; ii+=1) {
				for(int jj = 0 ; jj < cols1 ; jj+=1) {
					//cout << "ii = " << ii << " " << "jj = " << jj <<endl;
					ar.E_prev[(ii+1)*ar.n+jj+1] = buffer_E_prev_tmp[ii*cols1+jj];
					ar.R[(ii+1)*ar.n+jj+1] = buffer_R_tmp[ii*cols1+jj];
					ar.E[(ii+1)*ar.n+jj+1] = buffer_E_tmp[ii*cols1+jj];
}
	}	

					//if(myrank == 1) {
					//cout << "buffer_E_prev_tmp = "<<buffer_E_prev_tmp[ii*cols+jj]<<endl;
					//cout<<"rows = "<<rows1<<"cols = "<<cols1<<endl;
					//printMat("ar.E_prev",ar.E_prev,rows1,cols1);			
					//printMat("ar.E",ar.E,rows1,cols1);			
					//printMat("ar.R",ar.R,rows1,cols1);			
					//cout << "E_prev = "<<ar.E_prev[(ii+1)*ar.n+jj+1]<<endl;
				//}
			}
#else
	ar.m = m + 2;
	ar.n = n + 2;

	ar.E_prev = (double*)malloc(ar.m*ar.n*sizeof(double)); 
	ar.R = (double*)malloc(ar.m*ar.n*sizeof(double)); 
	ar.E = (double*)malloc(ar.m*ar.n*sizeof(double)); 

	ar.E_prev = E_prev;
	ar.E = E;
	ar.R = R;
#endif

//#ifdef _MPI_
//MPI_Finalize();
//#endif

}

double *alloc1D(int m,int n){
	int nx=n, ny=m;
	double *E;
	// Ensures that allocatdd memory is aligned on a 16 byte boundary
	assert(E= (double*) memalign(16, sizeof(double)*nx*ny) );
	return(E);
}


void printMat(const char mesg[], double *E, int m, int n){
	int i;
#if 0
	if (m>8)
		return;
#else
	if (m>34)
		return;
#endif
	printf("%s\n",mesg);
	for (i=0; i < (m+2)*(n+2); i++){
		int rowIndex = i / (n+2);
		int colIndex = i % (n+2);
		if ((colIndex>0) && (colIndex<n+1))
			if ((rowIndex > 0) && (rowIndex < m+1))
				printf("%6.3f ", E[i]);
		if (colIndex == n+1)
			printf("\n");
	}
}
