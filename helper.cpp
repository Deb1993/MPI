/* 
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>
#include "cblock.h"

using namespace std;

#ifdef _MPI_
#include <mpi.h>
#endif

#define E_prev(ii,jj) E_prev[(ii)*(n+2)+jj]
#define R(ii,jj) R[(ii)*(n+2)+jj]
#define E(ii,jj) E[(ii)*(n+2)+jj]

#ifdef _MPI_
typedef struct _array_chunk {

	int m,n;
	double *E ;
	double *E_prev;
	double *R;

} array_chunk; 
#endif


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
	//#ifdef _MPI_
	//MPI_Init(&argc,&argv);
	//#endif
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

		//cout << "E_prev matrix all = " << E_prev[i] <<endl;
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
	int rows,cols,incr_row,incr_col,incr_px,incr_py;
	E_copy = E;
	E_prev_copy = E_prev;
	R_copy = R;
	incr_px = cb.px;
	incr_py = cb.py;
	rows = n/cb.py;
	cols = n/cb.px;
	incr_row = n%(cb.py);
	incr_col = n%(cb.px);
	incr_px--;
	if(incr_row > cb.py - incr_py) {
		rows++;
		//incr_row--;
	}
	if(incr_col > cb.px - incr_px) {
		cols++;
		//incr_col--;
	}

	if(incr_px != 0) {
	E_prev = E_prev + cols;
	R = R + cols;
	E = E + cols;
	}
	else { 
	incr_py--;
	if(incr_py != 0) {
		E_copy = E_copy + rows;
		E_prev_copy = E_prev_copy + rows;
		R_copy = R_copy + rows;
		E_prev = E_prev_copy;
		R = R_copy;
		E = E_copy;
		}
	else {
		return;
		}	
	}

	double* buffer_E_prev = (double*)malloc(rows*cols*sizeof(double));	
	double* buffer_R = (double*)malloc(rows*cols*sizeof(double));	
	double* buffer_E = (double*)malloc(rows*cols*sizeof(double));	
	
      if(myrank == 0) {
		for( int rank = 1 ; rank < nprocs ; rank++) {
		//	int rows,cols;
		//	rows = (n+2)/cb.py;
		//	cols = (n+2)/cb.px;
		//	if(incr_row) {
		//		rows++;
		//		incr_row--;
		//	}
		//	if(incr_col) {
		//		cols++;
		//		incr_col--;
		//	}
			if(rank != 1) {
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
					E_copy = E_copy + rows;
					E_prev_copy = E_prev_copy + rows;
					R_copy = R_copy + rows;
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
			
			int rows,cols;
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
			for(int ii = 0 ; ii < rows ; ii+=1) {
				for(int jj = 0 ; jj < cols ; jj+=1) {
					//cout << "ii = " << ii << " " << "jj = " << jj <<endl;
					buffer_E_prev[ii*cols+jj] = E_prev(ii,jj);
					buffer_R[ii*cols+jj] = R(ii,jj);
					buffer_E[ii*cols+jj] = E(ii,jj);
					//if(rank == 5)
					//cout << "buffer_E_prev = "<<buffer_E_prev[ii*cols+jj]<<endl;
				}
			}
			//cout<< "MPI_SEND"<<endl;
			//cout<< "Rows while sending = "<<rows<<"Cols while sending = "<<cols<<endl;
      		MPI_Send(buffer_E_prev,rows*cols,MPI_DOUBLE,rank,0,MPI_COMM_WORLD);
      		MPI_Send(buffer_R,rows*cols,MPI_DOUBLE,rank,1,MPI_COMM_WORLD);
      		MPI_Send(buffer_E,rows*cols,MPI_DOUBLE,rank,2,MPI_COMM_WORLD);
      
      		//MPI_Wait(buffer_E_prev,sizeof(double),MPI_DOUBLE,rank,0,MPI_COMM_WORLD);
      		//MPI_Wait(buffer_R,sizeof(double),MPI_DOUBLE,rank,1,MPI_COMM_WORLD);
      		//MPI_Wait(buffer_E,sizeof(double),MPI_DOUBLE,rank,2,MPI_COMM_WORLD);
      	//cout<< "rank = "<<rank<<endl;
		}
      }
      else {
		int rows,cols,incr_row,incr_col;
		rows = n/cb.py;
		cols = n/cb.px;
		incr_row = n%(cb.py);
		incr_col = n%(cb.px);
		if(myrank/cb.px < incr_row) {
			rows++;
		}
		//cout<< "MPI_RCV"<<endl;
		if(myrank%cb.px < incr_col) {
			cols++;
		}
		//cout<<"Rank = "<<myrank<<endl;
	//	cout<<"Rows = "<<rows<<" "<<"Cols = "<<cols<<endl;
		double* buffer_E_prev_tmp = (double*)malloc(rows*cols*sizeof(double));	
		double* buffer_R_tmp = (double*)malloc(rows*cols*sizeof(double));	
		double* buffer_E_tmp = (double*)malloc(rows*cols*sizeof(double));	
		MPI_Recv(buffer_E_prev_tmp,rows*cols,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		MPI_Recv(buffer_R_tmp,rows*cols,MPI_DOUBLE,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		MPI_Recv(buffer_E_tmp,rows*cols,MPI_DOUBLE,0,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		ar.m = rows + 2;
		ar.n = cols + 2;
		ar.E_prev = (double*)malloc(ar.m*ar.n*sizeof(double)); 
		ar.R = (double*)malloc(ar.m*ar.n*sizeof(double)); 
		ar.E = (double*)malloc(ar.m*ar.n*sizeof(double)); 

			for(int ii = 0 ; ii < rows ; ii+=1) {
				for(int jj = 0 ; jj < cols ; jj+=1) {
					//cout << "ii = " << ii << " " << "jj = " << jj <<endl;
					ar.E_prev[(ii+1)*ar.n+jj+1] = buffer_E_prev_tmp[ii*cols+jj];
					ar.R[(ii+1)*ar.n+jj+1] = buffer_R_tmp[ii*cols+jj];
					ar.E[(ii+1)*ar.n+jj+1] = buffer_E_tmp[ii*cols+jj];
					if(myrank == 5) {
					//cout << "buffer_E_prev_tmp = "<<buffer_E_prev_tmp[ii*cols+jj]<<endl;
					//cout << "E_prev = "<<ar.E_prev[(ii+1)*ar.n+jj+1]<<endl;
				}
			}
}
		//MPI_Wait(buffer_E_prev,sizeof(double),MPI_DOUBLE,rank,0,MPI_COMM_WORLD);
		//MPI_Wait(buffer_R,sizeof(double),MPI_DOUBLE,rank,1,MPI_COMM_WORLD);
		//MPI_Wait(buffer_E,sizeof(double),MPI_DOUBLE,rank,2,MPI_COMM_WORLD);
	}	

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
