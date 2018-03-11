/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "cblock.h"
#include "arblock.h"
#include "Plotting.h"
#include <emmintrin.h>
#ifdef _MPI_
#include <mpi.h>
#endif

using namespace std;

#define E_prev(ii,jj) E_prev[(ii)*(n+2)+jj]
#define R(ii,jj) R[(ii)*(n+2)+jj]
#define E(ii,jj) E[(ii)*(n+2)+jj]

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);

extern control_block cb;
extern array_chunk ar;

#ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
__attribute__((optimize("no-tree-vectorize")))
#endif 

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq){
    double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

 // Simulated time is different from the integer timestep number
 double t = 0.0;

 double *E = *_E, *E_prev = *_E_prev;
 double *R_tmp = R;
 double *E_tmp = *_E;
 double *E_prev_tmp = *_E_prev;
 double mx, sumSq;
 int niter;
 int m = cb.m, n=cb.n;
 //int m = ar.m, n=ar.n;
 int innerBlockRowStartIndex = (n+2)+1;
 int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);
 int nprocs,myrank;

	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

int rows,cols,incr_row,incr_col;

rows = (n+2)/cb.py;
cols = (n+2)/cb.px;

incr_row = (n+2)%cb.py;
incr_col = (n+2)%cb.px;

//rows calculation
if(myrank/cb.px < incr_row) { 
rows++;
}

//rows calculation
if(myrank%cb.px < incr_col) {
cols++;
}


 // We continue to sweep over the mesh until the simulation has reached
 // the desired number of iterations
  for (niter = 0; niter < cb.niters; niter++){
  
      if  (cb.debug && (niter==0)){
	  stats(E_prev,m,n,&mx,&sumSq);
          double l2norm = L2Norm(sumSq);
	  repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
	  //repNorms(l2norm,mx,dt,rows,cols,-1, cb.stats_freq);
	  if (cb.plot_freq)
	      plotter->updatePlot(E,  -1, m+1, n+1);
      }

   /* 
    * Copy data from boundary of the computational box to the
    * padding region, set up for differencing computational box's boundary
    *
    * These are physical boundary conditions, and are not to be confused
    * with ghost cells that we would use in an MPI implementation
    *
    * The reason why we copy boundary conditions is to avoid
    * computing single sided differences at the boundaries
    * which increase the running time of solve()
    *
    */
    
    // 4 FOR LOOPS set up the padding needed for the boundary conditions
    int i,j;

   // Fills in the TOP Gh st Cells
	    for (i = 0; i < (n+2); i++) {
		    E_prev[i] = E_prev[i + (n+2)*2];
	    }

    // Fills in the RIGHT Ghost Cells
    for (i = (n+1); i < (m+2)*(n+2); i+=(n+2)) {
	    E_prev[i] = E_prev[i-2];
    }

    // Fills in the LEFT Ghost Cells
    for (i = 0; i < (m+2)*(n+2); i+=(n+2)) {
	    E_prev[i] = E_prev[i+2];
    }	

    // Fills in the BOTTOM Ghost Cells
    for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++) {
	    E_prev[i] = E_prev[i - (n+2)*2];
}

//////////////////////////////////////////////////////////////////////////////

#define FUSED 1

//#ifdef _MPI_
//
////E_tmp = E;
////E_prev _tmp = E_prev;
////R_rmp = R;
////double* E_copy;
//
////E_copy = E_prev_tmp + (n+2) + 1;
//
//double* buffer_top_rcv = (double*)malloc(cols*sizeof(double));
//double* buffer_bottom_rcv = (double*)malloc(cols*sizeof(double));
//double* buffer_right_rcv = (double*)malloc(rows*sizeof(double));
//double* buffer_left_rcv = (double*)malloc(rows*sizeof(double));
//double* buffer_top_snd = (double*)malloc(cols*sizeof(double));
//double* buffer_bottom_snd = (double*)malloc(cols*sizeof(double));
//double* buffer_right_snd = (double*)malloc(rows*sizeof(double));
//double* buffer_left_snd = (double*)malloc(rows*sizeof(double));
//
////E_copy = E_copy + (rank/px) * rows * (n+2);
////E_prev_tmp = E_copy + (rank%cb.px) * cols; 
////E_prev_tmp = E_prev_tmp + (n+2) + 1;
//
////Rows Message Passing
//if(cb.py > 1) {
//	if(rank/cb.px == 0) {
//		for(int j = 0; j < cols; j++) {
//			//E_prev_tmp = E_prev_tmp - (n+2);
//			//buffer_top_rcv[j] = E_prev_tmp[j];
//			E_prev_tmp = E_prev_tmp + (n+2)*(rows-1);
//			buffer_bottom_snd[j] = E_prev_tmp[j];
//			E_prev_tmp = E_prev_tmp - ((rows - 1) * (n+2));
//			}
//
//			MPI_Send(buffer_top_snd,rows*cols,MPI_DOUBLE,rank+cb.px,0,MPI_COMM_WORLD);
//			MPI_Recv(buffer_bottom_rcv,rows*cols,MPI_DOUBLE,rank+cb.px,0,MPI_COMM_WORLD);	
//		for(int j = 0 ; j < cols; j++) {
//			E_prev_tmp = E_prev_tmp + ((n+2)*rows);
//			E_prev_tmp[j] = buffer_top_rcv[j];
//			E_prev_tmp = E_prev_tmp - (rows * (n+2));
//		}		
//}	
//	else if(rank/cb.px == cb.py - 1) {
//		for(int j = 0; j < cols; j++) {
//			//E_prev_tmp = E_prev_tmp + (rows * (n+2));
//			//buffer_bottom_rcv[j] = E_prev_tmp[j];
//			//E_prev_tmp = E_prev_tmp - (rows * (n+2));
//			buffer_top_snd[j] = E_prev_tmp[j]; 
//		}
//			
//			MPI_Send(buffer_top_snd,rows*cols,MPI_DOUBLE,rank-cb.px,0,MPI_COMM_WORLD);
//			MPI_Rcv(buffer_top_rcv,rows*cols,MPI_DOUBLE,rank-cb.px,0,MPI_COMM_WORLD);
//			
//		for(int j = 0 ; j < cols ; j++) {
//			//E_prev_tmp = E_prev_tmp + (rows * (n+2));
//			//E_prev_tmp[j] = buffer_bottom_rcv[j];
//			E_prev_tmp = E_prev_tmp - (rows * (n+2));
//			E_prev_tmp = E_prev_tmp - (n+2);
//			E_prev_tmp[j] = buffer_top_rcv[j];
//			E_prev_tmp = E_prev_tmp + (n+2);
//		}
//	}
//	else {
//		for(int j = 0 ; j < cols ; j++) {
//			buffer_top_snd[j] = E_prev_tmp[j];
//			E_prev_tmp = E_prev_tmp + (rows-1)*(n+2);
//			buffer_bottom_snd[j] = E_prev_tmp[j];
//			E_prev_tmp = E_prev_tmp - (rows-1)*(n+2);
//		}
//			
//			MPI_Send(buffer_top_snd,rows*cols,MPI_DOUBLE,rank-cb.px,0,MPI_COMM_WORLD);
//			MPI_Send(buffer_bottom_snd,rows*cols,MPI_DOUBLE,rank+cb.px,0,MPI_COMM_WORLD);
//			MPI_Rcv(buffer_top_rcv,rows*cols,MPI_DOUBLE,rank-cb.px,0,MPI_COMM_WORLD);
//			MPI_Rcv(buffer_bottom_rcv,rows*cols,MPI_DOUBLE,rank+cb.px,0,MPI_COMM_WORLD);
//		
//		for(int j = 0 ; j < cols ; j++) {
//			E_prev_tmp = E_prev_tmp - (n+2);
//			E_prev_tmp[j] = buffer_top_rcv[j];
//			E_prev_tmp = E_prev_tmp + (n+2);
//			E_prev_tmp = E_prev_tmp + (rows*(n+2));
//			E_prev_tmp[j] = buffer_bottom_rcv[j];
//			E_prev_tmp = E_prev_tmp - (rows*(n+2));
//		}        	
//	}	
//}
////else {
////		for(int j = 0 ; j < cols ; j++) {
////			E_prev_tmp = E_prev_tmp - (n+2);
////			buffer_top_rcv[j] = E_prev_tmp[j];
////			E_prev_tmp = E_prev_tmp + (n+2);
////			E_prev_tmp = E_prev_tmp + (rows*(n+2));
////			buffer_bottom_rcv[j] = E_prev_tmp[j];
////			E_prev_tmp = E_prev_tmp - (rows*(n+2));
////		}
//}
//
////Columns Message Passing
//if(cb.px > 1) {
//	if(rank%cb.px == 0) {
//		for(int i = 0 ; i < rows ; i++) {
//			E_prev_tmp = E_prev_tmp - 1;
//			E_prev_tmp = E_prev_tmp + (n+2)*i;
//			buffer_left_rcv[i] = E_prev_tmp[0];
//			E_prev_tmp = E_prev_tmp + cols + 1;
//			buffer_right_snd[i] = E_prev_tmp[0];
//			E_prev_tmp = E_prev_tmp - (n+2)*i - cols;
//		}
//
//			MPI_Send(buffer_right_snd,rows*cols,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD);
//			MPI_Rcv(buffer_right_rcv,rows*cols,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD);
//
//		for(int i = 0 ; i < rows ; i++) {
//			E_prev_tmp = E_prev_tmp - 1;
//			E_prev_tmp = E_prev_tmp + (n+2)*i;
//			E_prev_tmp[0] = buffer_left_rcv[i];
//			E_prev_tmp = E_prev_tmp + cols + 1;
//			E_prev_tmp[0] = buffer_right_rcv[i];
//			E_prev_tmp = E_prev_tmp - (n+2)*i - cols; 	
//		}
//				
//	}
//	else if (rank%cb.px == cb.px - 1) {
//		for(int i = 0 ; i < rows ; i++) {
//			E_prev_tmp = E_prev_tmp + (n+2)*i;
//			E_prev_tmp = E_prev_tmp + cols;
//			buffer_right_rcv[i] = E_prev_tmp[0];
//			E_prev_tmp = E_prev_tmp - (n+2)*i - cols;
//			E_prev_tmp = E_prev_tmp - 1;
//			E_prev_tmp = E_prev_tmp + (n+2)*i;
//			buffer_left_snd[i] = E_prev[0];	
//			E_prev_tmp = E_prev_tmp - (n+2)*i + 1; 
//		}
//				
//			MPI_Send(buffer_left_snd,rows*cols,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD);
//			MPI_Rcv(buffer_left_rcv,rows*cols,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD);
//		
//		for(int i = 0 ; i < rows ; i++) {
//			E_prev_tmp = E_prev_tmp + (n+2)*i;
//			E_prev_tmp = E_prev_tmp + cols;
//			E_prev_tmp[0] = buffer_right_rcv[i];
//			E_prev_tmp = E_prev_tmp - (n+2)*i - cols;
//			E_prev_tmp = E_prev_tmp - 1;
//			E_prev_tmp = E_prev_tmp + (n+2)*i;
//			E_prev_tmp[0] = buffer_left_rcv[i];
//			E_prev_tmp = E_prev_tmp - (n+2)*i + 1;
//		}	
//	}
//	else {
//		for(int i = 0 ; i < rows ; i++) {
//		E_prev_tmp = E_prev_tmp - 1;
//		E_prev_tmp = E_prev_tmp + (n+2)*i;
//		buffer_left_snd[i] = E_prev_tmp[0];
//		E_prev_tmp = E_prev_tmp + cols + 1;
//		buffer_right_snd[i] = E_prev_mp[0];
//		E_prev_tmp = E_prev_tmp - (n+2)*i - cols;
//		}
//
//			MPI_Send(buffer_left_snd,rows*cols,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD);
//			MPI_Send(buffer_right_snd,rows*cols,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD);
//			MPI_Rcv(buffer_left_rcv,rows*cols,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD);
//			MPI_Rcv(buffer_right_rcv,rows*cols,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD);
//		
//		for(int i = 0 ; i < rows ; i++) {
//			E_prev_tmp = E_prev_tmp - 1;
//			E_prev_tmp = E_prev_tmp + (n+2)*i;
//			E_prev_tmp[0] = buffer_left_rcv[i];
//			E_prev_tmp = E_prev_tmp + cols + 1;
//			E_prev_tmp[0] = buffer_right_rcv[i];
//			E_prev_tmp = E_prev_tmp - (n+2)*i - cols;
//		}
//	}
//}
////else {
////		for(int i = 0 ; i < rows ; i++) {
////			E_prev_tmp = E_prev_tmp - 1;
////			E_prev_tmp = E_prev_tmp + (n+2)*i;
////			buffer_left_rcv[i] = E_prev_tmp[0];
////			E_prev_tmp = E_prev_tmp + cols + 1;
////			buffer_right_rcv[i] = E_prev_tmp[0];
////			E_prev_tmp = E_prev_tmp - (n+2)*i - cols;
////		}
////
//
//    for(j = (n+2) + 1; j <= rows - 1; j+=(n+2)) {
//        E_tmp = E + j;
//	E_prev_tmp = E_prev + j;
//        R_tmp = R + j;
//	for(i = 0; i < cols; i++) {
//	    E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
//            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
//            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
//        }
//    }
//
////#endif
//#else


#ifdef FUSED
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
	E_prev_tmp = E_prev + j;
        R_tmp = R + j;
	for(i = 0; i < n; i++) {
	    E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#else
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            }
    }

    /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
        R_tmp = R + j;
	E_prev_tmp = E_prev + j;
        for(i = 0; i < n; i++) {
	  E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
	  R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#endif

//#endif
     /////////////////////////////////////////////////////////////////////////////////

   if (cb.stats_freq){
     if ( !(niter % cb.stats_freq)){
        stats(E,m,n,&mx,&sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
    }
   }

   if (cb.plot_freq){
          if (!(niter % cb.plot_freq)){
	    plotter->updatePlot(E,  niter, m, n);
        }
    }

   // Swap current and previous meshes
   double *tmp = E; E = E_prev; E_prev = tmp;

 } //end of 'niter' loop at the beginning

  // return the L2 and infinity norms via in-out parameters
  stats(E_prev,m,n,&Linf,&sumSq);
  L2 = L2Norm(sumSq);

  // Swap pointers so we can re-use the arrays
  *_E = E;
  *_E_prev = E_prev;
}
