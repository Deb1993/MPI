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
#include "Plotting.h"
#include "cblock.h"
#include "arblock.h"
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

 //double *E = *_E, *E_prev = *_E_prev;
 double *E = ar.E, *E_prev = ar.E_prev;
 //double *R_tmp = R;
 double *R_tmp = ar.R;
 //double *E_tmp = *_E;
 double *E_tmp = ar.E;
 //double *E_prev_tmp = *_E_prev;
 double *E_prev_tmp = ar.E_prev;
 double mx, sumSq;
 int niter;
 //int m = cb.m, n=cb.n;
 int m = ar.m-2, n=ar.n-2;
 //cout<<" m1 is "<< m1<<endl;
 //cout<<" n1 is "<< n1<<endl;
 int innerBlockRowStartIndex = (n+2)+1;
 int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);
 int rank =0, np=1;
#ifdef _MPI_
 MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#endif
 int x1=rank%cb.px;
 int y1=rank/cb.px;
 //m = cb.m / cb.py + (y1 < (cb.m % cb.py));
 //n = cb.n / cb.px + (x1 < (cb.n % cb.px));

 cout<<" rank is "<< rank<<endl;
 cout<<" m is "<< m<<endl;
 cout<<" n is "<< n<<endl;
 int m1 = ar.m-2, n1=ar.n-2;
 cout<<" m1 is "<< m1<<endl;
 cout<<" n1 is "<< n1<<endl;
 // We continue to sweep over the mesh until the simulation has reached
 // the desired number of iterations
  for (niter = 0; niter < cb.niters; niter++){
  
      if  (cb.debug && (niter==0)){
	  stats(E_prev,m,n,&mx,&sumSq);
          double l2norm = L2Norm(sumSq);
	  repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
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

    // Fills in the TOP Ghost Cells
    if (y1==0)
    {
    for (i = 0; i < (n+2); i++) {
        E_prev[i] = E_prev[i + (n+2)*2];
    }
    }
     
    if (x1==cb.px-1)
    {
    // Fills in the RIGHT Ghost Cells
    for (i = (n+1); i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i-2];
    }
    }
    if (x1==0)
    {
    // Fills in the LEFT Ghost Cells
    for (i = 0; i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i+2];
    }
    }	
    
    if (y1==cb.py-1)
    {
    // Fills in the BOTTOM Ghost Cells
    for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++) {
        E_prev[i] = E_prev[i - (n+2)*2];
    }
    }

//////////////////////////////////////////////////////////////////////////////

#define FUSED 1


#ifdef _MPI_

 int N=n+2;
 int M=m+2;
 double* buffer_top_sen = (double*)malloc(N*sizeof(double));
 double* buffer_bottom_sen = (double*)malloc(N*sizeof(double));
 double* buffer_right_sen = (double*)malloc(M*sizeof(double));
 double* buffer_left_sen = (double*)malloc(M*sizeof(double));
 
 double* buffer_top_rec = (double*)malloc(N*sizeof(double));
 double* buffer_bottom_rec = (double*)malloc(N*sizeof(double));
 double* buffer_right_rec = (double*)malloc(M*sizeof(double));
 double* buffer_left_rec = (double*)malloc(M*sizeof(double));

if ((cb.px>1) || (cb.py>1))
{
                         
 // Put left column of the matrix in send left
 if (x1!=0)
  {
   int j=0;
   for(i=n+3; i<(m+1)*(n+2); i+=n+2)
     {		buffer_left_sen[j] =E_prev[i];
     j++;
}
	}

//Put right column of matrix in send right
 if (x1!=cb.px-1)
{ int j=0;
   for(i=n+2+n; i<(m+1)*(n+2); i+=n+2)
     	{	buffer_right_sen[j] =E_prev[i];
	       j++;
        }
}


//Put top row of matrix in send up
 if (y1!=0)
{ int j=0;
   for(i=n+3; i<(2*n+3); i++)
     {		buffer_top_sen[j] =E_prev[i];
       j++;
     }
}

 
//Put bottom row of matrix in send down
 if (y1!=cb.py-1)
{ int j=0;
   for(i=m*(n+2)+1; i<(m)*(n+2)+n+1; i++)
   {  		buffer_bottom_sen[j] =E_prev[i];
   		j++;
}
}



//MPI sending and receiving

MPI_Request send[4];
MPI_Request rec[4];
MPI_Status stat[4];
int count=0;
if(x1!=0)
{
//left
MPI_Isend(buffer_left_sen, M, MPI_DOUBLE, rank - 1, 0,MPI_COMM_WORLD , send + count);
MPI_Irecv(buffer_left_rec, M, MPI_DOUBLE, rank - 1, 0,MPI_COMM_WORLD , rec+ count);
count++;
}


if(x1!=cb.px-1)
{
//right
MPI_Isend(buffer_right_sen, M, MPI_DOUBLE, rank + 1, 0,MPI_COMM_WORLD , send+count);
MPI_Irecv(buffer_right_rec, M, MPI_DOUBLE, rank + 1, 0,MPI_COMM_WORLD , rec+count);
count++;
}




if(y1!=0)
{
//up
MPI_Isend(buffer_top_sen, N, MPI_DOUBLE, rank - cb.px, 0,MPI_COMM_WORLD , send+count);
MPI_Irecv(buffer_top_rec, N, MPI_DOUBLE, rank - cb.px, 0,MPI_COMM_WORLD , rec+count);
count++;
}


if(y1!=cb.py-1)
{
//down
MPI_Isend(buffer_bottom_sen, N, MPI_DOUBLE, rank + cb.px, 0,MPI_COMM_WORLD , send+count);
MPI_Irecv(buffer_bottom_rec, N, MPI_DOUBLE, rank + cb.px, 0,MPI_COMM_WORLD , rec+count);
count++;
}


MPI_Waitall(count,rec,stat);


 if (x1!=0)
  {
   int j=0;
   for(i=n+3; i<(m+1)*(n+2); i+=n+2)
     	{	E_prev[i]=buffer_left_rec[j];
         j++;
	}
}
//Put right column of matrix in send right
 if (x1!=cb.px-1)
{ int j=0;
   for(i=n+2+n; i<(m+1)*(n+2); i+=n+2)
 {    		E_prev[i]=buffer_right_sen[j];
	j++;
}
}

//Put top row of matrix in send up
 if (y1!=0)
{ int j=0;
   for(i=n+3; i<(2*n+3); i++)
{     		E_prev[i]= buffer_right_sen[j];
	j++; 
}
}
 
//Put bottom row of matrix in send down
 if (y1!=cb.py-1)
{ int j=0;
   for(i=(m)*(n+2)+1; i<(m)*(n+2)+n+1; i++)
{
     		E_prev[i]=buffer_right_sen[j];
	j++;
}
}


}
#endif



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
