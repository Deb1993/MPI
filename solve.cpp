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

void printMat1(const char mesg[], double *E, int m, int n){
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
 R = ar.R;
 //double *E_tmp = *_E;
 double *E_tmp = ar.E;
 //double *E_prev_tmp = *_E_prev;
 double *E_prev_tmp = ar.E_prev;
 double mx, sumSq,fsumSq,fLinf;
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
//if (rank ==3)
//{ cout<<" rank initial is" << rank <<endl;
//printMat1("printing initial eprev",E_prev,m,n);}

 int x1=rank%cb.px;
 int y1=rank/cb.px;
 //m = cb.m / cb.py + (y1 < (cb.m % cb.py));
 //n = cb.n / cb.px + (x1 < (cb.n % cb.px));

 int m1 = ar.m-2, n1=ar.n-2;
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

    //printMat1("printing initial eprev",E_prev,m,n);
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

//if (rank==3)
//{printMat1("printing eprev after padding",E_prev,m,n);
//cout<<"rank padding is"<<rank<<endl;}
//////////////////////////////////////////////////////////////////////////////

#define FUSED 1

if(!cb.noComm) {
#ifdef _MPI_

 int N=n;
 int M=m;
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
MPI_Isend(buffer_left_sen, m, MPI_DOUBLE, rank - 1, 0,MPI_COMM_WORLD , send + count);
//printmat("leftbuffer", buffer_left_sen,1,m);
/*cout<<"buferleftsen";
for (i=0;i<m;i++)
	cout<<buffer_left_sen[i]<<" ";
cout<<endl;*/
MPI_Irecv(buffer_left_rec, m, MPI_DOUBLE, rank - 1, 0,MPI_COMM_WORLD , rec+ count);
count++;
}


if(x1!=cb.px-1)
{
//right
MPI_Isend(buffer_right_sen, m, MPI_DOUBLE, rank + 1, 0,MPI_COMM_WORLD , send+count);
MPI_Irecv(buffer_right_rec, m, MPI_DOUBLE, rank + 1, 0,MPI_COMM_WORLD , rec+count);
count++;
}




if(y1!=0)
{
//up

//if (rank==3)
//{
//cout<<"bufertopsen";
//for (i=0;i<n;i++)
//	cout<<buffer_top_sen[i]<<" ";
//cout<<endl;
//}

MPI_Isend(buffer_top_sen, n, MPI_DOUBLE, rank - cb.px, 0,MPI_COMM_WORLD , send+count);
MPI_Irecv(buffer_top_rec, n, MPI_DOUBLE, rank - cb.px, 0,MPI_COMM_WORLD , rec+count);
count++;
}


if(y1!=cb.py-1)
{
//down
MPI_Isend(buffer_bottom_sen, n, MPI_DOUBLE, rank + cb.px, 0,MPI_COMM_WORLD , send+count);
MPI_Irecv(buffer_bottom_rec, n, MPI_DOUBLE, rank + cb.px, 0,MPI_COMM_WORLD , rec+count);
count++;
}


MPI_Waitall(count,rec,stat);


 if (x1!=0)
  {
//left
   int j=0;
   for(i=n+2; i<(m+1)*(n+2); i+=n+2)
     	{	E_prev[i]=buffer_left_rec[j];
         j++;
	}
}
//Put right column of matrix in send right
 if (x1!=cb.px-1)
{ int j=0;
   for(i=n+3+n; i<(m+1)*(n+2); i+=n+2)
 {    		E_prev[i]=buffer_right_rec[j];
	j++;
}
}

//Put top row of matrix in send up
 if (y1!=0)
{ int j=0;
   for(i=1; i<(n+1); i++)
{     		E_prev[i]= buffer_top_rec[j];
	j++; 
}
}
 
//Put bottom row of matrix in send down
 if (y1!=cb.py-1)
{ int j=0;
   for(i=(m+1)*(n+2)+1; i<(m+1)*(n+2)+n+1; i++)
{
     		E_prev[i]=buffer_bottom_rec[j];
	j++;
}
}


}
#endif
}
//if (rank==1)
//{
//cout<<"rank final is"<<rank<<endl;
//printMat1("printing final eprev",E_prev,m,n);
//}

#ifdef FUSED
    // Solve for the excitation, a PDE

if (rank==0)
{
//cout<<"rank final is"<<rank<<endl;
//printMat1("printing eprev before op",E_prev,m,n);
}
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
//MPI_Reduce(void* send_data, void* recv_data,int count,MPI_Datatype datatype,MPI_Op op,int root,MPI_Comm communicator)
#ifdef _MPI_
if (!cb.noComm)
{	MPI_Reduce(&sumSq,&fsumSq ,1,MPI_DOUBLE,MPI_SUM,0 ,MPI_COMM_WORLD);
	MPI_Reduce(&Linf,&fLinf ,1,MPI_DOUBLE,MPI_MAX,0 ,MPI_COMM_WORLD);

  Linf = fLinf;
}
else
{	fsumSq =sumSq;

}
#else
	fsumSq =sumSq;
#endif
  L2 = L2Norm(fsumSq);

if (rank==0)
{
//cout<<"rank final is"<<rank<<endl;
//printMat1("printing final eprev",E_prev,m,n);
}

#ifdef _MPI_
free(ar.R);
free(ar.E);
free(ar.E_prev);
#else

  // Swap pointers so we can re-use the arrays
  *_E = E;
  *_E_prev = E_prev;

#endif
}
