
#include <cuda_runtime.h>
#include <cufft.h>
//#include <cutil.h>
//#include <cutil_inline.h>

#include "stdio.h"
#include "math.h"
#include "memory.h"

#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "Tomo_OSTR.h"
#include "readdata.h"
//#include <direct.h>
//#include <helper_cuda.h>
#include <stdio.h>

//declare CPU ram global var here (can be delcared in any .cu or .cpp file )
int h_IMGSIZx;
int h_IMGSIZy;
int h_IMGSIZz;
float h_Vsize_x;
float h_Vsize_y;
float h_Vsize_z;
float h_x_p0;
float h_y_p0;
float h_z_p0;

float h_x_d0;
float h_y_d0;

float h_detectorR;
float h_sourceR;
float h_sourceY;

int h_ANGLES;
int *h_index;
float *h_angles;


int h_BACKPRJ_ThreX; // This doesn't need to be copied to device memo since it'll be only used in dim3(h_BACKPRJ_ThreX,h_BACKPRJ_ThreY) and it's on host. 
int h_BACKPRJ_ThreY; 
int h_BACKPRJ_GridX;
int h_BACKPRJ_GridY;
int h_nBatchXdim;

int h_BINSx;
int h_BINSy;
float h_Bsize_x;
float h_Bsize_y;

int h_nBatchBINSx;
int h_nBatchBINSy;
int h_PRJ_ThreX;
int h_PRJ_ThreY;
int h_PRJ_GridX;
int h_PRJ_GridY;

float h_beta;
float h_delta;
int h_iter_num;
int h_subset_num;

int h_IO_Iter;
int h_method;

char h_ini_name[300];
char h_angle_filename[300];
char h_prj_folder[200];
char h_prj_name[200];
char h_scat_folder[200];
char h_scat_name[200];
char h_blank_folder[200];
char h_blank_name[200];
char h_recon_folder[200];
char h_recon_name[200];


__constant__ int IMGSIZx;
__constant__ int IMGSIZy;
__constant__ int IMGSIZz;
__constant__ float Vsize_x;
__constant__ float Vsize_y;
__constant__ float Vsize_z;
__constant__ float x_p0;
__constant__ float y_p0;
__constant__ float z_p0;

__constant__ float x_d0;
__constant__ float y_d0;

__constant__ float detectorR;
__constant__ float sourceR;
__constant__ float sourceY;

__constant__ int BACKPRJ_ThreX; // these will not be used, but just in case if later inside device function need these param
__constant__ int BACKPRJ_ThreY;
__constant__ int BACKPRJ_GridX;
__constant__ int BACKPRJ_GridY;
__constant__ int nBatchXdim;
__constant__ int ANGLES;

__constant__ int BINSx;
__constant__ int BINSy;
__constant__ float Bsize_x;
__constant__ float Bsize_y;

__constant__ int nBatchBINSx;
__constant__ int nBatchBINSy;
__constant__ int PRJ_ThreX;
__constant__ int PRJ_ThreY;
__constant__ int PRJ_GridX;
__constant__ int PRJ_GridY;


__global__ void G_Q2_prior(
	float *RDD, 
	float *RD,
	float *estbuf,
	float z_xy_ratio,
	int   nbatchIDx)
{
	// Block index
	const int bx = blockIdx.x;
	const int by = blockIdx.y;

	// Thread index
	// const int tx = threadIdx.x;
	const int tx = threadIdx.x+(nbatchIDx*blockDim.x);
	const int ty = threadIdx.y;

	int		ind_x, ind_y, ind_z;
	int		ind_nr_x, ind_nr_y, ind_nr_z;
	int     	ind_nr;
	int		bin_ind;
	long		ind_voxel;

	int		status;
	int		cent=1;

	float	distance,y_z_ratio,x_z_ratio;
	float   diff, RDD_tmp;
	//const int tid = tx * blockDim.y + ty;
	const int tid=tx;//+ty*blockDim.x;//+nbatchIDx*blockDim.x;//*blockDim.y;  

	//Calculate the index of the voxel being considered.
	//ind_x = nbatchIDx*((int)(IMGSIZx/h_nBatchXdim))+ tid;
	ind_x = tid;
	ind_y = bx;
	ind_z = by;

	ind_voxel=(ind_z*IMGSIZy+ind_y)*IMGSIZx+ind_x;  //(if prj is scanner data, need x_y_flip)
	//ind_voxel=(ind_z*IMGSIZx+ind_x)*IMGSIZy+ind_y;		

	for ( ind_nr_z = (ind_z-1);ind_nr_z < (ind_z+2); ind_nr_z++ ){
		for ( ind_nr_y = (ind_y-1); ind_nr_y < (ind_y+2); ind_nr_y++ ){
			for( ind_nr_x = (ind_x-1); ind_nr_x < (ind_x+2); ind_nr_x++){


				distance = sqrt(float((ind_nr_x-ind_x)*(ind_nr_x-ind_x)+(ind_nr_y-ind_y)*(ind_nr_y-ind_y)+(ind_nr_z-ind_z)*(ind_nr_z-ind_z)*z_xy_ratio*z_xy_ratio));

				if(distance==0.0){
					distance = 1.0;
					cent     = 0;
				}
				if(ind_nr_x<0 | ind_nr_y<0 | ind_nr_z<0 | ind_nr_x>(IMGSIZx-1) | ind_nr_y>(IMGSIZy-1) | ind_nr_z>(IMGSIZz-1))
					ind_nr = ind_voxel;
				else
					ind_nr = ind_nr_x+ind_nr_y*IMGSIZx+ind_nr_z*IMGSIZx*IMGSIZy; 

				diff    = estbuf[ind_voxel]-estbuf[ind_nr];
				RDD_tmp = cent*(1.0/distance);

				RDD[ind_voxel] = RDD[ind_voxel] + RDD_tmp;
				RD[ind_voxel]  = RD[ind_voxel]  + RDD_tmp*diff;

				cent=1; // reset cent;
			}// ind_nr_x loop
		}//ind_nr_y loop
	}//ind_nr_z loop

}


__global__ void G_Fessler_prior(
	float *RDD, 
	float *RD,
	float *estbuf,
	float delta,
	float z_xy_ratio,
	int   nbatchIDx)
{
	// Block index
	const int bx = blockIdx.x;
	const int by = blockIdx.y;

	// Thread index
	//const int tx = threadIdx.x;
	const int tx = threadIdx.x+(nbatchIDx*blockDim.x);
	const int ty = threadIdx.y;

	int		ind_x, ind_y, ind_z;
	int		ind_nr_x, ind_nr_y, ind_nr_z;
	int             ind_nr;
	int		bin_ind;
	long		ind_voxel;

	int		status;
	int		cent=1;

	float	distance,y_z_ratio,x_z_ratio;
	float   diff, denominator, RDD_tmp;
	//const int tid = tx * blockDim.y + ty;
	const int tid=tx;//+ty*blockDim.x;//+nbatchIDx*blockDim.x;//*blockDim.y;  

	//Calculate the index of the voxel being considered.
	//ind_x = nbatchIDx*((int)(IMGSIZx/h_nBatchXdim))+ tid;
	ind_x = tid;
	ind_y = bx;
	ind_z = by;

	ind_voxel=(ind_z*IMGSIZy+ind_y)*IMGSIZx+ind_x;  //(if prj is scanner data, need x_y_flip)
	//ind_voxel=(ind_z*IMGSIZx+ind_x)*IMGSIZy+ind_y;		

	for ( ind_nr_z = (ind_z-1);ind_nr_z < (ind_z+2); ind_nr_z++ ){
		for ( ind_nr_y = (ind_y-1); ind_nr_y < (ind_y+2); ind_nr_y++ ){
			for( ind_nr_x = (ind_x-1); ind_nr_x < (ind_x+2); ind_nr_x++){


				distance = sqrt(float((ind_nr_x-ind_x)*(ind_nr_x-ind_x)+(ind_nr_y-ind_y)*(ind_nr_y-ind_y)+(ind_nr_z-ind_z)*(ind_nr_z-ind_z)*z_xy_ratio*z_xy_ratio));

				if(distance == 0.0){
					distance = 1.0;
					cent     = 0;
				}
				if(ind_nr_x<0 || ind_nr_y<0 || ind_nr_z<0 || ind_nr_x>(IMGSIZx-1) || ind_nr_y>(IMGSIZy-1) || ind_nr_z>(IMGSIZz-1))
					ind_nr = ind_voxel;
				else
					ind_nr = ind_nr_x + ind_nr_y*IMGSIZx + ind_nr_z*IMGSIZx*IMGSIZy; 

				diff        = estbuf[ind_voxel]-estbuf[ind_nr];
				denominator = 1.0+abs(diff/delta);
				RDD_tmp     = cent*(1.0/distance)/denominator;

				RDD[ind_voxel] = RDD[ind_voxel] + RDD_tmp;
				RD[ind_voxel]  = RD[ind_voxel]  + RDD_tmp*diff;

				cent=1; // reset cent;
			}// ind_nr_x loop
		}//ind_nr_y loop
	}//ind_nr_z loop

}


__global__ void G_Huber_prior(
	float *priorbuf, 
	float *estbuf,
	float delta,
	int nbatchIDx)
{
	// Block index
	const int bx = blockIdx.x;
	const int by = blockIdx.y;

	// Thread index
	//const int tx = threadIdx.x;
	const int tx = threadIdx.x+(nbatchIDx*blockDim.x);
	const int ty = threadIdx.y;

	int		ind_x, ind_y, ind_z;
	int		ind_nr_x, ind_nr_y, ind_nr_z;
	int     ind_nr;
	int		bin_ind;
	long	ind_voxel;

	int		status;

	float	distance;
	float   diff, denominator;
	//const int tid = tx * blockDim.y + ty;
	const int tid=tx;//+ty*blockDim.x;//+nbatchIDx*blockDim.x;//*blockDim.y;  

	//Calculate the index of the voxel being considered.
	//ind_x = nbatchIDx*((int)(IMGSIZx/h_nBatchXdim))+ tid;
	ind_x = tid;
	ind_y = bx;
	ind_z = by;

	ind_voxel=(ind_z*IMGSIZy+ind_y)*IMGSIZx+ind_x;  //(if prj is scanner data, need x_y_flip)
	//ind_voxel=(ind_z*IMGSIZx+ind_x)*IMGSIZy+ind_y;		

	for ( ind_nr_z = (ind_z-1);ind_nr_z < (ind_z+2); ind_nr_z++ )
	{
		for ( ind_nr_y = (ind_y-1); ind_nr_y < (ind_y+2); ind_nr_y++ )
		{
			for( ind_nr_x = (ind_x-1); ind_nr_x < (ind_x+2); ind_nr_x++)
			{


				distance = sqrt(float((ind_nr_x-ind_x)*(ind_nr_x-ind_x)+(ind_nr_y-ind_y)*(ind_nr_y-ind_y)+(ind_nr_z-ind_z)*(ind_nr_z-ind_z)));

				if(distance==0.0)
					distance=1.0;

				if(ind_nr_x<0 | ind_nr_y<0 | ind_nr_z<0 | ind_nr_x>(IMGSIZx-1) | ind_nr_y>(IMGSIZy-1) | ind_nr_z>(IMGSIZz-1))
					ind_nr = ind_voxel;
				else
					ind_nr = ind_nr_x+ind_nr_y*IMGSIZx+ind_nr_z*IMGSIZx*IMGSIZy; 

				diff=estbuf[ind_voxel]-estbuf[ind_nr];
				denominator = sqrt(1.0+(diff/delta)*(diff/delta));

				priorbuf[ind_voxel]=priorbuf[ind_voxel] + (1.0/distance)*diff/denominator;

			}// ind_nr_x loop
		}//ind_nr_y loop
	}//ind_nr_z loop




}

__device__ void rayTrace3D_GPU_direct_notexturememory(
	float *d_objbuf, 
	float x0, 
	float y0, 
	float z0, 
	float x1, 
	float y1, 
	float z1, 
	int *status, 
	float *fsum, 
	float *fsum_norm)
{

	float Length, s_temp;
	float	min_lx, max_lx, min_ly, max_ly, min_lz, max_lz;
	float	min_l, max_l, min_l_new;
	int		ind;
	float	dx,dy,dz;
	float	sum, sum_norm;

	int		prev_x, prev_y, prev_z;
	float	tmp_length;

	*status	= -1;
	sum		= 0;
	sum_norm= 0;

	dx=x1-x0;
	dy=y1-y0;
	dz=z1-z0;
	Length=sqrt( dx*dx+dy*dy+dz*dz );
	if (x1!=x0)
	{
		min_lx=(x_p0-x0)/dx;  //x_p0 is the location of 0th pixel in x axis
		max_lx=min_lx+IMGSIZx*Vsize_x/dx;
		if (min_lx>max_lx)
		{
			//SWAP(min_lx, max_lx);
			s_temp = min_lx;
			min_lx = max_lx;
			max_lx = s_temp;
		}
	}
	else 
	{
		// the line perpendicular to x axis
		if (x0 >= IMGSIZx*Vsize_x+x_p0 || x0<=x_p0)
		{
			*status = -1; return;
		}
		min_lx=-1e3;
		max_lx=1e3;
	}


	if (y0 != y1)
	{
		min_ly=(y_p0-y0)/dy;
		max_ly=min_ly+IMGSIZy*Vsize_y/dy;
		if (min_ly>max_ly)
		{
			//SWAP(min_ly, max_ly);
			s_temp = min_ly;
			min_ly = max_ly;
			max_ly = s_temp;

		}
	}
	else 
	{
		// the line perpendicular to y axis
		if (y0 >= IMGSIZy*Vsize_y+y_p0 || y0<=y_p0)
		{
			*status = -1; return;
		}
		min_ly= -1e3;
		max_ly=1e3;
	}

	if (z0 != z1)
	{
		min_lz=(z_p0-z0)/dz;
		max_lz=min_lz+IMGSIZz*Vsize_z/dz;
		if (min_lz>max_lz)
		{
			//SWAP(min_lz, max_lz);
			s_temp = min_lz;
			min_lz = max_lz;
			max_lz = s_temp;

		}
	}
	else 
	{
		// the line perpendicular to z axis
		if (z0 >= IMGSIZz*Vsize_z+z_p0 || z0<=z_p0)
		{
			*status = -1; return;
		}
		min_lz = -1e3;
		max_lz = 1e3;
	}

	max_l = max_lx;
	if (max_l > max_ly) max_l=max_ly;
	if (max_l > max_lz) max_l=max_lz;
	min_l = min_lx;
	if (min_l < min_ly) min_l=min_ly;
	if (min_l < min_lz) min_l=min_lz;

	if (min_l>=max_l) 
	{
		*status = -1; return;
	}


	if (min_lx != min_l) 
	{
		prev_x=(short)floorf( (min_l* dx + x0 - x_p0) / Vsize_x );
		if (x0<x1)
			min_lx= ((prev_x+1)*Vsize_x+x_p0-x0)/ dx;
		else 
			if 
				(x0==x1) min_lx=1e3;
			else min_lx=(prev_x*Vsize_x+x_p0-x0) / dx;
	}
	else
	{
		if (x0<x1) 
		{
			prev_x= 0;
			min_lx = ( Vsize_x+x_p0-x0 )/ dx;
		}
		else 
		{
			prev_x = IMGSIZx-1;
			min_lx = ( prev_x*Vsize_x+x_p0-x0 )/ dx;
			//in case of the ray is on plane x=0;
		}
	}


	if (min_ly != min_l) 
	{
		prev_y=(short)floorf( (min_l* dy + y0 - y_p0)/Vsize_y );
		if (y0<y1) 
			min_ly= ( (prev_y+1)*Vsize_y+y_p0-y0)/ dy;
		else 
			if (y0==y1) 
				min_ly=1e3;
			else 
				min_ly=(prev_y*Vsize_y+y_p0-y0)/ dy;
	}
	else
	{
		if (y0<y1) 
		{
			prev_y=0;
			min_ly = ( Vsize_y+y_p0-y0 )/ dy;
		}
		else 
		{
			prev_y = IMGSIZy-1;
			min_ly = ( prev_y*Vsize_y+y_p0-y0 )/ dy;
		}
	}


	if (min_lz != min_l) 
	{
		prev_z=(short)floorf( (min_l* dz + z0 - z_p0)/Vsize_z );
		if (z0<z1) 
			min_lz= ( (prev_z+1)*Vsize_z+z_p0-z0)/ dz;
		else 
			if (z0==z1) 
				min_lz=1e3;
			else 
				min_lz=(prev_z*Vsize_z+z_p0-z0)/ dz;
	}
	else
	{
		if (z0<z1) 
		{
			prev_z= 0;
			min_lz = ( Vsize_z+z_p0-z0 )/ dz;
		}
		else 
		{
			prev_z =(short)IMGSIZz-1;
			min_lz = ( prev_z*Vsize_z+z_p0-z0 )/dz;
		}
	}


	ind=0;
	min_l_new=min_lx;
	if (min_l_new>min_ly) min_l_new=min_ly;
	if (min_l_new>min_lz) min_l_new=min_lz;

	float	incx, incy, incz;
	incx = Vsize_x/dx;
	incy = Vsize_y/dy;
	incz = Vsize_z/dz;


	while ( (max_l-min_l_new)/max_l>0.000001)
	{
		tmp_length = (min_l_new-min_l)*Length;	//<-a_ij

		if ((prev_x>=0)&&(prev_x<IMGSIZx)&&(prev_y>=0)&&(prev_y<IMGSIZy)&&(prev_z>=0)&&(prev_z<IMGSIZz))
		{				
			sum			= sum		+ (d_objbuf[(prev_z*IMGSIZy+prev_y)*IMGSIZx+prev_x]*tmp_length);
			sum_norm	= sum_norm	+ tmp_length;
		}

		ind++;
		if (min_l_new == min_lx) 
		{
			if (x0<x1) 
			{
				prev_x = prev_x + 1;
				min_lx	= min_lx + incx;//Vsize_x/dx;
			}
			else 
			{
				prev_x = prev_x - 1;
				min_lx = min_lx - incx;//Vsize_x/dx;
			}
		}
		else 
			prev_x = prev_x;



		if (min_l_new == min_ly) 
		{
			if (y0<y1) 
			{
				prev_y = prev_y + 1;
				min_ly = min_ly + incy;//Vsize_y / dy;
			}
			else 
			{
				prev_y = prev_y - 1;
				min_ly = min_ly- incy;//Vsize_y/dy;
			}
		}
		else 
			prev_y = prev_y;



		if (min_l_new == min_lz) 
		{
			if (z0<z1) 
			{
				prev_z = prev_z + 1;
				min_lz = min_lz + incz;//Vsize_z/dz;
			}
			else 
			{
				prev_z = prev_z - 1;
				min_lz = min_lz - incz;//Vsize_z/dz;
			}
		}
		else 
			prev_z = prev_z ;

		min_l = min_l_new;
		min_l_new=min_lx;
		if (min_l_new>min_ly) min_l_new=min_ly;
		if (min_l_new>min_lz) min_l_new=min_lz;
	} //End of while

	tmp_length=(max_l-min_l)*Length;
	if ((prev_x>=0)&&(prev_x<IMGSIZx)&&(prev_y>=0)&&(prev_y<IMGSIZy)&&(prev_z>=0)&&(prev_z<IMGSIZz))
	{		
		sum			= sum		+ d_objbuf[(prev_z*IMGSIZy+prev_y)*IMGSIZx+prev_x]*tmp_length;
		sum_norm	= sum_norm	+ tmp_length;
	}
	*status = 1;
	*fsum = sum;
	*fsum_norm = sum_norm;
};

__device__ void rayTrace3D_GPU_direct_notexturememory_OSTR(
	float *d_objbuf, 
	float x0, 
	float y0, 
	float z0, 
	float x1, 
	float y1, 
	float z1, 
	int *status, 
	float *fsum)
{

	float Length, s_temp;
	float	min_lx, max_lx, min_ly, max_ly, min_lz, max_lz;
	float	min_l, max_l, min_l_new;
	int		ind;
	float	dx,dy,dz;
	float	sum;

	int		prev_x, prev_y, prev_z;
	float	tmp_length;

	*status	= -1;
	sum		= 0;

	dx=x1-x0;
	dy=y1-y0;
	dz=z1-z0;
	Length=sqrt( dx*dx+dy*dy+dz*dz );
	if (x1!=x0)
	{
		min_lx=(x_p0-x0)/dx;  //x_p0 is the location of 0th pixel in x axis
		max_lx=min_lx+IMGSIZx*Vsize_x/dx;
		if (min_lx>max_lx)
		{
			//SWAP(min_lx, max_lx);
			s_temp = min_lx;
			min_lx = max_lx;
			max_lx = s_temp;
		}
	}
	else 
	{
		// the line perpendicular to x axis
		if (x0 >= IMGSIZx*Vsize_x+x_p0 || x0<=x_p0)
		{
			*status = -1; return;
		}
		min_lx=-1e3;
		max_lx=1e3;
	}


	if (y0 != y1)
	{
		min_ly=(y_p0-y0)/dy;
		max_ly=min_ly+IMGSIZy*Vsize_y/dy;
		if (min_ly>max_ly)
		{
			//SWAP(min_ly, max_ly);
			s_temp = min_ly;
			min_ly = max_ly;
			max_ly = s_temp;

		}
	}
	else 
	{
		// the line perpendicular to y axis
		if (y0 >= IMGSIZy*Vsize_y+y_p0 || y0<=y_p0)
		{
			*status = -1; return;
		}
		min_ly= -1e3;
		max_ly=1e3;
	}

	if (z0 != z1)
	{
		min_lz=(z_p0-z0)/dz;
		max_lz=min_lz+IMGSIZz*Vsize_z/dz;
		if (min_lz>max_lz)
		{
			//SWAP(min_lz, max_lz);
			s_temp = min_lz;
			min_lz = max_lz;
			max_lz = s_temp;

		}
	}
	else 
	{
		// the line perpendicular to z axis
		if (z0 >= IMGSIZz*Vsize_z+z_p0 || z0<=z_p0)
		{
			*status = -1; return;
		}
		min_lz = -1e3;
		max_lz = 1e3;
	}

	max_l = max_lx;
	if (max_l > max_ly) max_l=max_ly;
	if (max_l > max_lz) max_l=max_lz;
	min_l = min_lx;
	if (min_l < min_ly) min_l=min_ly;
	if (min_l < min_lz) min_l=min_lz;

	if (min_l >= max_l) 
	{
		*status = -1; return;
	}


	if (min_lx != min_l) 
	{
		prev_x=(short)floorf( (min_l* dx + x0 - x_p0) / Vsize_x );
		if (x0<x1)
			min_lx= ((prev_x+1)*Vsize_x+x_p0-x0)/ dx;
		else 
			if 
				(x0==x1) min_lx=1e3;
			else min_lx=(prev_x*Vsize_x+x_p0-x0) / dx;
	}
	else
	{
		if (x0<x1) 
		{
			prev_x= 0;
			min_lx = ( Vsize_x+x_p0-x0 )/ dx;
		}
		else 
		{
			prev_x = IMGSIZx-1;
			min_lx = ( prev_x*Vsize_x+x_p0-x0 )/ dx;
			//in case of the ray is on plane x=0;
		}
	}


	if (min_ly != min_l) 
	{
		prev_y=(short)floorf( (min_l* dy + y0 - y_p0)/Vsize_y );
		if (y0<y1) 
			min_ly= ( (prev_y+1)*Vsize_y+y_p0-y0)/ dy;
		else 
			if (y0==y1) 
				min_ly=1e3;
			else 
				min_ly=(prev_y*Vsize_y+y_p0-y0)/ dy;
	}
	else
	{
		if (y0<y1) 
		{
			prev_y=0;
			min_ly = ( Vsize_y+y_p0-y0 )/ dy;
		}
		else 
		{
			prev_y = IMGSIZy-1;
			min_ly = ( prev_y*Vsize_y+y_p0-y0 )/ dy;
		}
	}


	if (min_lz != min_l) 
	{
		prev_z=(short)floorf( (min_l* dz + z0 - z_p0)/Vsize_z );
		if (z0<z1) 
			min_lz= ( (prev_z+1)*Vsize_z+z_p0-z0)/ dz;
		else 
			if (z0==z1) 
				min_lz=1e3;
			else 
				min_lz=(prev_z*Vsize_z+z_p0-z0)/ dz;
	}
	else
	{
		if (z0<z1) 
		{
			prev_z= 0;
			min_lz = ( Vsize_z+z_p0-z0 )/ dz;
		}
		else 
		{
			prev_z =(short)IMGSIZz-1;
			min_lz = ( prev_z*Vsize_z+z_p0-z0 )/dz;
		}
	}


	ind=0;
	min_l_new=min_lx;
	if (min_l_new>min_ly) min_l_new=min_ly;
	if (min_l_new>min_lz) min_l_new=min_lz;

	float	incx, incy, incz;
	incx = Vsize_x/dx;
	incy = Vsize_y/dy;
	incz = Vsize_z/dz;


	while ( (max_l-min_l_new)/max_l>0.000001)
	{
		tmp_length = (min_l_new-min_l)*Length;	//<-a_ij

		if ((prev_x>=0)&&(prev_x<IMGSIZx)&&(prev_y>=0)&&(prev_y<IMGSIZy)&&(prev_z>=0)&&(prev_z<IMGSIZz))
		{				
			sum			= sum		+ (d_objbuf[(prev_z*IMGSIZy+prev_y)*IMGSIZx+prev_x]*tmp_length);
		}

		ind++;
		if (min_l_new == min_lx) 
		{
			if (x0<x1) 
			{
				prev_x = prev_x + 1;
				min_lx	= min_lx + incx;//Vsize_x/dx;
			}
			else 
			{
				prev_x = prev_x - 1;
				min_lx = min_lx - incx;//Vsize_x/dx;
			}
		}
		else 
			prev_x = prev_x;



		if (min_l_new == min_ly) 
		{
			if (y0<y1) 
			{
				prev_y = prev_y + 1;
				min_ly = min_ly + incy;//Vsize_y / dy;
			}
			else 
			{
				prev_y = prev_y - 1;
				min_ly = min_ly- incy;//Vsize_y/dy;
			}
		}
		else 
			prev_y = prev_y;



		if (min_l_new == min_lz) 
		{
			if (z0<z1) 
			{
				prev_z = prev_z + 1;
				min_lz = min_lz + incz;//Vsize_z/dz;
			}
			else 
			{
				prev_z = prev_z - 1;
				min_lz = min_lz - incz;//Vsize_z/dz;
			}
		}
		else 
			prev_z = prev_z ;

		min_l = min_l_new;
		min_l_new=min_lx;
		if (min_l_new>min_ly) min_l_new=min_ly;
		if (min_l_new>min_lz) min_l_new=min_lz;
	} //End of while

	tmp_length=(max_l-min_l)*Length;
	if ((prev_x>=0)&&(prev_x<IMGSIZx)&&(prev_y>=0)&&(prev_y<IMGSIZy)&&(prev_z>=0)&&(prev_z<IMGSIZz))
	{		
		sum			= sum		+ d_objbuf[(prev_z*IMGSIZy+prev_y)*IMGSIZx+prev_x]*tmp_length;
	}
	*status = 1;
	*fsum = sum;
};

__device__ void rayTrace3D_GPU_direct_notexturememory_normprj(
	float x0, 
	float y0, 
	float z0, 
	float x1, 
	float y1, 
	float z1, 
	int *status, 
	float *fsum_norm)
{

	float   Length, s_temp;
	float	min_lx, max_lx, min_ly, max_ly, min_lz, max_lz;
	float	min_l, max_l, min_l_new;
	int		ind;
	float	dx,dy,dz;
	float	sum_norm;

	int	prev_x, prev_y, prev_z;
	float	tmp_length;

	*status	 = -1;
	sum_norm = 0;

	dx=x1-x0;
	dy=y1-y0;
	dz=z1-z0;
	Length=sqrt( dx*dx+dy*dy+dz*dz );
	if (x1!=x0)
	{
		min_lx=(x_p0-x0)/dx;  //x_p0 is the location of 0th pixel in x axis
		max_lx=min_lx+IMGSIZx*Vsize_x/dx;
		if (min_lx>max_lx)
		{
			//SWAP(min_lx, max_lx);
			s_temp = min_lx;
			min_lx = max_lx;
			max_lx = s_temp;
		}
	}
	else 
	{
		// the line perpendicular to x axis
		if (x0 >= IMGSIZx*Vsize_x+x_p0 || x0<=x_p0)
		{
			*status = -1; return;
		}
		min_lx=-1e3;
		max_lx=1e3;
	}


	if (y0 != y1)
	{
		min_ly=(y_p0-y0)/dy;
		max_ly=min_ly+IMGSIZy*Vsize_y/dy;
		if (min_ly>max_ly)
		{
			//SWAP(min_ly, max_ly);
			s_temp = min_ly;
			min_ly = max_ly;
			max_ly = s_temp;

		}
	}
	else 
	{
		// the line perpendicular to y axis
		if (y0 >= IMGSIZy*Vsize_y+y_p0 || y0<=y_p0)
		{
			*status = -1; return;
		}
		min_ly= -1e3;
		max_ly=1e3;
	}

	if (z0 != z1)
	{
		min_lz=(z_p0-z0)/dz;
		max_lz=min_lz+IMGSIZz*Vsize_z/dz;
		if (min_lz>max_lz)
		{
			//SWAP(min_lz, max_lz);
			s_temp = min_lz;
			min_lz = max_lz;
			max_lz = s_temp;

		}
	}
	else 
	{
		// the line perpendicular to z axis
		if (z0 >= IMGSIZz*Vsize_z+z_p0 || z0<=z_p0)
		{
			*status = -1; return;
		}
		min_lz = -1e3;
		max_lz = 1e3;
	}

	max_l = max_lx;
	if (max_l > max_ly) max_l=max_ly;
	if (max_l > max_lz) max_l=max_lz;
	min_l = min_lx;
	if (min_l < min_ly) min_l=min_ly;
	if (min_l < min_lz) min_l=min_lz;

	if (min_l>=max_l) 
	{
		*status = -1; return;
	}


	if (min_lx != min_l) 
	{
		prev_x=(short)floorf( (min_l* dx + x0 - x_p0) / Vsize_x );
		if (x0<x1)
			min_lx= ((prev_x+1)*Vsize_x+x_p0-x0)/ dx;
		else 
			if 
				(x0==x1) min_lx=1e3;
			else min_lx=(prev_x*Vsize_x+x_p0-x0) / dx;
	}
	else
	{
		if (x0<x1) 
		{
			prev_x= 0;
			min_lx = ( Vsize_x+x_p0-x0 )/ dx;
		}
		else 
		{
			prev_x = IMGSIZx-1;
			min_lx = ( prev_x*Vsize_x+x_p0-x0 )/ dx;
			//in case of the ray is on plane x=0;
		}
	}


	if (min_ly != min_l) 
	{
		prev_y=(short)floorf( (min_l* dy + y0 - y_p0)/Vsize_y );
		if (y0<y1) 
			min_ly= ( (prev_y+1)*Vsize_y+y_p0-y0)/ dy;
		else 
			if (y0==y1) 
				min_ly=1e3;
			else 
				min_ly=(prev_y*Vsize_y+y_p0-y0)/ dy;
	}
	else
	{
		if (y0<y1) 
		{
			prev_y=0;
			min_ly = ( Vsize_y+y_p0-y0 )/ dy;
		}
		else 
		{
			prev_y = IMGSIZy-1;
			min_ly = ( prev_y*Vsize_y+y_p0-y0 )/ dy;
		}
	}


	if (min_lz != min_l) 
	{
		prev_z=(short)floorf( (min_l* dz + z0 - z_p0)/Vsize_z );
		if (z0<z1) 
			min_lz= ( (prev_z+1)*Vsize_z+z_p0-z0)/ dz;
		else 
			if (z0==z1) 
				min_lz=1e3;
			else 
				min_lz=(prev_z*Vsize_z+z_p0-z0)/ dz;
	}
	else
	{
		if (z0<z1) 
		{
			prev_z= 0;
			min_lz = ( Vsize_z+z_p0-z0 )/ dz;
		}
		else 
		{
			prev_z =(short)IMGSIZz-1;
			min_lz = ( prev_z*Vsize_z+z_p0-z0 )/dz;
		}
	}


	ind=0;
	min_l_new=min_lx;
	if (min_l_new>min_ly) min_l_new=min_ly;
	if (min_l_new>min_lz) min_l_new=min_lz;

	float	incx, incy, incz;
	incx = Vsize_x/dx;
	incy = Vsize_y/dy;
	incz = Vsize_z/dz;


	while ( (max_l-min_l_new)/max_l>0.000001)
	{
		tmp_length = (min_l_new-min_l)*Length;	//<-a_ij

		if ((prev_x>=0)&&(prev_x<IMGSIZx)&&(prev_y>=0)&&(prev_y<IMGSIZy)&&(prev_z>=0)&&(prev_z<IMGSIZz))
		{				
			sum_norm	= sum_norm	+ 1*tmp_length;
		}

		ind++;
		if (min_l_new == min_lx) 
		{
			if (x0<x1) 
			{
				prev_x = prev_x + 1;
				min_lx	= min_lx + incx;//Vsize_x/dx;
			}
			else 
			{
				prev_x = prev_x - 1;
				min_lx = min_lx - incx;//Vsize_x/dx;
			}
		}
		else 
			prev_x = prev_x;



		if (min_l_new == min_ly) 
		{
			if (y0<y1) 
			{
				prev_y = prev_y + 1;
				min_ly = min_ly + incy;//Vsize_y / dy;
			}
			else 
			{
				prev_y = prev_y - 1;
				min_ly = min_ly- incy;//Vsize_y/dy;
			}
		}
		else 
			prev_y = prev_y;



		if (min_l_new == min_lz) 
		{
			if (z0<z1) 
			{
				prev_z = prev_z + 1;
				min_lz = min_lz + incz;//Vsize_z/dz;
			}
			else 
			{
				prev_z = prev_z - 1;
				min_lz = min_lz - incz;//Vsize_z/dz;
			}
		}
		else 
			prev_z = prev_z ;

		min_l     = min_l_new;
		min_l_new = min_lx;

		if (min_l_new>min_ly) 
			min_l_new = min_ly;
		if (min_l_new>min_lz) 
			min_l_new = min_lz;

	} //End of while

	tmp_length = (max_l-min_l)*Length;
	if ((prev_x >= 0)&&(prev_x < IMGSIZx)&&(prev_y >= 0)&&(prev_y < IMGSIZy)&&(prev_z >= 0)&&(prev_z < IMGSIZz))
	{		
		sum_norm	= sum_norm	+ 1*tmp_length;
	}
	*status    = 1;
	*fsum_norm = Length;//sum_norm;
};



__global__ void ray_trace_gpu_manyangles_direct_notexturememory(
	float *d_objbuf, 
	float *d_prjbuf, 
	float *d_normprj, 
	float *d_angles, 
	int *d_index, 
	int angleStart, 
	int angleEnd, 
	int nbBinsX, 
	int nbBinsY)
{	
	const int ix  = blockIdx.x * blockDim.x + threadIdx.x; 
	const int iy  = blockIdx.y * blockDim.y + threadIdx.y;

	int	i, j;
	int	bin_ind;

	float	x1, y1, z1, x0,z0;;

	float	theta, sin_theta,cos_theta;
	float	sum;
	float   sum_norm;
	int		status;
	int		a, s;

	float bin_x_pos, bin_y_pos;

	for (a=angleStart; a<angleEnd; a++)
	{
		s = d_index[a];
		theta	= d_angles[s];
		sin_theta=sinf(theta);
		cos_theta=cosf(theta);
		x0=sourceR*sin_theta;
		z0=sourceR*cos_theta;

		//calculate bin index.
		i = nbBinsX*((int)BINSx/nBatchBINSx) + ix;
		j = nbBinsY*((int)BINSy/nBatchBINSy) + iy;

		bin_x_pos =	(x_d0+(i+0.5)*Bsize_x);			
		bin_y_pos = (y_d0+(j+0.5)*Bsize_y);

		//Calculate spatial coordinate of the center of this detector bin.
		// Stationary version
		x1			=	 bin_x_pos;
		z1			=	-detectorR;
		y1			=	bin_y_pos;

		// Iso-centric version
		/*x1			=	 bin_x_pos*cos_theta-detectorR*sin_theta;
		z1			=	-bin_x_pos*sin_theta-detectorR*cos_theta;
		y1			=	bin_y_pos;
		*/
		bin_ind=((a-angleStart)*BINSx+i)*BINSy+j;	//Bin index.
		//bin_ind=((a-angleStart)*BINSy+j)*BINSx+i;	//Bin index. Need x_y_flip() in BP. Output's coor-system will be as same as TOMO scanner (NOT runnable)

		rayTrace3D_GPU_direct_notexturememory(d_objbuf, x0, sourceY, z0, x1, y1, z1, &status, &sum, &sum_norm);
		if (status!=-1)
		{
			//We compute both forward projection of the estimating object and forward projection of the uniform object.
			d_prjbuf	[bin_ind] = sum;
			d_normprj   [bin_ind]= sum_norm;
		}
	}
	__syncthreads();	//This line can be removed.
}

__global__ void ray_trace_gpu_manyangles_direct_notexturememory_cos(
	float *d_objbuf, 
	float *d_prjbuf, 
	float *d_normprj, 
	float *d_angles, 
	int *d_index, 
	int angleStart, 
	int angleEnd, 
	int nbBinsX, 
	int nbBinsY)
{	
	const int ix  = blockIdx.x * blockDim.x + threadIdx.x; 
	const int iy  = blockIdx.y * blockDim.y + threadIdx.y;

	int	i, j;
	int	bin_ind;

	float	x1, y1, z1, x0,z0;;

	float	theta, sin_theta,cos_theta;
	float	sum;
	float   sum_norm;
	int		status;
	int		a, s;

	float bin_x_pos, bin_y_pos;

	for (a=angleStart; a<angleEnd; a++)
	{
		s = d_index[a];
		theta	= d_angles[s];
		sin_theta=sinf(theta);
		cos_theta=cosf(theta);
		x0=sourceR*sin_theta;
		z0=sourceR*cos_theta;

		//calculate bin index.
		i = nbBinsX*((int)BINSx/nBatchBINSx) + ix;
		j = nbBinsY*((int)BINSy/nBatchBINSy) + iy;

		bin_x_pos =	(x_d0+(i+0.5)*Bsize_x);			
		bin_y_pos = (y_d0+(j+0.5)*Bsize_y);

		//Calculate spatial coordinate of the center of this detector bin.
		// Stationary version
		x1			=	 bin_x_pos;
		z1			=	-detectorR;
		y1			=	bin_y_pos;

		// Iso-centric version
		/*x1			=	 bin_x_pos*cos_theta-detectorR*sin_theta;
		z1			=	-bin_x_pos*sin_theta-detectorR*cos_theta;
		y1			=	bin_y_pos;
		*/
		bin_ind=((a-angleStart)*BINSx+i)*BINSy+j;	//Bin index.
		//bin_ind=((a-angleStart)*BINSy+j)*BINSx+i;	//Bin index. Need x_y_flip() in BP. Output's coor-system will be as same as TOMO scanner (NOT runnable)

		rayTrace3D_GPU_direct_notexturememory(d_objbuf, x0, sourceY, z0, x1, y1, z1, &status, &sum, &sum_norm);
		if (status!=-1)
		{
			//We compute both forward projection of the estimating object and forward projection of the uniform object.
			d_prjbuf	[bin_ind] = sum*cos_theta;
			d_normprj   [bin_ind]= sum_norm;
		}
	}
	__syncthreads();	//This line can be removed.
}

__global__ void ray_trace_gpu_manyangles_direct_notexturememory_OSTR_cos(
	float *d_objbuf, 
	float *d_prjbuf,  
	float *d_angles, 
	int *d_index, 
	int angleStart, 
	int angleEnd, 
	int nbBinsX, 
	int nbBinsY)
{	
	const int ix  = blockIdx.x * blockDim.x + threadIdx.x; 
	const int iy  = blockIdx.y * blockDim.y + threadIdx.y;

	int	i, j;
	int	bin_ind;

	float	x1, y1, z1, x0,z0;;

	float	theta, sin_theta,cos_theta;
	float	sum;
	int		status;
	int		a, s;

	float bin_x_pos, bin_y_pos;

	for (a=angleStart; a<angleEnd; a++)
	{
		s = d_index[a];
		theta	= d_angles[s];
		sin_theta=sinf(theta);
		cos_theta=cosf(theta);
		x0=sourceR*sin_theta;
		z0=sourceR*cos_theta;

		//calculate bin index.
		i = nbBinsX*((int)BINSx/nBatchBINSx) + ix;
		j = nbBinsY*((int)BINSy/nBatchBINSy) + iy;

		bin_x_pos =	(x_d0+(i+0.5)*Bsize_x);			
		bin_y_pos = (y_d0+(j+0.5)*Bsize_y);

		//Calculate spatial coordinate of the center of this detector bin.
		// Stationary version
		x1			=	 bin_x_pos;
		z1			=	-detectorR;
		y1			=	bin_y_pos;

		// Iso-centric version
		/*x1			=	 bin_x_pos*cos_theta-detectorR*sin_theta;
		z1			=	-bin_x_pos*sin_theta-detectorR*cos_theta;
		y1			=	bin_y_pos;
		*/
		bin_ind=((a-angleStart)*BINSx+i)*BINSy+j;	//Bin index.
		//bin_ind=((a-angleStart)*BINSy+j)*BINSx+i;	//Bin index. Need x_y_flip() in BP. Output's coor-system will be as same as TOMO scanner (NOT runnable)

		rayTrace3D_GPU_direct_notexturememory_OSTR(d_objbuf, x0, sourceY, z0, x1, y1, z1, &status, &sum);
		if (status!=-1)
		{
			//We compute both forward projection of the estimating object and forward projection of the uniform object.
			d_prjbuf	[bin_ind] = sum*cos_theta;
		}
	}
	__syncthreads();	//This line can be removed.
}

__global__ void ray_trace_gpu_manyangles_direct_notexturememory_normprj(
	float *d_normprj, 
	float *d_angles, 
	int *d_index, 
	int angleStart, 
	int angleEnd, 
	int nbBinsX, 
	int nbBinsY)
{	




	const int ix  = blockIdx.x * blockDim.x + threadIdx.x; 
	const int iy  = blockIdx.y * blockDim.y + threadIdx.y;

	int	i, j;
	int	bin_ind;

	float	x1, y1, z1, x0,z0;;

	float	theta, sin_theta,cos_theta;
	float   sum_norm;
	int		status;
	int		a, s;

	float bin_x_pos, bin_y_pos;

	for (a=angleStart; a<angleEnd; a++)
	{

		s = d_index[a];
		theta	= d_angles[s];
		sin_theta=sinf(theta);
		cos_theta=cosf(theta);
		x0=sourceR*sin_theta;
		z0=sourceR*cos_theta;

		//calculate bin index.
		i = nbBinsX*((int)BINSx/nBatchBINSx) + ix;
		j = nbBinsY*((int)BINSy/nBatchBINSy) + iy;

		bin_x_pos = (x_d0+(i+0.5)*Bsize_x);			
		bin_y_pos = (y_d0+(j+0.5)*Bsize_y);

		//Calculate spatial coordinate of the center of this detector bin.
		// Stationary version
		x1			=	 bin_x_pos;
		z1			=	-detectorR;
		y1			=	bin_y_pos;

		// Iso-centric version
		/*x1			=	 bin_x_pos*cos_theta-detectorR*sin_theta;
		z1			=	-bin_x_pos*sin_theta-detectorR*cos_theta;
		y1			=	bin_y_pos;
		*/
		bin_ind=((a-angleStart)*BINSx+i)*BINSy+j;	//Bin index.
		//bin_ind=((a-angleStart)*BINSy+j)*BINSx+i;	//Bin index. Need x_y_flip() in BP. Output's coor-system will be as same as TOMO scanner (NOT runnable)

		
		rayTrace3D_GPU_direct_notexturememory_normprj(x0, sourceY, z0, x1, y1, z1, &status, &sum_norm);

		
		if (status !=-1)
		{
			//We compute both forward projection of the estimating object and forward projection of the uniform object.
			d_normprj[bin_ind] = sum_norm;
                        //printf("%d %f", bin_ind, sum_norm);
		}
	}
	//__syncthreads();	//This line can be removed.
}



__global__ void ray_trace_4R_gpu_manyangles_direct_notexturememory(/// need modification
	float *d_objbuf, 
	float *d_4R_prjbuf, 
	float *d_angles, 
	int *d_index, 
	int angleStart, 
	int angleEnd, 
	int nbBinsX, 
	int nbBinsY)
{	
	const int ix  = blockIdx.x * blockDim.x + threadIdx.x; 
	const int iy  = blockIdx.y * blockDim.y + threadIdx.y;

	int	i, j;
	int	bin_ind;

	float	x1, y1, z1, x0,z0;;

	float	theta, sin_theta,cos_theta;
	float	sum,sum_norm;
	int		status;
	int		a, s;

	float bin_x_pos, bin_y_pos;

	for (a=angleStart; a<angleEnd; a++){

		s = d_index[a];
		theta	= d_angles[s];
		sin_theta=sinf(theta);
		cos_theta=cosf(theta);
		x0=sourceR*sin_theta;
		z0=sourceR*cos_theta;

		//calculate bin index.
		i = nbBinsX*((int)2*BINSx/nBatchBINSx_4R) + ix;
		j = nbBinsY*((int)2*BINSy/nBatchBINSy_4R) + iy;

		bin_x_pos =	(x_d0+(i+0.5)/2*Bsize_x);			
		bin_y_pos = (y_d0+(j+0.5)/2*Bsize_y);

		//Calculate spatial coordinate of the center of this detector bin.
		x1			=	 bin_x_pos;
		z1			=	-detectorR;
		y1			=	bin_y_pos;

		/*x1			=	 bin_x_pos*cos_theta-detectorR*sin_theta;
		z1			=	-bin_x_pos*sin_theta-detectorR*cos_theta;
		y1			=	bin_y_pos;
		*/

		//bin_ind=((a-angleStart)*BINSx*2+i)*BINSy*2+j;	//Bin index. Do NOT need x_y_flip() in BP
		bin_ind=((a-angleStart)*BINSy*2+j)*BINSx*2+i;	//Bin index. Need x_y_flip() in BP. Output's coor-system will be as same as TOMO scanner (NOT runnable)

		rayTrace3D_GPU_direct_notexturememory(d_objbuf, x0, sourceY, z0, x1, y1, z1, &status, &sum,&sum_norm);
		if (status!=-1){

			//We compute both forward projection of the estimating object and forward projection of the uniform object.
			d_4R_prjbuf	[bin_ind] = sum;
		}
	}
	//__syncthreads();	//This line can be removed.
}

__global__ void merge_4R_to_1R_global(
	float *d_prjbuf, 
	float *d_4R_prjbuf, 
	float *d_angles, 
	int *d_index, 
	int angleStart, 
	int angleEnd)
{
	const int ix  = blockIdx.x * blockDim.x + threadIdx.x; 
	const int iy  = blockIdx.y * blockDim.y + threadIdx.y;

	

	int	a, i, j, r;
	int i_4R, j_4R;
	int	bin_ind, bin_ind_4R_1,bin_ind_4R_2,bin_ind_4R_3,bin_ind_4R_4;

	i = ix;
	j = iy;

	i_4R = i*2;
	j_4R = j*2;

	for (a=angleStart; a<angleEnd; a++){

		bin_ind=((a-angleStart)*BINSy+j)*BINSx+i;	//Bin index. Need x_y_flip() in BP. Output's coor-system will be as same as TOMO scanner (NOT runnable)
		bin_ind_4R_1=((a-angleStart)*BINSy*2+j_4R)*BINSx*2+i_4R;
		bin_ind_4R_2=((a-angleStart)*BINSy*2+j_4R)*BINSx*2+(i_4R+1);
		bin_ind_4R_3=((a-angleStart)*BINSy*2+(j_4R+1) )*BINSx*2+i_4R;
		bin_ind_4R_3=((a-angleStart)*BINSy*2+(j_4R+1) )*BINSx*2+ (i_4R+1);
		d_prjbuf[bin_ind]=(d_4R_prjbuf[bin_ind_4R_1]+d_4R_prjbuf[bin_ind_4R_2]+d_4R_prjbuf[bin_ind_4R_3]+d_4R_prjbuf[bin_ind_4R_4])/4;
	}

}


__global__ void backprj_gpu_manyviews_SBP(
	float *d_objbuf,
	float *d_prjbuf,
	int *d_index,
	float *d_angles,
	int angleStart,
	int angleEnd, 
	int nbatchIDx)
{	
	// Block index
    	const int bx = blockIdx.x;
    	const int by = blockIdx.y;

    	// Thread index
	// const int tx = threadIdx.x;
    	const int tx = threadIdx.x+(nbatchIDx*blockDim.x);
    	const int ty = threadIdx.y;
    
    	// const int tid = tx * blockDim.y + ty;
	const int tid=tx;//+ty*blockDim.x;//+nbatchIDx*blockDim.x;//*blockDim.y;  


	int		ind_x, ind_y, ind_z;
	int		bin_ind;
	long	ind_voxel;

	float	x1, y1, z1, x0, y0, z0;
	float	x1r, y1r, z1r, x0r, y0r, z0r;	//Rotated locations of (x1,y1,z1) and (x0,y0,z0)

	float	theta, sin_theta,cos_theta;
	int		s, a;

	float	t;
	float	x2, y2;

	float	imb, jmb;
	int		ilb, jlb;
	float	fracI, fracJ, d1, d2, d1_sen, d2_sen;

	float	u_term;

	float	weight, total_sum;	//Parameters for handeling non-uniform sensitivity.
	//float	xCORr_p;	//Projection of the cetner of rotation onto the detector plane.
	//float	u;					//Distance from the projection of the voxel being considered and the projection of CoR in detector domain.

	//Calculate the index of the voxel being considered.
	//ind_x = nbatchIDx*((int)(IMGSIZx/h_nBatchXdim))+ tid;
	ind_x = tid;
	ind_y = bx;
	ind_z = by;

	ind_voxel=(ind_z*IMGSIZy+ind_y)*IMGSIZx+ind_x;  //(if prj is scanner data, need x_y_flip)
	//ind_voxel=(ind_z*IMGSIZx+ind_x)*IMGSIZy+ind_y;		
	//-------------------------------------------------------------------
	float	depth_weight;	
	//-------------------------------------------------------------------

	total_sum = 0;
	//total_sensitivity = 0;

	//Scan over all angles in the subset.
	for (a=angleStart; a<angleEnd; a++)
	{
		u_term = 0;

		s = d_index[a];		
		theta	= d_angles[s];
		sin_theta=sinf(theta);
		cos_theta=cosf(theta);

		//(x0,y0,z0) - source position.
		x0=sourceR*sin_theta;
		z0=sourceR*cos_theta;
		y0=sourceY;

		//(x1,y1,z1) - center of voxel
		x1 = (ind_x+0.5)*Vsize_x + x_p0;
		y1 = (ind_y+0.5)*Vsize_y + y_p0;
		z1 = (ind_z+0.5)*Vsize_z + z_p0;

		//Check FDK paper for this weight factor. This weight can be set to 1, in a simple case.
		depth_weight = (x0*x0+y0*y0+z0*z0)/((x0-x1)*(x0-x1) + (y0-y1)*(y0-y1)+(z0-z1)*(z0-z1));

		// Do NOT Rotate (x0,y0,z0)  -theta  around the y-axis.
		y0r =y0;		
		x0r =x0;
		z0r =z0;
		


		//Do NOT Rotate (x1,y1,z1)  -theta around the y-axis.
		y1r = y1;		
		z1r = z1;
		x1r = x1;
		//Rotate (x1,y1,z1)  -theta around the y-axis.
		//y1r = y1;		
		//z1r = z1 * cos_theta - x1 * (-sin_theta);
		//x1r = x1 * cos_theta + z1 * (-sin_theta);
		
		//The above rotations are used to simplify the calculation of the bin index.
		//Try to imagine the role of these rotations, if not, i will explain when we meet on June 18.

		//Determine the intersection of the ray passing through (x0r,y0r,z0r), (x1r,y1r,z1r) with the plane z = -detectorR
		if (z1r != z0r)
		{
			t = (-detectorR - z0r) / (z1r - z0r);

			x2 = x0r + (x1r - x0r) * t;
			y2 = y0r + (y1r - y0r) * t;

			weight=1;
			//----------------------------------------------------------------------------		
			//BACKPROJECTION USING INTERPOLATION

			//Calculate the continuous position (in bin_index coordinate) of the projection of voxel in the detector plane.
			imb = ((float)(x2 - x_d0)/Bsize_x);	
			jmb = ((float)(y2 - y_d0)/Bsize_y);

			ilb = floorf(imb);
			if (imb<ilb+0.5) ilb = ilb - 1;

			jlb = floorf(jmb);
			if (jmb<jlb+0.5) jlb = jlb - 1;

			fracI = imb - (ilb+0.5);
			fracJ = jmb - (jlb+0.5);


			d1 = d2 = 0;

			//Interpolation.
			if (	(ilb<BINSx)&&(ilb>=0) && (jlb<BINSy)&&(jlb>=0))
			{	
				bin_ind = ilb		* BINSy		+ jlb;				
				
					d1 = (1-fracI) * d_prjbuf[(a-angleStart)*BINSx*BINSy + bin_ind] ;
				
			}
			if (	(ilb<BINSx-1)&&(ilb>=-1) && (jlb<BINSy)&&(jlb>=0))
			{	
				bin_ind = (ilb + 1)	* BINSy		+ jlb;

					d1 = d1 + fracI  * d_prjbuf[(a-angleStart)*BINSx*BINSy + bin_ind];
			}

			if (	(ilb<BINSx)&&(ilb>=0) && (jlb<BINSy-1)&&(jlb>=-1))
			{	
				bin_ind = ilb		* BINSy		+  jlb + 1;

					d2 = (1-fracI) * d_prjbuf[(a-angleStart)*BINSx*BINSy + bin_ind];			
			
			}

			if (	(ilb<BINSx-1)&&(ilb>=-1) && (jlb<BINSy-1)&&(jlb>=-1))
			{	

				bin_ind = (ilb + 1) * BINSy		+  jlb + 1;
			
					d2 = d2 + fracI  * d_prjbuf[(a-angleStart)*BINSx*BINSy + bin_ind];			
				
			}

			u_term		= (1 - fracJ) * d1 + fracJ * d2;				

			////???
			u_term		= u_term*Vsize_z*depth_weight;//Giang multiply u_term with Vsize_z*depth_weight to make the result close to the chord length in forward projection
			//This line is not needed if you want to use ordinary interpolation method 



			total_sum	= total_sum	+ (u_term*weight);					//summation for all angle
			

		}//end of if z1r != z0r

		//cudaThreadSynchronize();
	}//end of a	(loop all angle)

	d_objbuf[ind_voxel] = d_objbuf[ind_voxel]+total_sum;
	//if(d_objbuf[ind_voxel]<0)d_objbuf[ind_voxel]=0; // Note: we do not constrain positivity for SBP
	
}


__global__ void backprj_OSSART_gpu_manyviews(
	float *d_objbuf,
	float *d_prjbuf,
	int *d_index,
	float *d_angles,
	int angleStart,
	int angleEnd, 
	int nbatchIDx, 
	float lambda)
{	
	// Block index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Thread index
	//const int tx = threadIdx.x;
    const int tx = threadIdx.x+(nbatchIDx*blockDim.x);
    const int ty = threadIdx.y;
    
    //const int tid = tx * blockDim.y + ty;
	const int tid=tx;//+ty*blockDim.x;//+nbatchIDx*blockDim.x;//*blockDim.y;  


	int		ind_x, ind_y, ind_z;
	int		bin_ind;
	long	ind_voxel;

	float	x1, y1, z1, x0, y0, z0;
	float	x1r, y1r, z1r, x0r, y0r, z0r;	//Rotated locations of (x1,y1,z1) and (x0,y0,z0)

	float	theta, sin_theta,cos_theta;
	int		s, a;

	float	t;
	float	x2, y2;

	float	imb, jmb;
	int		ilb, jlb;
	float	fracI, fracJ, d1, d2, d1_sen, d2_sen;

	float	u_term, u_sensitivity, total_sensitivity;

	float	weight, total_sum;	//Parameters for handeling non-uniform sensitivity.
	//float	xCORr_p;	//Projection of the cetner of rotation onto the detector plane.
	//float	u;					//Distance from the projection of the voxel being considered and the projection of CoR in detector domain.

	//Calculate the index of the voxel being considered.
	//ind_x = nbatchIDx*((int)(IMGSIZx/h_nBatchXdim))+ tid;
	ind_x = tid;
	ind_y = bx;
	ind_z = by;

	ind_voxel=(ind_z*IMGSIZy+ind_y)*IMGSIZx+ind_x;  //(if prj is scanner data, need x_y_flip)
	//ind_voxel=(ind_z*IMGSIZx+ind_x)*IMGSIZy+ind_y;		
	//-------------------------------------------------------------------
	float	depth_weight;	
	//-------------------------------------------------------------------

	total_sum = 0;
	total_sensitivity = 0;

	//Scan over all angles in the subset.
	for (a=angleStart; a<angleEnd; a++)
	{
		u_term = 0;

		s = d_index[a];		
		theta	= d_angles[s];
		sin_theta=sinf(theta);
		cos_theta=cosf(theta);

		//(x0,y0,z0) - source position.
		x0=sourceR*sin_theta;
		z0=sourceR*cos_theta;
		y0=sourceY;

		//(x1,y1,z1) - center of voxel
		x1 = (ind_x+0.5)*Vsize_x + x_p0;
		y1 = (ind_y+0.5)*Vsize_y + y_p0;
		z1 = (ind_z+0.5)*Vsize_z + z_p0;

		//Check FDK paper for this weight factor. This weight can be set to 1, in a simple case.
		depth_weight = (x0*x0+y0*y0+z0*z0)/((x0-x1)*(x0-x1) + (y0-y1)*(y0-y1)+(z0-z1)*(z0-z1));

		// Do NOT Rotate (x0,y0,z0)  -theta  around the y-axis.
		y0r =y0;		
		x0r =x0;
		z0r =z0;
		


		//Do NOT Rotate (x1,y1,z1)  -theta around the y-axis.
		y1r = y1;		
		z1r = z1;
		x1r = x1;
		//Rotate (x1,y1,z1)  -theta around the y-axis.
		//y1r = y1;		
		//z1r = z1 * cos_theta - x1 * (-sin_theta);
		//x1r = x1 * cos_theta + z1 * (-sin_theta);
		
		//The above rotations are used to simplify the calculation of the bin index.
		//Try to imagine the role of these rotations, if not, i will explain when we meet on June 18.

		//Determine the intersection of the ray passing through (x0r,y0r,z0r), (x1r,y1r,z1r) with the plane z = -detectorR
		if (z1r != z0r)
		{
			t = (-detectorR - z0r) / (z1r - z0r);

			x2 = x0r + (x1r - x0r) * t;
			y2 = y0r + (y1r - y0r) * t;

			weight=1;
			//----------------------------------------------------------------------------		
			//BACKPROJECTION USING INTERPOLATION

			//Calculate the continuous position (in bin_index coordinate) of the projection of voxel in the detector plane.
			imb = ((float)(x2 - x_d0)/Bsize_x);	
			jmb = ((float)(y2 - y_d0)/Bsize_y);

			ilb = floorf(imb);
			if (imb<ilb+0.5) ilb = ilb - 1;

			jlb = floorf(jmb);
			if (jmb<jlb+0.5) jlb = jlb - 1;

			fracI = imb - (ilb+0.5);
			fracJ = jmb - (jlb+0.5);


			d1 = d2 = 0;
			d1_sen = d2_sen = 0;

			//Interpolation.
			if (	(ilb<BINSx)&&(ilb>=0) && (jlb<BINSy)&&(jlb>=0))
			{	
				bin_ind = ilb		* BINSy		+ jlb;				
				
					d1 = (1-fracI) * d_prjbuf[(a-angleStart)*BINSx*BINSy + bin_ind] ;
					d1_sen = (1-fracI);
				
			}
			if (	(ilb<BINSx-1)&&(ilb>=-1) && (jlb<BINSy)&&(jlb>=0))
			{	
				bin_ind = (ilb + 1)	* BINSy		+ jlb;

					d1 = d1 + fracI  * d_prjbuf[(a-angleStart)*BINSx*BINSy + bin_ind];
					d1_sen = d1_sen + fracI;

			}

			if (	(ilb<BINSx)&&(ilb>=0) && (jlb<BINSy-1)&&(jlb>=-1))
			{	
				bin_ind = ilb		* BINSy		+  jlb + 1;

					d2 = (1-fracI) * d_prjbuf[(a-angleStart)*BINSx*BINSy + bin_ind];			
					d2_sen = (1-fracI);
			
			}

			if (	(ilb<BINSx-1)&&(ilb>=-1) && (jlb<BINSy-1)&&(jlb>=-1))
			{	

				bin_ind = (ilb + 1) * BINSy		+  jlb + 1;
			
					d2 = d2 + fracI  * d_prjbuf[(a-angleStart)*BINSx*BINSy + bin_ind];			
					d2_sen = d2_sen + fracI;
				
			}

			u_term		= (1 - fracJ) * d1 + fracJ * d2;				

			////???
			u_term		= u_term*Vsize_z*depth_weight;//Giang multiply u_term with Vsize_z*depth_weight to make the result close to the chord length in forward projection
			//This line is not needed if you want to use ordinary interpolation method 


			u_sensitivity=((1-fracJ)*d1_sen+fracJ*d2_sen);
			u_sensitivity = u_sensitivity *Vsize_z*depth_weight;	//We multiply u_sensitivity with Vsize_z*depth_weight to make the result close to the chord length in forward projection
			//This line is not needed if you want to use ordinary interpolation method 



			total_sum	= total_sum	+ (u_term*weight);					//summation for all angle
			
			total_sensitivity=total_sensitivity+(u_sensitivity*weight);

		}//end of if z1r != z0r

		//cudaThreadSynchronize();
	}//end of a	(loop all angle)


	u_term=0;
	if(total_sensitivity!=0)u_term=(total_sum/total_sensitivity);
	//This implementation is for the case when all computations are done in GPU.

	d_objbuf[ind_voxel] =d_objbuf[ind_voxel]+lambda*u_term;
	if(d_objbuf[ind_voxel]<0)d_objbuf[ind_voxel]=0;

}

__global__ void backprj_OSSART_R_gpu_manyviews(
	float *d_objbuf,
	float *d_prjbuf,
	float *d_prior,
	int	  *d_index,
	float *d_angles,
	int angleStart,
	int angleEnd, 
	int nbatchIDx, 
	float lambda,
	float beta)
{	
	// Block index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Thread index
	//const int tx = threadIdx.x;
    const int tx = threadIdx.x+(nbatchIDx*blockDim.x);
    const int ty = threadIdx.y;
    
    //const int tid = tx * blockDim.y + ty;
	const int tid=tx;//+ty*blockDim.x;//+nbatchIDx*blockDim.x;//*blockDim.y;  


	int		ind_x, ind_y, ind_z;
	int		bin_ind;
	long	ind_voxel;

	float	x1, y1, z1, x0, y0, z0;
	float	x1r, y1r, z1r, x0r, y0r, z0r;	//Rotated locations of (x1,y1,z1) and (x0,y0,z0)

	float	theta, sin_theta,cos_theta;
	int		s, a;

	float	t;
	float	x2, y2;

	float	imb, jmb;
	int		ilb, jlb;
	float	fracI, fracJ, d1, d2, d1_sen, d2_sen;

	float	u_term, beta_term, u_sensitivity, total_sensitivity;

	float	weight, total_sum;	//Parameters for handeling non-uniform sensitivity.
	//float	xCORr_p;	//Projection of the cetner of rotation onto the detector plane.
	//float	u;					//Distance from the projection of the voxel being considered and the projection of CoR in detector domain.

	//Calculate the index of the voxel being considered.
	//ind_x = nbatchIDx*((int)(IMGSIZx/h_nBatchXdim))+ tid;
	ind_x = tid;
	ind_y = bx;
	ind_z = by;

	ind_voxel=(ind_z*IMGSIZy+ind_y)*IMGSIZx+ind_x;  //(if prj is scanner data, need x_y_flip)
	//ind_voxel=(ind_z*IMGSIZx+ind_x)*IMGSIZy+ind_y;		
	//-------------------------------------------------------------------
	float	depth_weight;	
	//-------------------------------------------------------------------

	total_sum = 0;
	total_sensitivity = 0;

	//Scan over all angles in the subset.
	for (a=angleStart; a<angleEnd; a++)
	{
		u_term = 0;

		s = d_index[a];		
		theta	= d_angles[s];
		sin_theta=sinf(theta);
		cos_theta=cosf(theta);

		//(x0,y0,z0) - source position.
		x0=sourceR*sin_theta;
		z0=sourceR*cos_theta;
		y0=sourceY;

		//(x1,y1,z1) - center of voxel
		x1 = (ind_x+0.5)*Vsize_x + x_p0;
		y1 = (ind_y+0.5)*Vsize_y + y_p0;
		z1 = (ind_z+0.5)*Vsize_z + z_p0;

		//Check FDK paper for this weight factor. This weight can be set to 1, in a simple case.
		depth_weight = (x0*x0+y0*y0+z0*z0)/((x0-x1)*(x0-x1) + (y0-y1)*(y0-y1)+(z0-z1)*(z0-z1));

		// Do NOT Rotate (x0,y0,z0)  -theta  around the y-axis.
		y0r =y0;		
		x0r =x0;
		z0r =z0;
		


		//Do NOT Rotate (x1,y1,z1)  -theta around the y-axis.
		y1r = y1;		
		z1r = z1;
		x1r = x1;
		//Rotate (x1,y1,z1)  -theta around the y-axis.
		//y1r = y1;		
		//z1r = z1 * cos_theta - x1 * (-sin_theta);
		//x1r = x1 * cos_theta + z1 * (-sin_theta);
		
		//The above rotations are used to simplify the calculation of the bin index.
		//Try to imagine the role of these rotations, if not, i will explain when we meet on June 18.

		//Determine the intersection of the ray passing through (x0r,y0r,z0r), (x1r,y1r,z1r) with the plane z = -detectorR
		if (z1r != z0r)
		{
			t = (-detectorR - z0r) / (z1r - z0r);

			x2 = x0r + (x1r - x0r) * t;
			y2 = y0r + (y1r - y0r) * t;

			weight=1;
			//----------------------------------------------------------------------------		
			//BACKPROJECTION USING INTERPOLATION

			//Calculate the continuous position (in bin_index coordinate) of the projection of voxel in the detector plane.
			imb = ((float)(x2 - x_d0)/Bsize_x);	
			jmb = ((float)(y2 - y_d0)/Bsize_y);

			ilb = floorf(imb);
			if (imb<ilb+0.5) ilb = ilb - 1;

			jlb = floorf(jmb);
			if (jmb<jlb+0.5) jlb = jlb - 1;

			fracI = imb - (ilb+0.5);
			fracJ = jmb - (jlb+0.5);


			d1 = d2 = 0;
			d1_sen = d2_sen = 0;

			//Interpolation.
			if (	(ilb<BINSx)&&(ilb>=0) && (jlb<BINSy)&&(jlb>=0))
			{	
				bin_ind = ilb		* BINSy		+ jlb;				
				
					d1 = (1-fracI) * d_prjbuf[(a-angleStart)*BINSx*BINSy + bin_ind] ;
					d1_sen = (1-fracI);
				
			}
			if (	(ilb<BINSx-1)&&(ilb>=-1) && (jlb<BINSy)&&(jlb>=0))
			{	
				bin_ind = (ilb + 1)	* BINSy		+ jlb;

					d1 = d1 + fracI  * d_prjbuf[(a-angleStart)*BINSx*BINSy + bin_ind];
					d1_sen = d1_sen + fracI;

			}

			if (	(ilb<BINSx)&&(ilb>=0) && (jlb<BINSy-1)&&(jlb>=-1))
			{	
				bin_ind = ilb		* BINSy		+  jlb + 1;

					d2 = (1-fracI) * d_prjbuf[(a-angleStart)*BINSx*BINSy + bin_ind];			
					d2_sen = (1-fracI);
			
			}

			if (	(ilb<BINSx-1)&&(ilb>=-1) && (jlb<BINSy-1)&&(jlb>=-1))
			{	

				bin_ind = (ilb + 1) * BINSy		+  jlb + 1;
			
					d2 = d2 + fracI  * d_prjbuf[(a-angleStart)*BINSx*BINSy + bin_ind];			
					d2_sen = d2_sen + fracI;
				
			}

			u_term		= (1 - fracJ) * d1 + fracJ * d2;				

			////???
			u_term		= u_term*Vsize_z*depth_weight;//Giang multiply u_term with Vsize_z*depth_weight to make the result close to the chord length in forward projection
			//This line is not needed if you want to use ordinary interpolation method 


			u_sensitivity=((1-fracJ)*d1_sen+fracJ*d2_sen);
			u_sensitivity = u_sensitivity *Vsize_z*depth_weight;	//We multiply u_sensitivity with Vsize_z*depth_weight to make the result close to the chord length in forward projection
			//This line is not needed if you want to use ordinary interpolation method 



			total_sum	= total_sum	+ (u_term*weight);					//summation for all angle
			
			total_sensitivity=total_sensitivity+(u_sensitivity*weight);

		}//end of if z1r != z0r

		//cudaThreadSynchronize();
	}//end of a	(loop all angle)

	u_term=0;
	beta_term=0;
	if(total_sensitivity!=0){
		u_term=(total_sum/total_sensitivity);
		beta_term=(beta*d_prior[ind_voxel])/total_sensitivity;
	}
	//This implementation is for the case when all computations are done in GPU.



	d_objbuf[ind_voxel] =d_objbuf[ind_voxel]+lambda*(u_term+beta_term);
	if(d_objbuf[ind_voxel]<0)d_objbuf[ind_voxel]=0;
}

__global__ void SART_prj_diff_kernel(
	float *diff_line,
	float *prjbuf,
	float *prj_est,
	float *normprj,
	int *d_index,
	int angleStart,
	int angleEnd,
	int nbBinsX, 
	int nbBinsY){
	// Block index
	const int bx = blockIdx.x;
	const int by = blockIdx.y;

	// Thread index

	const int ix = threadIdx.x+(bx*blockDim.x);
	const int iy = threadIdx.y+(by*blockDim.y);

	int i,j;
	int	bin_ind;
	int a;

	for(a=angleStart;a<angleEnd;a++){

		i = nbBinsX*((int)BINSx/nBatchBINSx) + ix;
		j = nbBinsY*((int)BINSy/nBatchBINSy) + iy;

		bin_ind=((a-angleStart)*BINSx+i)*BINSy+j;	//Bin index. Do NOT need x_y_flip() in BP
		//bin_ind=((a-angleStart)*BINSy+j)*BINSx+i;	//Bin index. Need x_y_flip() in BP. Output's coor-system will be as same as TOMO scanner (NOT runnable)

		if (*(normprj+bin_ind)!=0){
			*(diff_line+bin_ind)=(*(prjbuf+bin_ind)-*(prj_est+bin_ind))/(*(normprj+bin_ind));
		}
		else{
			*(diff_line+bin_ind)=0;
		}
	}
}


extern "C"
	void Reconstruction();

void prior_GPU_OSTR_Q2(
	float *d_RDD,
	float *d_RD,
	float *d_est,
	float z_xy_ratio)
{
	int		nbatchIDx;

	dim3 BACKPRJ_THREAD(h_BACKPRJ_ThreX, h_BACKPRJ_ThreY);
	dim3 BACKPRJ_GRID  (h_BACKPRJ_GridX, h_BACKPRJ_GridY);


	//system("PAUSE");
	// YL method


	for (nbatchIDx = 0; nbatchIDx<h_nBatchXdim; nbatchIDx++)
	{
		G_Q2_prior<<< BACKPRJ_GRID, BACKPRJ_THREAD >>>(d_RDD, d_RD, d_est,z_xy_ratio, nbatchIDx);
		//Check out the content of this kernel in file ConebeamCT_kernel.cu
		////CUT_CHECK_ERROR("Kernel execution failed");
		cudaThreadSynchronize();
	}
}



void prior_GPU_OSTR(
	float *d_RDD,
	float *d_RD,
	float *d_est,
	float delta,
	float z_xy_ratio)
{
	int		nbatchIDx;

	dim3 BACKPRJ_THREAD(h_BACKPRJ_ThreX, h_BACKPRJ_ThreY);
	dim3 BACKPRJ_GRID  (h_BACKPRJ_GridX, h_BACKPRJ_GridY);


	//system("PAUSE");
	// YL method

	if (delta == 0){
		printf("delta cannot be ZARO!!\n");
		system("PAUSE");
		exit(1);
	}

	for (nbatchIDx = 0; nbatchIDx < h_nBatchXdim; nbatchIDx++)
	{
		G_Fessler_prior<<< BACKPRJ_GRID, BACKPRJ_THREAD >>>(d_RDD, d_RD, d_est, delta, z_xy_ratio, nbatchIDx);
		//Check out the content of this kernel in file ConebeamCT_kernel.cu
		//CUT_CHECK_ERROR("Kernel execution failed");
		cudaThreadSynchronize();
	}
}



void prior_GPU_SART(
	float *d_prior,
	float *d_est,
	float delta)
{
	int		nbatchIDx;

	dim3 BACKPRJ_THREAD(h_BACKPRJ_ThreX, h_BACKPRJ_ThreY);
	dim3 BACKPRJ_GRID  (h_BACKPRJ_GridX, h_BACKPRJ_GridY);
	
	
	//system("PAUSE");
	// YL method
	
	if (delta==0){
		printf("delta cannot be ZARO!!\n");
		system("PAUSE");exit(1);
	}

	for (nbatchIDx = 0; nbatchIDx<h_nBatchXdim; nbatchIDx++)
	{
		G_Huber_prior<<< BACKPRJ_GRID, BACKPRJ_THREAD >>>(d_prior, d_est, delta, nbatchIDx);
		//Check out the content of this kernel in file ConebeamCT_kernel.cu
		//CUT_CHECK_ERROR("Kernel execution failed");
		cudaThreadSynchronize();
	}
}


void fprojectCB_1R_GPU_SART (
	float *estbuf, 
	float *prj_est, 
	float *d_normprj,
	float *d_angles, 
	int *d_index, 
	int angleStart,
	int angleEnd)
{
	int		nbBinsX, nbBinsY;

	dim3 PRJ_THREAD(h_PRJ_ThreX, h_PRJ_ThreY);
	dim3 PRJ_GRID  (h_PRJ_GridX, h_PRJ_GridY);

	//If the detector is too large, we perform projection in a batch-type implementation.
	//For each batch we perform projection for (BINSx/h_nBatchBINSx) * (h_BINSy/h_nBatchBINSy) bins.
	for (nbBinsX=0; nbBinsX<h_nBatchBINSx; nbBinsX++)	
		for (nbBinsY=0; nbBinsY<h_nBatchBINSy; nbBinsY++)
		{			

			ray_trace_gpu_manyangles_direct_notexturememory	<<< PRJ_GRID, PRJ_THREAD >>>(estbuf, prj_est, d_normprj,d_angles, d_index, angleStart, angleEnd ,  nbBinsX, nbBinsY);
			//Check out the content of this kernel in file ConebeamCT_kernel.cu
			//CUT_CHECK_ERROR("Kernel execution failed");
			cudaThreadSynchronize();

		}
}

void fprojectCB_1R_GPU_SART_cos (
	float *estbuf, 
	float *prj_est, 
	float *d_normprj, 
	float *d_angles, 
	int *d_index, 
	int angleStart,
	int angleEnd)
{
	int		nbBinsX, nbBinsY;

	dim3 PRJ_THREAD(h_PRJ_ThreX, h_PRJ_ThreY);
	dim3 PRJ_GRID  (h_PRJ_GridX, h_PRJ_GridY);

	//If the detector is too large, we perform projection in a batch-type implementation.
	//For each batch we perform projection for (h_BINSx/h_nBatchBINSx) * (h_BINSy/h_nBatchBINSy) bins.
	for (nbBinsX=0; nbBinsX<h_nBatchBINSx; nbBinsX++)	
		for (nbBinsY=0; nbBinsY<h_nBatchBINSy; nbBinsY++)
		{			

			ray_trace_gpu_manyangles_direct_notexturememory_cos	<<< PRJ_GRID, PRJ_THREAD >>>(estbuf, prj_est, d_normprj,d_angles, d_index, angleStart, angleEnd ,  nbBinsX, nbBinsY);
			//Check out the content of this kernel in file ConebeamCT_kernel.cu
			//CUT_CHECK_ERROR("Kernel execution failed");
			cudaThreadSynchronize();

		}
}

void fprojectCB_1R_GPU_OSTR_cos (
	float *estbuf, 
	float *prj_est, 
	float *d_angles, 
	int *d_index, 
	int angleStart,
	int angleEnd)
{
	int		nbBinsX, nbBinsY;

	dim3 PRJ_THREAD(h_PRJ_ThreX, h_PRJ_ThreY);
	dim3 PRJ_GRID  (h_PRJ_GridX, h_PRJ_GridY);

	//If the detector is too large, we perform projection in a batch-type implementation.
	//For each batch we perform projection for (h_BINSx/h_nBatchBINSx) * (h_BINSy/h_nBatchBINSy) bins.
	for (nbBinsX=0; nbBinsX<h_nBatchBINSx; nbBinsX++)	
		for (nbBinsY=0; nbBinsY<h_nBatchBINSy; nbBinsY++)
		{			

			ray_trace_gpu_manyangles_direct_notexturememory_OSTR_cos	<<< PRJ_GRID, PRJ_THREAD >>>(estbuf, prj_est,d_angles, d_index, angleStart, angleEnd ,  nbBinsX, nbBinsY);
			//Check out the content of this kernel in file ConebeamCT_kernel.cu
			//CUT_CHECK_ERROR("Kernel execution failed");
			cudaThreadSynchronize();

		}
}

void fprojectCB_1R_GPU_OSTR_normprj ( 
	float *d_normprj, 
	float *d_angles, 
	int *d_index, 
	int angleStart,
	int angleEnd)
{
	int		nbBinsX, nbBinsY;

	dim3 PRJ_THREAD(h_PRJ_ThreX, h_PRJ_ThreY);
	dim3 PRJ_GRID  (h_PRJ_GridX, h_PRJ_GridY);

	//If the detector is too large, we perform projection in a batch-type implementation.
	//For each batch we perform projection for (h_BINSx/h_nBatchBINSx) * (h_BINSy/h_nBatchBINSy) bins.
	for (nbBinsX=0; nbBinsX<h_nBatchBINSx; nbBinsX++)	
		for (nbBinsY=0; nbBinsY<h_nBatchBINSy; nbBinsY++)
		{			
			printf("angle start %d %d", angleStart, angleEnd);
			ray_trace_gpu_manyangles_direct_notexturememory_normprj	<<< PRJ_GRID, PRJ_THREAD >>>(d_normprj, d_angles, d_index, angleStart, angleEnd ,  nbBinsX, nbBinsY);
			//Check out the content of this kernel in file ConebeamCT_kernel.cu
			//CUT_CHECK_ERROR("Kernel execution failed");
			
			cudaThreadSynchronize();
                        cudaDeviceSynchronize();

			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) 
    				printf("Error: %s\n", cudaGetErrorString(err));
			printf("after kernel ended %d %d", angleStart, angleEnd);
                        printf("cuda sync done");

		}
}




void fprojectCB1_4R_GPU_SART(
	float *d_objbuf, 
	float *d_prjbuf,  
	float *d_4R_prjbuf, 
	float *d_angles, 
	int *d_index, 
	int angleStart, 
	int angleEnd){
	int		nbBinsX, nbBinsY;

	dim3 PRJ_THREAD_4R (PRJ_ThreX_4R, PRJ_ThreY_4R);
	dim3 PRJ_GRID_4R  (PRJ_GridX_4R, PRJ_GridY_4R);

	dim3 PRJ_THREAD (h_PRJ_ThreX, h_PRJ_ThreY);
	dim3 PRJ_GRID  (h_PRJ_GridX, h_PRJ_GridY);

	//If the detector is too large, we perform projection in a batch-type implementation.
	//For each batch we perform projection for (h_BINSx/h_nBatchBINSx) * (h_BINSy/h_nBatchBINSy) bins.

	for (nbBinsX=0; nbBinsX<nBatchBINSx_4R; nbBinsX++)	{
		for (nbBinsY=0; nbBinsY<nBatchBINSy_4R; nbBinsY++){			
			ray_trace_4R_gpu_manyangles_direct_notexturememory	<<< PRJ_GRID_4R, PRJ_THREAD_4R >>>(d_objbuf, d_4R_prjbuf, d_angles, d_index, angleStart, angleEnd , nbBinsX,nbBinsY);

			//Check out the content of this kernel in file ConebeamCT_kernel.cu
			//CUT_CHECK_ERROR("Kernel execution failed");
			//system("PAUSE");exit(3);
			cudaThreadSynchronize();
		}
	}
	//system("PAUSE");exit(3);

	merge_4R_to_1R_global <<<PRJ_GRID,PRJ_THREAD>>>(d_prjbuf, d_4R_prjbuf,d_angles, d_index, angleStart, angleEnd);
	//CUT_CHECK_ERROR("Kernel execution failed----- merge_4R_to_1R_global");
	cudaThreadSynchronize();

}

void bprojectCB_GPU_SBP(
	float *d_objbuf, 
	float *d_prjbuf, 
	int *d_index, 
	float *d_angles, 
	int angleStart,
	int angleEnd)
{
	int		nbatchIDx;

	dim3 BACKPRJ_THREAD(h_BACKPRJ_ThreX, h_BACKPRJ_ThreY);
	dim3 BACKPRJ_GRID  (h_BACKPRJ_GridX, h_BACKPRJ_GridY);
	//system("PAUSE");
// YL method
	for (nbatchIDx = 0; nbatchIDx<h_nBatchXdim; nbatchIDx++)
	{

		backprj_gpu_manyviews_SBP<<< BACKPRJ_GRID, BACKPRJ_THREAD >>>(d_objbuf, d_prjbuf, d_index, d_angles, angleStart, angleEnd , nbatchIDx);
		//Check out the content of this kernel in file ConebeamCT_kernel.cu
		//CUT_CHECK_ERROR("Kernel execution failed");
		cudaThreadSynchronize();
	}
}


void bprojectCB_4B_GPU_SART(
	float *d_objbuf, 
	float *d_prjbuf, 
	int *d_index, 
	float *d_angles, 
	int angleStart,
	int angleEnd, 
	float lambda)
{
	int		nbatchIDx;

	dim3 BACKPRJ_THREAD(h_BACKPRJ_ThreX, h_BACKPRJ_ThreY);
	dim3 BACKPRJ_GRID  (h_BACKPRJ_GridX, h_BACKPRJ_GridY);
	//system("PAUSE");
// YL method
	for (nbatchIDx = 0; nbatchIDx<h_nBatchXdim; nbatchIDx++)
	{

		backprj_OSSART_gpu_manyviews<<< BACKPRJ_GRID, BACKPRJ_THREAD >>>(d_objbuf, d_prjbuf, d_index, d_angles, angleStart, angleEnd , nbatchIDx, lambda);
		//Check out the content of this kernel in file ConebeamCT_kernel.cu
		//CUT_CHECK_ERROR("Kernel execution failed");
		cudaThreadSynchronize();
	}
// YL method end
// 
/* Backup
	for (nbatchIDx = 0; nbatchIDx<h_nBatchXdim; nbatchIDx++)
	{
		system("PAUSE");
		backprj_gpu1_manyviews_SBP<<< BACKPRJ_GRID, BACKPRJ_THREAD >>>(d_objbuf, d_prjbuf, d_index, d_angles, angleStart, angleEnd , nbatchIDx, d_DeadCell, lambda);
		//Check out the content of this kernel in file ConebeamCT_kernel.cu
		//CUT_CHECK_ERROR("Kernel execution failed");
		cudaThreadSynchronize();
	}
	*/
}

void bprojectCB_4B_GPU_SART_R(
	float *d_objbuf, 
	float *d_prjbuf,
	float *d_prior,
	int *d_index, 
	float *d_angles, 
	int angleStart,
	int angleEnd, 
	float lambda,
	float beta)
{
	int		nbatchIDx;

	dim3 BACKPRJ_THREAD(h_BACKPRJ_ThreX, h_BACKPRJ_ThreY);
	dim3 BACKPRJ_GRID  (h_BACKPRJ_GridX, h_BACKPRJ_GridY);
	//system("PAUSE");
// YL method
	for (nbatchIDx = 0; nbatchIDx<h_nBatchXdim; nbatchIDx++)
	{

		backprj_OSSART_R_gpu_manyviews<<< BACKPRJ_GRID, BACKPRJ_THREAD >>>(d_objbuf, d_prjbuf, d_prior, d_index, d_angles, angleStart, angleEnd , nbatchIDx, lambda, beta);
		//Check out the content of this kernel in file ConebeamCT_kernel.cu
		//CUT_CHECK_ERROR("Kernel execution failed");
		cudaThreadSynchronize();
	}
// YL method end
// 
/* Backup
	for (nbatchIDx = 0; nbatchIDx<h_nBatchXdim; nbatchIDx++)
	{
		system("PAUSE");
		backprj_gpu1_manyviews_SBP<<< BACKPRJ_GRID, BACKPRJ_THREAD >>>(d_objbuf, d_prjbuf, d_index, d_angles, angleStart, angleEnd , nbatchIDx, d_DeadCell, lambda);
		//Check out the content of this kernel in file ConebeamCT_kernel.cu
		//CUT_CHECK_ERROR("Kernel execution failed");
		cudaThreadSynchronize();
	}
	*/
}


void SART_prj_diff(
	float *diff_line,
	float *prjbuf,
	float *prj_est,
	float *normprj,
	int *d_index,
	int angleStart, 
	int angleEnd){
	int nbBinsX,nbBinsY;

	dim3 PRJ_THREAD(h_PRJ_ThreX, h_PRJ_ThreY);
	dim3 PRJ_GRID (h_PRJ_GridX,  h_PRJ_GridY);


	for (nbBinsX=0; nbBinsX<h_nBatchBINSx; nbBinsX++)	{
		for (nbBinsY=0; nbBinsY<h_nBatchBINSy; nbBinsY++){
			SART_prj_diff_kernel<<<PRJ_GRID,PRJ_THREAD>>>(diff_line,prjbuf,prj_est,normprj,d_index,angleStart,angleEnd,nbBinsX,nbBinsY);
			//CUT_CHECK_ERROR("Kernel execution failed");
			cudaThreadSynchronize();
		}
	}
}


int main( int argc, char *argv[]) 
{
	char datafile[300];
	if (argc != 2){
		printf("Please ONLY enter the name of datafile!\n");
		system("PAUSE");exit(1);
	}
	else strcpy (datafile,argv[1]);
	
	FILE *fp;
	// char    *datafile = "K:/Yihuan_Lu/GPU/CUDA/Tomo_sbp_scanner_Interleaved/BP_dat_file.txt";//test purpose 
	fp = fopen(datafile,"r");
	if( fp == NULL ) {
		printf("Error: datafile not found!!");
		system("PAUSE");exit(1);
	}

	readINTEGERpara(datafile,(char *)"image_size_x",1,&h_IMGSIZx);
	readINTEGERpara(datafile,(char *)"image_size_y",1,&h_IMGSIZy);
	readINTEGERpara(datafile,(char *)"image_size_z",1,&h_IMGSIZz);

	readFLOATpara(datafile,(char *)"Voxel_size_x_mm",1,&h_Vsize_x);
	readFLOATpara(datafile,(char *)"Voxel_size_y_mm",1,&h_Vsize_y);
	readFLOATpara(datafile,(char *)"Voxel_size_z_mm",1,&h_Vsize_z);

	readFLOATpara(datafile,(char *)"object_xpos_mm",1,&h_x_p0);
	readFLOATpara(datafile,(char *)"object_ypos_mm",1,&h_y_p0);
	readFLOATpara(datafile,(char *)"object_zpos_mm",1,&h_z_p0);

	readFLOATpara(datafile,(char *)"detector_x0_mm",1,&h_x_d0);
	readFLOATpara(datafile,(char *)"detector_y0_mm",1,&h_y_d0);
	readFLOATpara(datafile,(char *)"detector_COR_mm",1,&h_detectorR);
	readFLOATpara(datafile,(char *)"source_COR_mm",1,&h_sourceR);
	readFLOATpara(datafile,(char *)"source_Y_shift_mm",1,&h_sourceY);

	readINTEGERpara(datafile,(char *)"angular_samples",1,&h_ANGLES);
	//printf("ANGLE SAMPLE NUM is : %d\n", h_ANGLES);

	readINTEGERpara(datafile,(char *)"BACKPRJ_ThreX",1,&h_BACKPRJ_ThreX);
	readINTEGERpara(datafile,(char *)"BACKPRJ_ThreY",1,&h_BACKPRJ_ThreY);
	readINTEGERpara(datafile,(char *)"BACKPRJ_GridX",1,&h_BACKPRJ_GridX);
	readINTEGERpara(datafile,(char *)"BACKPRJ_GridY",1,&h_BACKPRJ_GridY);
	readINTEGERpara(datafile,(char *)"nBatchXdim", 1,&h_nBatchXdim);

	readINTEGERpara(datafile,(char *)"bin_samples_x",1,&h_BINSx);
	readINTEGERpara(datafile,(char *)"bin_samples_y",1,&h_BINSy);

	readFLOATpara(datafile,(char *)"Bin_size_x_mm",1,&h_Bsize_x);
	readFLOATpara(datafile,(char *)"Bin_size_y_mm",1,&h_Bsize_y);

	readINTEGERpara(datafile,(char *)"nBatchBINSx",1,&h_nBatchBINSx);
	readINTEGERpara(datafile,(char *)"nBatchBINSy",1,&h_nBatchBINSy);

	readINTEGERpara(datafile,(char *)"PRJ_ThreX",1,&h_PRJ_ThreX);
	readINTEGERpara(datafile,(char *)"PRJ_ThreY",1,&h_PRJ_ThreY);
	readINTEGERpara(datafile,(char *)"PRJ_GridX",1,&h_PRJ_GridX);
	readINTEGERpara(datafile,(char *)"PRJ_GridY",1,&h_PRJ_GridY);

	readINTEGERpara(datafile,(char *)"Readout_Every_X_Iters",1,&h_IO_Iter);
	readINTEGERpara(datafile,(char *)"Recon_Method",1,&h_method);

	readFLOATpara(datafile,(char *)"Beta",1,&h_beta);
	readFLOATpara(datafile,(char *)"Delta",1,&h_delta);
	readINTEGERpara(datafile,(char *)"iteration_number",1,&h_iter_num);
	readINTEGERpara(datafile,(char *)"subset_number",1,&h_subset_num);
	
	if ((h_ANGLES%h_subset_num)!=0){
		printf("Error!! subset_number is not a divisor of  angular_samples. Program will terminate!\n");
		system("PAUSE");exit(1);
	}

	/*if (h_delta==0){
		printf("delta CANNOT be zero! Program will terminate!\n");
		system("PAUSE");exit(1);
	}
	if (h_iter_num==0){
		printf("iteration number CANNOT be zero! Program will terminate!\n");
		system("PAUSE");exit(1);
	}
	*/
	readSTRINGpara(datafile,(char *)"ini_guess",1,h_ini_name);		// No need for device
	readSTRINGpara(datafile,(char *)"angle_name",1,h_angle_filename);		// No need for device 
	readSTRINGpara(datafile,(char *)"projection_folder",1,h_prj_folder);	// No need for device 
	readSTRINGpara(datafile,(char *)"projection_name",1,h_prj_name);			// No need for device
	readSTRINGpara(datafile,(char *)"scatter_folder",1,h_scat_folder);	// No need for device 
	readSTRINGpara(datafile,(char *)"scatter_name",1,h_scat_name);			// No need for device
	readSTRINGpara(datafile,(char *)"blank_folder",1,h_blank_folder);	// No need for device 
	readSTRINGpara(datafile,(char *)"blank_name",1,h_blank_name);			// No need for device
	readSTRINGpara(datafile,(char *)"output_recon_folder",1,h_recon_folder); // No need for device
	readSTRINGpara(datafile,(char *)"output_recon_name",1,h_recon_name); // No need for device


	cudaMemcpyToSymbol(IMGSIZx,&h_IMGSIZx, sizeof(int));
	cudaMemcpyToSymbol(IMGSIZy,&h_IMGSIZy, sizeof(int));
	cudaMemcpyToSymbol(IMGSIZz,&h_IMGSIZz, sizeof(int));

	cudaMemcpyToSymbol(Vsize_x,&h_Vsize_x, sizeof(float));
	cudaMemcpyToSymbol(Vsize_y,&h_Vsize_y, sizeof(float));
	cudaMemcpyToSymbol(Vsize_z,&h_Vsize_z, sizeof(float));

	cudaMemcpyToSymbol(x_p0,&h_x_p0, sizeof(float));
	cudaMemcpyToSymbol(y_p0,&h_y_p0, sizeof(float));
	cudaMemcpyToSymbol(z_p0,&h_z_p0, sizeof(float));

	cudaMemcpyToSymbol(x_d0,&h_x_d0, sizeof(float));
	cudaMemcpyToSymbol(y_d0,&h_y_d0, sizeof(float));

	cudaMemcpyToSymbol(detectorR,&h_detectorR, sizeof(float));
	cudaMemcpyToSymbol(sourceR,&h_sourceR, sizeof(float));
	cudaMemcpyToSymbol(sourceY,&h_sourceY, sizeof(float));

	cudaMemcpyToSymbol(ANGLES,&h_ANGLES, sizeof(int));

	cudaMemcpyToSymbol(BACKPRJ_ThreX,&h_BACKPRJ_ThreX, sizeof(int));
	cudaMemcpyToSymbol(BACKPRJ_ThreY,&h_BACKPRJ_ThreY, sizeof(int));
	cudaMemcpyToSymbol(BACKPRJ_GridX,&h_BACKPRJ_GridX, sizeof(int));
	cudaMemcpyToSymbol(BACKPRJ_GridY,&h_BACKPRJ_GridY, sizeof(int));
	cudaMemcpyToSymbol(nBatchXdim,&h_nBatchXdim, sizeof(int));

	cudaMemcpyToSymbol(BINSx,&h_BINSx, sizeof(int));
	cudaMemcpyToSymbol(BINSy,&h_BINSy, sizeof(int));
	cudaMemcpyToSymbol(Bsize_x,&h_Bsize_x, sizeof(float));
	cudaMemcpyToSymbol(Bsize_y,&h_Bsize_y, sizeof(float));

	cudaMemcpyToSymbol(nBatchBINSx,&h_nBatchBINSx, sizeof(int));
	cudaMemcpyToSymbol(nBatchBINSy,&h_nBatchBINSy, sizeof(int));
	cudaMemcpyToSymbol(PRJ_ThreX,&h_PRJ_ThreX, sizeof(int));
	cudaMemcpyToSymbol(PRJ_ThreY,&h_PRJ_ThreY, sizeof(int));
	cudaMemcpyToSymbol(PRJ_GridX,&h_PRJ_GridX, sizeof(int));
	cudaMemcpyToSymbol(PRJ_GridY,&h_PRJ_GridY, sizeof(int));

	printf("%d %d %d\n",h_IMGSIZx,h_IMGSIZy,h_IMGSIZz);

	h_index=new int[h_ANGLES];				//Store the access ordering scheme table
	h_angles=new float[h_ANGLES];

	Reconstruction();
	delete []h_angles;
	delete []h_index;
	printf("Recon successfully done!!\n");
	return 0;
}
