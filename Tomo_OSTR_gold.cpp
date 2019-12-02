/*
 * Copyright 2014 Yihuan Lu. All rights reserved.
 * Host code - All CPU functions are described here.
 */

#include <cuda_runtime.h>
#include <cufft.h>
//#include <cutil.h>
//#include <cutil_inline.h>
#include <helper_cuda.h>

#include "stdio.h"
#include "math.h"
#include "memory.h"

#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "Tomo_OSTR.h"
#include "readdata.h"

//#include <direct.h>


////////////////////////////////////////////////////////////////////////////////
// export C interface
typedef unsigned char	byte;
FILE	*fp;

extern "C"
void Reconstruction();

int		load_data(
	char *fname,
	void *buf,
	char type, 
	int col, 
	int row)
{
        const char *s;
	if ((fp = fopen(fname, "rb"))==NULL) {
		printf("Unable to open file :%s\n", fname);                
		return(0);
	}

	switch( type )
	{
		case	'i' : fread((char*)buf, sizeof(int)*col, row, fp);
			      break;
		case	'g' : //fread((char*)buf, sizeof(__int32)*col, row, fp);
			      break;
		case	's' : fread((char*)buf, sizeof(short)*col, row, fp);
			      break;
		case	'b' : fread((char*)buf, col, row, fp);
			      break;
		case	'f' : fread((char*)buf, sizeof(float)*col, row, fp);
				  break;
		default	    : fread((char*)buf, sizeof(float)*col, row, fp);
	}

	fclose(fp);
	return(1);
}

void	save_data(
	char *fname, 
	void *buf, 
	char type, 
	int col, 
	int row)
{
	if ( (fp = fopen(fname, "wb")) == NULL ){
		printf("Can't open file %s in written mode! \n", fname);
		exit(1);
	}
	switch( type )
	{
		case	'i' : fwrite((char*)buf, sizeof(int)*col, row, fp);
			      break;
		case	'g' : //fwrite((char*)buf, sizeof(__int32)*col, row, fp);
			      break;
		case	's' : fwrite((char*)buf, sizeof(short)*col, row, fp);
			      break;
		case	'b' : fwrite((char*)buf, col, row, fp);
			      break;
		case	'f' : fwrite((char*)buf, sizeof(float)*col, row, fp);
			      break;
		default	    : fwrite((char*)buf, sizeof(float)*col, row, fp);
	}
	fclose(fp);
}

void	x_y_flip(
	float *host_prjbuf_temp, 
	float *host_prjbuf_1view){
	int bin_ind, bin_ind_temp;
	int i,j;
	for(i=0;i<h_BINSx;i++){
		for(j=0;j<h_BINSy;j++){
			bin_ind_temp= j*h_BINSx+i;
			bin_ind = i*h_BINSy+j;
			host_prjbuf_1view[bin_ind]=host_prjbuf_temp[bin_ind_temp];
		}
	}
}

void   load_prj(
	float *prj_allangle,
	char  *prj_folder,
	char  *prj_name){
	float		*host_prj_1view_temp;	//h_BINSx*h_BINSy
	float		*host_prj_temp;		//h_BINSx*h_BINSy for x_y_flip reason
	int			b_size = h_BINSx*h_BINSy;
	int flag2=0;
	int i,s;
	int status =0;
	char prjname_origin[200], prjid[200], buffer[10];


	host_prj_1view_temp = (float *)calloc(b_size,sizeof(float) );
	if (host_prj_1view_temp == NULL){
		printf("memory problem of host_prj_1view_temp !!\n");
		system("PAUSE");
		exit(1);
	}
	host_prj_temp       = (float *)calloc( b_size,sizeof(float) );
	if (host_prj_temp == NULL){
		printf("memory problem of host_prj_temp !!\n");
		system("PAUSE");
		exit(1);
	}


	for (int viewangle=0; viewangle<h_ANGLES; viewangle++)
	{
		memset(host_prj_1view_temp,0,sizeof(float)*b_size);
		memset(host_prj_temp,0,sizeof(float)*b_size);
		s = h_index[viewangle]+1;

		//Loading projection data for angle s
sprintf(buffer, "%d", s);
		//itoa(s, buffer, 10);
		strcpy(prjid,prj_name);// Unfiltered: breast3prj
		if (s<1000){
			strcpy(prjid,prj_name);
			strcat(prjid,"0");
			if (s<100)	{
				strcpy(prjid,prj_name);
				strcat(prjid,"00");
				if (s<10)	{
					strcpy(prjid,prj_name);
					strcat(prjid,"000");}
			}
		}
		strcat(prjid, buffer);
		//strcat(prjid, ".IMA");

		strcpy(prjname_origin, prj_folder);
		strcat(prjname_origin, prjid);

		printf("current prj file name: %s\n",prjname_origin);//system("PAUSE");exit(1);
		//system("PAUSE");
		status = load_data(prjname_origin, host_prj_temp, 'f', 1, b_size);
		if( !status )  { printf("File %s is missing\n", prjname_origin); system("PAUSE"); exit(1);}

		x_y_flip(host_prj_temp, host_prj_1view_temp);  // EXTREMELY important

		for (i=0;i<h_BINSx*h_BINSy;i++) {
			prj_allangle[flag2*h_BINSx*h_BINSy + i] = host_prj_1view_temp[i];
		}  // all angle together

		flag2=flag2+1;

		//system("PAUSE");
	}

	free((void *)host_prj_1view_temp);
	free((void *)host_prj_temp);
}


// compute_yry now is on CPU, if this takes too long, we can further accelerate to compute on GPU
void  compute_yry(
	float *host_yry_allangle,
	float *host_prj_allangle,
	float *host_scat_allangle)
{
		long i=0;
		float dif=0;

		int all_b_size  =  h_ANGLES*h_BINSx*h_BINSy;

		for(i=0;i<all_b_size;++i){
			if (host_prj_allangle[i]== 0){
				host_yry_allangle[i]=0;
			}
			else{
				dif=host_prj_allangle[i]-host_scat_allangle[i];
				if (dif<=0){
					dif=host_prj_allangle[i];
				}
				host_yry_allangle[i]=(dif*dif)/host_prj_allangle[i];
			}
		}
}

// compute_gamma_yry now is on CPU, if this takes too long, we can further accelerate to compute on GPU
void compute_gamma_yry(
	float *host_gamma_yry_allangle,
	float *host_yry_allangle,
	float *host_gamma_allangle){
		long  i=0;
		int all_b_size  =  h_ANGLES*h_BINSx*h_BINSy;
		for(i=0;i<all_b_size;++i){
			host_gamma_yry_allangle[i]=host_yry_allangle[i]*host_gamma_allangle[i];
		}

}

void regroup_prj(
	float *host_uponregroup_allangle,
	float *host_allangle_tmp){
		int  i,j,k;
		int all_b_size  =  h_ANGLES*h_BINSx*h_BINSy;
		int ANGLES_per_sub;
		int sub_b_size;
		ANGLES_per_sub = h_ANGLES/h_subset_num;
		int b_size = h_BINSx*h_BINSy;
		
		cudaMemcpy(host_allangle_tmp,host_uponregroup_allangle,all_b_size*sizeof(float),cudaMemcpyHostToHost);
		//CUDA_SAFE_CALL(cudaMemcpy(host_allangle_tmp,host_uponregroup_allangle,all_b_size*sizeof(float),cudaMemcpyHostToHost));

		long flag=0;
		for(i=0;i<h_subset_num;++i){
			for(j=0;j<ANGLES_per_sub;++j){
				for (k=0;k<b_size;++k){
					host_uponregroup_allangle[flag]=host_allangle_tmp[(j*h_subset_num+i)*b_size+k];
					++flag;
				}
			}
		}
}


void compute_h(
	float *host_h_sub,
	float *host_prj_sub,
	float *host_blank_sub,
	float *host_line_sub,
	float *host_scat_sub){
		long  i=0;
		float y_tmp;
		int ANGLES_per_sub = h_ANGLES/h_subset_num;
		int sub_b_size  =  ANGLES_per_sub*h_BINSx*h_BINSy;
		for (i=0;i<sub_b_size;++i){
			y_tmp=host_blank_sub[i]*exp(-host_line_sub[i]);
			host_h_sub[i]=(host_prj_sub[i]/(y_tmp+host_scat_sub[i])-1)*y_tmp;

		}
}


int printit = 0;

void update_est(	
	float *host_est,
	float *host_capL,
	float *host_RD,
	float *host_d,
	float *host_RDD){
		int i   = 0;
		int f_size;
		f_size	=	h_IMGSIZx*h_IMGSIZy*h_IMGSIZz;

		//printf("f_size is %d", f_size);

		for (i=0; i<f_size; ++i){
			// Pranjal edit			
			/*if (host_RDD[i] == 0.0){
				host_RDD[i] = 0.001;
			}
			if (host_d[i] == 0.0){
				host_d[i] = 0.001;
			}*/
			host_est[i] = host_est[i]-(host_capL[i]+h_beta*host_RD[i])/(host_d[i]+2*h_beta*host_RDD[i]);

			if(printit == 1){
				printf("i = %d host_est =%f host_d %f h_beta %f host_RDD %f total %f\n", i, host_est[i], host_d[i], h_beta, host_RDD[i], host_d[i]+2*h_beta*host_RDD[i]);
			}
			
			if (host_est[i] < 0){
				host_est[i] = 0;
			}
		}
}



/*--------------------------------------------------------------------*/
//Load an 3D image (stored as series of 2D slice) into the variable buf.
void ConebeamCT_OSTR_LangePrior_GPU(){
	
	// [editing] put those global var in and comment them. e.g. h_ANGLES, h_subset etc.
	int			status;
	int			f_size, b_size, all_b_size, sub_b_size;

	// Variables on device
	float		*d_est;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz
	float		*d_prior;		//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz
	float		*d_d;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz  ;[editing] This is the precomputed 'd' term in the update equation. It is an estimate term of a inner updating from Fessler's paper.   
	float		*d_capL;		//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz  // this term is the bkprj of 'h' term in the iteration loop

	float		*d_RD;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz
	float		*d_RDD;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz
	

	float		*d_prj_1view;	//h_BINSx*h_BINSy
	float		*d_normprj;		//h_BINSx*h_BINSy
	float		*d_diff_line;	//h_BINSx*h_BINSy
	


	float		*d_line_sub;	 //ANGLES_per_sub*h_BINSx*h_BINSy; Note: ANGLES_per_sub will be calculated soon
	float		*d_h_sub;		 //ANGLES_per_sub*h_BINSx*h_BINSy; Note: ANGLES_per_sub will be calculated soon
	//float		*d_prj_allangle; //h_ANGLES*h_BINSx*h_BINSy; [editing] This term will only be used for compute 'yry' term then be deleted
	//float		*d_yry_allangle; //h_ANGLES*h_BINSx*h_BINSy; [editing] This term will only be used for compute 'd' term then be deleted
	//float		*d_scat_allangle;//h_ANGLES*h_BINSx*h_BINSy; [editing] This term will only be used for compute 'yry' term then be deleted
	float		*d_gamma_allangle;//h_ANGLES*h_BINSx*h_BINSy; [editing] This is the gamma term in the equation to precompute 'd' term.
								//								   The memory space will be located before others and *d_gamma_allangle will be computed before other variables' space being allocated, 
								//  								since 'd_gamma_allangle' is just a middle var to compute 'd' term. 'd' term is the one will be used in iteration loop.   
	float		*d_gamma_yry_allangle;//h_ANGLES*h_BINSx*h_BINSy; [editing] This is the gamma*yry term in the equation to precompute 'd' term. 


	//Variables on host
	float		*host_est;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz
	float		*host_d;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz  ; [editing] This is the precomputed 'd' term in the update equation. It is an estimate term of a inner updating from Fessler's paper.   
	float		*host_capL;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz ; map to d_capL;
	float		*host_RDD;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz
	float		*host_RD;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz

	//float		*host_prj_temp;				//h_BINSx*h_BINSy for x_y_flip reason
	//float		*host_prj_5view;			//5*h_BINSx*h_BINSy, this is perticular designed for GTX550Ti 2GB GPU since memory is not big enough to hold all prj images on board at same time. 
	float		*host_prj_allangle;			//h_BINSx*h_BINSy*h_ANGLES 
	float		*host_prj_sub_tmp;			//h_BINSx*h_BINSy*ANGLES_per_sub,

	float		*host_scat_allangle;		//h_BINSx*h_BINSy*h_ANGLES
	float		*host_blank_allangle;		//h_BINSx*h_BINSy*h_ANGLES
	float		*host_yry_allangle;			//h_BINSx*h_BINSy*h_ANGLES
	float		*host_gamma_allangle;		//h_BINSx*h_BINSy*h_ANGLES ; [editing] mapping to *d_gamma_allangle
	float		*host_gamma_yry_allangle;	//h_BINSx*h_BINSy*h_ANGLES ; [editing] mapping to *d_gamma_yry_allangle
	float		*host_allangle_tmp;			//h_BINSx*h_BINSy*h_ANGLES ; [editing]

	float		*host_line_sub;				//ANGLES_per_sub*h_BINSx*h_BINSy; Note: ANGLES_per_sub will be calculated soon
	float		*host_prj_sub;				//h_BINSx*h_BINSy*ANGLES_per_sub
	float		*host_scat_sub;				//h_BINSx*h_BINSy*ANGLES_per_sub
	float		*host_blank_sub;			//h_BINSx*h_BINSy*ANGLES_per_sub
	float		*host_h_sub;				//h_BINSx*h_BINSy*ANGLES_per_sub

	float		*d_angles;			// store angle vector 
	int			*d_index;			// h_index angle
	//int			*d_index_ori;			// h_index angle

	int			a,i,j,s, iter;
	float		z_xy_ratio;
	float		beta = h_beta;
	float		delta = h_delta;
	char		prjname_origin[200], prj_path[200], prjid[200] , estid[200] , buffer[10];

	int			angleStart, angleEnd, viewangle;
	int			ANGLES_per_sub; 


	f_size	       =	h_IMGSIZx*h_IMGSIZy*h_IMGSIZz;	// recon space
	b_size	       =	h_BINSx*h_BINSy;		//	detector space
	all_b_size     =  h_ANGLES*h_BINSx*h_BINSy;
	ANGLES_per_sub =  h_ANGLES/h_subset_num;
	sub_b_size     =  ANGLES_per_sub*b_size;

	printf("ANGLES_per_sub %d", ANGLES_per_sub);
	if (h_Vsize_x == h_Vsize_y){
		z_xy_ratio = h_Vsize_z/h_Vsize_x;
	}
	else{
		printf("h_Vsize_x does not equal to h_Vsize_y! prior can't be calculated. Please contact YL for modification.");
		exit(1);
	}
	


	printf("Before assign any big memory \n");
	//system("PAUSE");


	host_prj_allangle = (float *)calloc( all_b_size , sizeof(float) );if (host_prj_allangle == NULL){printf("memory problem of host_prj_allangle !!\n");system("PAUSE");exit(1);}
	host_scat_allangle = (float *)calloc( all_b_size , sizeof(float) );	if (host_scat_allangle == NULL){printf("memory problem of host_scat_allangle !!\n");system("PAUSE");exit(1);}
	memset(host_prj_allangle,0,sizeof(float)*all_b_size);
	memset(host_scat_allangle,0,sizeof(float)*all_b_size);

	host_yry_allangle = (float *)calloc( all_b_size , sizeof(float) );	if (host_yry_allangle == NULL){printf("memory problem of host_yry_allangle !!\n");system("PAUSE");exit(1);}
	host_gamma_allangle = (float *)calloc( all_b_size , sizeof(float) );	if (host_gamma_allangle == NULL){printf("memory problem of host_gamma_allangle !!\n");system("PAUSE");exit(1);}
	host_gamma_yry_allangle = (float *)calloc( all_b_size , sizeof(float) );	if (host_gamma_yry_allangle == NULL){printf("memory problem of host_gamma_yry_allangle !!\n");system("PAUSE");exit(1);}
	// [editing] compute 'yry' and copy to *d_yry_allangle
	memset(host_yry_allangle,0,sizeof(float)*all_b_size);
	memset(host_gamma_allangle,0,sizeof(float)*all_b_size);
	memset(host_gamma_yry_allangle,0,sizeof(float)*all_b_size);


	// load all angle projection data to *host_prj_allangle
	load_prj(host_prj_allangle,h_prj_folder,h_prj_name);

	// load all angle scatter data to *host_scat_allangle
	load_prj(host_scat_allangle,h_scat_folder,h_scat_name);


	compute_yry(host_yry_allangle, host_prj_allangle, host_scat_allangle);
	// [editing] END: compute 'yry' and copy to *d_yry_allangle

	


	cudaMalloc((void**) &d_angles, h_ANGLES*sizeof(float));//CUDA_SAFE_CALL(cudaMalloc((void**) &d_angles, h_ANGLES*sizeof(float)));
	cudaMemset(d_angles, 0, h_ANGLES*sizeof(float));//CUDA_SAFE_CALL(cudaMemset(d_angles, 0, h_ANGLES*sizeof(float)));

	cudaMalloc((void**) &d_index, h_ANGLES*sizeof(int));//CUDA_SAFE_CALL(cudaMalloc((void**) &d_index, h_ANGLES*sizeof(int)));
	cudaMemset(d_index, 0, h_ANGLES*sizeof(int));//CUDA_SAFE_CALL(cudaMemset(d_index, 0, h_ANGLES*sizeof(int)));

	//Copy the continuous values of rotating h_angles to device
	//CUDA_SAFE_CALL(cudaMemcpy(d_angles, h_angles, h_ANGLES*sizeof(float), cudaMemcpyHostToDevice) );
        cudaMemcpy(d_angles, h_angles, h_ANGLES*sizeof(float), cudaMemcpyHostToDevice);

	//Copy the order of h_angles (for OS-type algorithm) to device
	//CUDA_SAFE_CALL(cudaMemcpy(d_index, h_index, h_ANGLES*sizeof(int), cudaMemcpyHostToDevice) );
	cudaMemcpy(d_index, h_index, h_ANGLES*sizeof(int), cudaMemcpyHostToDevice);


	// [editing]: here we fwd prj to get 'gamma' term, for bigger obj/prj, we may need to fwdprj subangle set.
	//CUDA_SAFE_CALL(cudaMalloc((void**)&d_gamma_allangle, all_b_size*sizeof(float)));
	cudaMalloc((void**)&d_gamma_allangle, all_b_size*sizeof(float));

	if (d_gamma_allangle == NULL){
		printf("memory problem of d_gamma_allangle !!\n");
		system("PAUSE");
		exit(1);
	}
	//CUDA_SAFE_CALL(cudaMemset(d_gamma_allangle, 0, all_b_size*sizeof(float)));
	cudaMemset(d_gamma_allangle, 0, all_b_size*sizeof(float));


	// calculate 'gamma' term by fwdprj all '1' obj
	angleStart = 0;
	angleEnd   = h_ANGLES; // here is a non-intuitive thing, angleStart is the REAL C starting index, but angleEnd is '<angleEnd'; which does not follow common sense. Here total number of angles that will be projected is angleEnd-angleStart
	fprojectCB_1R_GPU_OSTR_normprj(
		d_gamma_allangle,
		d_angles,
		d_index,
		angleStart,
		angleEnd);

	
	cudaError_t err1 = cudaGetLastError();
	if (err1 != cudaSuccess) 
    		printf("Error in CPP 1 part : %s\n", cudaGetErrorString(err1));

	cudaMemcpy(host_gamma_allangle, d_gamma_allangle, all_b_size*sizeof(float), cudaMemcpyDeviceToHost);	


	/*
	// Pranjal Code
	int saveid = 1000;
	sprintf(buffer,"%d",saveid);

	strcpy(estid,h_recon_folder);
	strcat(estid,h_recon_name);
	strcat(estid,buffer);

        save_data(
		estid,
		host_gamma_allangle,
		'f',
		h_ANGLES,
		h_BINSx*h_BINSy);
	exit(0);*/


	cudaError_t err2 = cudaGetLastError();
	if (err2 != cudaSuccess) 
    		printf("Error in CPP 2 part : %s\n", cudaGetErrorString(err2));

        cudaFree(d_gamma_allangle);

	cudaError_t err3 = cudaGetLastError();
	if (err3 != cudaSuccess) 
    		printf("Error in CPP 3 part : %s\n", cudaGetErrorString(err3));	
	

	// Pranjal edit
	compute_gamma_yry(host_gamma_yry_allangle, host_yry_allangle, host_gamma_allangle); // [host_gamma_yry_allangle test BF]

		
	printf("After computing gamma_yry; Before free yry and gamma\n");


	free((void *)host_yry_allangle);
	free((void *)host_gamma_allangle);

	printf("After free yry and gamma \n");

	cudaMalloc((void**)&d_d, f_size*sizeof(float));
	if (d_d == NULL){
		printf("memory problem of d_d !!\n");
		system("PAUSE");
		exit(1);
	}
	cudaMemset(d_d, 0, f_size*sizeof(float));

	cudaMalloc((void**)&d_gamma_yry_allangle, all_b_size*sizeof(float));
	cudaMemset(d_gamma_yry_allangle, 0,       all_b_size*sizeof(float));
	
	cudaMemcpy(d_gamma_yry_allangle, host_gamma_yry_allangle, all_b_size*sizeof(float), cudaMemcpyHostToDevice);
	free((void *)host_gamma_yry_allangle); // we won't need host_gamma_yry_allangle anymore
	
	cudaError_t err4 = cudaGetLastError();
	if (err4 != cudaSuccess) 
    		printf("Error in CPP 4 part : %s\n", cudaGetErrorString(err4));

	printf("After free gamma_yry \n");

	angleStart = 0;
	angleEnd   = h_ANGLES;
	
	printf("Before compute d \n");
	bprojectCB_GPU_SBP(
		(float *)d_d, 
		(float *)d_gamma_yry_allangle, 
		d_index, 
		d_angles, 
		angleStart,
		angleEnd); 

	cudaError_t err5 = cudaGetLastError();
	if (err5 != cudaSuccess) 
    		printf("Error in CPP 5 part : %s\n", cudaGetErrorString(err5));





	//CUDA_SAFE_CALL(cudaFree(d_gamma_yry_allangle));
	cudaFree(d_gamma_yry_allangle);
	printf("After compute d_gamma_yry_allangle \n");

	host_d = (float *)calloc( f_size,sizeof(float) );
	if (host_d == NULL){
		printf("memory problem of host_d !!\n");
		system("PAUSE");
		exit(1);
	}
	memset(host_d,0,sizeof(float)*f_size);//[test]

	cudaMemcpy(host_d, d_d, f_size*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_d);
	//CUDA_SAFE_CALL(cudaMemcpy(host_d,d_d,f_size*sizeof(float),cudaMemcpyDeviceToHost));	
	//CUDA_SAFE_CALL(cudaFree(d_d));// we free d_d since it will run update on CPU.

	printf("After free d \n");

	/////////// load in the initial guess for d_est, the recon////////////
	// Note: we allocate d_est here for GPU memory budget reason (after freeing other GPU space which won't be used from now)
	host_est = (float *)calloc( f_size,sizeof(float) );if (host_est == NULL){printf("memory problem of host_est !!\n");system("PAUSE");exit(1);}
	memset(host_est, 0, sizeof(float)*f_size);

	cudaMalloc((void**)&d_est, f_size*sizeof(float));	
	//CUDA_SAFE_CALL(cudaMalloc((void**)&d_est, f_size*sizeof(float)));if (d_est == NULL){printf("memory problem of d_est !!\n");system("PAUSE");exit(1);} 
	//CUDA_SAFE_CALL(cudaMemset(d_est, 0, f_size*sizeof(float)));
	cudaMemset(d_est, 0, f_size*sizeof(float));

	if (h_ini_name==NULL){
		memset(host_est, 0.001, sizeof(float)*f_size);
	}
	else{
		status = load_data(h_ini_name, host_est, 'f', h_IMGSIZz, h_IMGSIZx*h_IMGSIZy);
		if(!status){
			printf("Cannot load initial guess stored at %s to do initializing, initial value will be set as 0.001!\n", h_ini_name);
			memset(host_est,0.001,sizeof(float)*f_size);
		}
	}

	cudaMemcpy(d_est, host_est, f_size*sizeof(float), cudaMemcpyHostToDevice);
	/////////// END :load in the initial guess for d_est, the recon////////////



	// load all angle blankscan data to *host_blank_allangle
	host_blank_allangle = (float *)calloc( all_b_size , sizeof(float) );	
	if (host_blank_allangle == NULL){
		printf("memory problem of host_blank_allangle !!\n");
		system("PAUSE");
		exit(1);
	}

	memset(host_blank_allangle, 0, sizeof(float)*all_b_size);
	load_prj(host_blank_allangle, h_blank_folder, h_blank_name);


	//////////////////// Important modification here///////////////////////
	// Current version OS angle order is optimized: 
	// If total angle number is 25, and we use 5 subsets, then
	// we group the subsets as following: first subset: angle 0,4,9,14,19; second subset: 1,5,10,15,20 etc.
	// 
	// Here we re-order the h_index, but do not re-order h_angles. Reason: h_index is used to fetch h_angles. e.g. if h_index is 0,1,2,3,4, and h_angles is -0.3,-0.15,0,0.15,0.3, then when later 
	// fwd prj or bkprj happens, the angles in order will be: -0.3,-0.15,0,0.15,0.3; In another case, if h_index is 0,2,1,4,3, h_angles is still -0.3,-0.15,0,0.15,0.3, but then when later 
	// fwd prj or bkprj happens, the angles in order will be: -0.3,0,-0.15,0.3,0.15

	host_allangle_tmp = (float *)calloc( all_b_size , sizeof(float) );	if (host_allangle_tmp == NULL){printf("memory problem of host_allangle_tmp !!\n");system("PAUSE");exit(1);}
	memset(host_allangle_tmp,0,sizeof(float)*all_b_size);

	regroup_prj(host_prj_allangle,host_allangle_tmp);
	memset(host_allangle_tmp,0,sizeof(float)*all_b_size);

	regroup_prj(host_scat_allangle,host_allangle_tmp);
	memset(host_allangle_tmp,0,sizeof(float)*all_b_size);

	regroup_prj(host_blank_allangle,host_allangle_tmp);
	memset(host_allangle_tmp,0,sizeof(float)*all_b_size);

	free((void *)host_allangle_tmp); // we won't need host_allangle_tmp anymore

	int *index_tmp;
	index_tmp = (int *)calloc( h_ANGLES , sizeof(int) );
	memset(index_tmp,0,sizeof(int)*h_ANGLES);
	for (i=0;i<h_ANGLES;++i){
		index_tmp[i]=h_index[i];
	}
	int flag = 0;
	for(i=0;i<h_subset_num;++i){
		for(j=0;j<ANGLES_per_sub;++j){
			h_index[flag]=index_tmp[j*h_subset_num+i];
			++flag;
		}
	}
	free((void *) index_tmp);
	cudaMemcpy(d_index, h_index, h_ANGLES*sizeof(int), cudaMemcpyHostToDevice);
	//CUDA_SAFE_CALL(cudaMemcpy(d_index, h_index, h_ANGLES*sizeof(int), cudaMemcpyHostToDevice) ); // update d_index with new order scheme
	///////////////////END of Important modification here///////////////////////////////////////////




	host_line_sub  = (float *)calloc( sub_b_size , sizeof(float) );	if (host_line_sub == NULL){printf("memory problem of host_line_sub !!\n");system("PAUSE");exit(1);}
	host_prj_sub   = (float *)calloc( sub_b_size , sizeof(float) );	if (host_prj_sub == NULL){printf("memory problem of host_prj_sub !!\n");system("PAUSE");exit(1);}
	host_scat_sub  = (float *)calloc( sub_b_size , sizeof(float) );	if (host_scat_sub == NULL){printf("memory problem of host_scat_sub !!\n");system("PAUSE");exit(1);}
	host_blank_sub = (float *)calloc( sub_b_size , sizeof(float) );	if (host_blank_sub == NULL){printf("memory problem of host__sub !!\n");system("PAUSE");exit(1);}

	memset(host_line_sub,0,sizeof(float)*sub_b_size);
	memset(host_prj_sub,0,sizeof(float)*sub_b_size);
	memset(host_scat_sub,0,sizeof(float)*sub_b_size);
	memset(host_blank_sub,0,sizeof(float)*sub_b_size);


	// allocate CPU memo for host_h_sub;
	host_h_sub = (float *)calloc( sub_b_size , sizeof(float) );	if (host_h_sub == NULL){printf("memory problem of host_h_sub !!\n");system("PAUSE");exit(1);}
	memset(host_h_sub,0,sizeof(float)*sub_b_size);

	// allocate CPU memo for host_capL, host_RDD, host_RD;
	host_capL = (float *)calloc( f_size , sizeof(float) );	if (host_capL == NULL){printf("memory problem of host_capL !!\n");system("PAUSE");exit(1);}
	memset(host_capL,0,sizeof(float)*f_size);

	host_RDD = (float *)calloc( f_size,sizeof(float) );if (host_RDD == NULL){printf("memory problem of host_RDD !!\n");system("PAUSE");exit(1);}
	memset(host_RDD,0,sizeof(float)*f_size);//[test]

	host_RD = (float *)calloc( f_size,sizeof(float) );if (host_RD == NULL){printf("memory problem of host_RD !!\n");system("PAUSE");exit(1);}
	memset(host_RD,0,sizeof(float)*f_size);//[test]



	cudaMalloc((void**)&d_line_sub, sub_b_size*sizeof(float));
	cudaMemset(d_line_sub, 0, sub_b_size*sizeof(float));
	// Note: we allocate d_line_sub here for GPU memory budget reason (after freeing other GPU space which won't be used from now)
	


	cudaMalloc((void**)&d_h_sub, sub_b_size*sizeof(float));
	cudaMemset(d_h_sub, 0, sub_b_size*sizeof(float));
	// allocate GPU memo for d_h_sub;
		

	cudaMalloc((void**)&d_capL, f_size*sizeof(float));
	cudaMemset(d_capL, 0, f_size*sizeof(float));
	// allocate GPU memo for d_capL;
		

	cudaMalloc((void**)&d_RDD, f_size*sizeof(float));
	cudaMemset(d_RDD, 0, f_size*sizeof(float));
	// allocate GPU memo for d_RDD;
		

	cudaMalloc((void**)&d_RD, f_size*sizeof(float));
	cudaMemset(d_RD, 0, f_size*sizeof(float));
	// allocate GPU memo for d_capL;
		


	/* //PRANJAL CODE
	int saveid = 1000;
	sprintf(buffer,"%d",saveid);

	strcpy(estid,h_recon_folder);
	strcat(estid,h_recon_name);
	strcat(estid,buffer);

        save_data(
		estid,
		host_gamma_yry_allangle,
		'f',
		h_ANGLES,
		h_BINSx*h_BINSy);
	exit(0);*/

	


	printf("Before iteration loop \n");
	//////////////////// start iteration loop///////////////////
	for (iter=0; iter < h_iter_num; ++iter){
		for (a=0; a<h_subset_num; ++a){

			printf("a is %d\n", a);			
			angleStart = a*ANGLES_per_sub;
			angleEnd   = (a+1)*ANGLES_per_sub;

			// fwdprj d_est to get d_line_sub
			fprojectCB_1R_GPU_OSTR_cos(
				d_est,
				d_line_sub,
				d_angles,
				d_index,
				angleStart,
				angleEnd);

			cudaMemcpy(host_line_sub, d_line_sub, sub_b_size*sizeof(float), cudaMemcpyDeviceToHost);

			// Now copy subset data to host_scat_sub/host_blank_sub/host_prj_sub/ create h term 20:22 April 2014 
			for(i=0;i<sub_b_size;i++){
				host_prj_sub[i]   = host_prj_allangle[angleStart*b_size+i];
			}
			for(i=0;i<sub_b_size;i++){
				host_scat_sub[i]  = host_scat_allangle[angleStart*b_size+i];
			}
			for(i=0;i<sub_b_size;i++){
				host_blank_sub[i] = host_blank_allangle[angleStart*b_size+i];
			}


			// here we compute 'h' term on CPU, if GPU memory allows, we can further accelerate this using GPU 
			compute_h(host_h_sub, host_prj_sub, host_blank_sub, host_line_sub, host_scat_sub);
			cudaMemcpy(d_h_sub, host_h_sub, sub_b_size*sizeof(float), cudaMemcpyHostToDevice);

			// now we bkprj 'h' to get 'capL' term, Note: we calculate d_capL without multiplying the subset number, we will do the multiplication for host_capL;
			bprojectCB_GPU_SBP(
				(float *)d_capL, 
				(float *)d_h_sub, 
				d_index, 
				d_angles, 
				angleStart,
				angleEnd); 

			cudaMemcpy(host_capL, d_capL, f_size*sizeof(float), cudaMemcpyDeviceToHost);

			for(i=0; i<f_size; i++){
				host_capL[i] = h_subset_num*host_capL[i];
			}



			prior_GPU_OSTR(
				d_RDD,
				d_RD,
				d_est,
				h_delta,
				z_xy_ratio);

	
			cudaError_t err9 = cudaGetLastError();
			if (err9 != cudaSuccess) 
    				printf("Error after memcpy: %s\n", cudaGetErrorString(err9));

			cudaMemcpy(host_RDD, d_RDD, f_size*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(host_RD,  d_RD,  f_size*sizeof(float), cudaMemcpyDeviceToHost);


			// Pranjal Edit
			/*int saveid = 555+iter;
			sprintf(buffer,"%d", saveid);

			//itoa(saveid,buffer,10);
			strcpy(estid,h_recon_folder);
			strcat(estid,h_recon_name);
			strcat(estid,buffer);


			save_data(
				estid,
				host_RDD,
				'f',
				h_IMGSIZz,
				h_IMGSIZx*h_IMGSIZy);*/



			if(iter == 1){
				printit = 0;
                        }
			//if(iter  == 0){
			   update_est(	
				host_est,
				host_capL,
				host_RD,
				host_d,
				host_RDD);
			//}

			

			/*int saveid = 1000+iter;
			sprintf(buffer,"%d",saveid);

			strcpy(estid,h_recon_folder);
			strcat(estid,h_recon_name);
			strcat(estid,buffer);

			save_data(
				estid,
				host_line_sub,
				'f',
				h_ANGLES,
				h_BINSx*h_BINSy);*/



			cudaMemcpy(d_est, host_est, f_size*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(host_est, d_est, f_size*sizeof(float), cudaMemcpyDeviceToHost);

			cudaMemset(d_line_sub, 0, sub_b_size*sizeof(float));
			cudaMemset(d_capL, 0, f_size*sizeof(float));
			cudaMemset(d_RDD, 0, f_size*sizeof(float));
			cudaMemset(d_RD, 0, f_size*sizeof(float));

			memset(host_line_sub,0,sizeof(float)*sub_b_size);
			memset(host_h_sub,0,sizeof(float)*sub_b_size);
			memset(host_prj_sub,0,sizeof(float)*sub_b_size);
			memset(host_scat_sub,0,sizeof(float)*sub_b_size);
			memset(host_blank_sub,0,sizeof(float)*sub_b_size);

			memset(host_capL,0,sizeof(float)*f_size);
			memset(host_RDD,0,sizeof(float)*f_size);
			memset(host_RD,0,sizeof(float)*f_size);

			
			
			printf("Finished: %d sub-iter, %d iter \n",a+1,iter+1);



			// Pranjal Code

			//system("PAUSE");

		}

		int saveid = iter;
		sprintf(buffer,"%d",saveid);

		//itoa(saveid,buffer,10);
		strcpy(estid,h_recon_folder);
		strcat(estid,h_recon_name);
		strcat(estid,buffer);

		if ((iter+1)%h_IO_Iter==0){
			save_data(
				estid,
				host_est,
				'f',
				h_IMGSIZz,
				h_IMGSIZx*h_IMGSIZy);
		}

	}
	/////////////////////////////////END of Iteration loop///////////////////////////////////////


	printf("After iteration loop; Before free memory \n");
	//system("PAUSE");

	cudaFree(d_capL);
	cudaFree(d_RDD);
	cudaFree(d_RD);
	cudaFree(d_index);
	cudaFree(d_angles);
	cudaFree(d_est);
	cudaFree(d_line_sub);

	free((void *)host_prj_allangle);
	free((void *)host_scat_allangle);
	free((void *)host_d);
	free((void *)host_est);
	free((void *)host_line_sub);
	free((void *)host_prj_sub);
	free((void *)host_scat_sub);
	free((void *)host_blank_sub);
	free((void *)host_h_sub);
	free((void *)host_capL);
	free((void *)host_RDD);
	free((void *)host_RD);

	printf("After free memory \n");


}

void ConebeamCT_OSTR_Q2Prior_GPU(){
	
	// [editing] put those global var in and comment them. e.g. h_ANGLES, h_subset etc.
	int			status;
	int			f_size, b_size, all_b_size, sub_b_size;

	// Variables on device
	float		*d_est;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz
	float		*d_prior;		//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz
	float		*d_d;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz  ;[editing] This is the precomputed 'd' term in the update equation. It is an estimate term of a inner updating from Fessler's paper.   
	float		*d_capL;		//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz  // this term is the bkprj of 'h' term in the iteration loop

	float		*d_RD;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz
	float		*d_RDD;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz
	

	float		*d_prj_1view;	//h_BINSx*h_BINSy
	float		*d_normprj;		//h_BINSx*h_BINSy
	float		*d_diff_line;	//h_BINSx*h_BINSy
	


	float		*d_line_sub;	 //ANGLES_per_sub*h_BINSx*h_BINSy; Note: ANGLES_per_sub will be calculated soon
	float		*d_h_sub;		 //ANGLES_per_sub*h_BINSx*h_BINSy; Note: ANGLES_per_sub will be calculated soon
	//float		*d_prj_allangle; //h_ANGLES*h_BINSx*h_BINSy; [editing] This term will only be used for compute 'yry' term then be deleted
	//float		*d_yry_allangle; //h_ANGLES*h_BINSx*h_BINSy; [editing] This term will only be used for compute 'd' term then be deleted
	//float		*d_scat_allangle;//h_ANGLES*h_BINSx*h_BINSy; [editing] This term will only be used for compute 'yry' term then be deleted
	float		*d_gamma_allangle;//h_ANGLES*h_BINSx*h_BINSy; [editing] This is the gamma term in the equation to precompute 'd' term.
								//								   The memory space will be located before others and *d_gamma_allangle will be computed before other variables' space being allocated, 
								//  								since 'd_gamma_allangle' is just a middle var to compute 'd' term. 'd' term is the one will be used in iteration loop.   
	float		*d_gamma_yry_allangle;//h_ANGLES*h_BINSx*h_BINSy; [editing] This is the gamma*yry term in the equation to precompute 'd' term. 


	//Variables on host
	float		*host_est;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz
	float		*host_d;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz  ; [editing] This is the precomputed 'd' term in the update equation. It is an estimate term of a inner updating from Fessler's paper.   
	float		*host_capL;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz ; map to d_capL;
	float		*host_RDD;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz
	float		*host_RD;			//h_IMGSIZx*h_IMGSIZy*h_IMGSIZz

	//float		*host_prj_temp;				//h_BINSx*h_BINSy for x_y_flip reason
	//float		*host_prj_5view;			//5*h_BINSx*h_BINSy, this is perticular designed for GTX550Ti 2GB GPU since memory is not big enough to hold all prj images on board at same time. 
	float		*host_prj_allangle;			//h_BINSx*h_BINSy*h_ANGLES 
	float		*host_prj_sub_tmp;			//h_BINSx*h_BINSy*ANGLES_per_sub,

	float		*host_scat_allangle;		//h_BINSx*h_BINSy*h_ANGLES
	float		*host_blank_allangle;		//h_BINSx*h_BINSy*h_ANGLES
	float		*host_yry_allangle;			//h_BINSx*h_BINSy*h_ANGLES
	float		*host_gamma_allangle;		//h_BINSx*h_BINSy*h_ANGLES ; [editing] mapping to *d_gamma_allangle
	float		*host_gamma_yry_allangle;	//h_BINSx*h_BINSy*h_ANGLES ; [editing] mapping to *d_gamma_yry_allangle
	float		*host_allangle_tmp;			//h_BINSx*h_BINSy*h_ANGLES ; [editing]

	float		*host_line_sub;				//ANGLES_per_sub*h_BINSx*h_BINSy; Note: ANGLES_per_sub will be calculated soon
	float		*host_prj_sub;				//h_BINSx*h_BINSy*ANGLES_per_sub
	float		*host_scat_sub;				//h_BINSx*h_BINSy*ANGLES_per_sub
	float		*host_blank_sub;			//h_BINSx*h_BINSy*ANGLES_per_sub
	float		*host_h_sub;				//h_BINSx*h_BINSy*ANGLES_per_sub

	float		*d_angles;			// store angle vector 
	int			*d_index;			// h_index angle
	//int			*d_index_ori;			// h_index angle

	int			a,i,j,s, iter;
	float		z_xy_ratio;
	float		beta = h_beta;
	float		delta = h_delta;
	char		prjname_origin[200], prj_path[200], prjid[200] , estid[200] , buffer[10];

	int			angleStart, angleEnd, viewangle;
	int			ANGLES_per_sub; 


	f_size	=	h_IMGSIZx*h_IMGSIZy*h_IMGSIZz;	// recon space
	b_size	=	h_BINSx*h_BINSy;		//	detector space
	all_b_size  =  h_ANGLES*h_BINSx*h_BINSy;
	ANGLES_per_sub = h_ANGLES/h_subset_num;
	sub_b_size = ANGLES_per_sub*b_size;

	if (h_Vsize_x==h_Vsize_y){
		z_xy_ratio=h_Vsize_z/h_Vsize_x;
	}else{
		printf("h_Vsize_x does not equal to h_Vsize_y! prior can't be calculated. Please contact YL for modification.");
		exit(1);
	}
	


	printf("Before assign any big memory \n");
	//system("PAUSE");


	host_prj_allangle = (float *)calloc( all_b_size , sizeof(float) );
	if (host_prj_allangle == NULL){
		printf("memory problem of host_prj_allangle !!\n");
		system("PAUSE");
		exit(1);
	}
	host_scat_allangle = (float *)calloc( all_b_size , sizeof(float) );	
	if (host_scat_allangle == NULL){
		printf("memory problem of host_scat_allangle !!\n");
		system("PAUSE");
		exit(1);
	}
	memset(host_prj_allangle,0,sizeof(float)*all_b_size);
	memset(host_scat_allangle,0,sizeof(float)*all_b_size);

	host_yry_allangle = (float *)calloc( all_b_size , sizeof(float) );	if (host_yry_allangle == NULL){printf("memory problem of host_yry_allangle !!\n");system("PAUSE");exit(1);}
	host_gamma_allangle = (float *)calloc( all_b_size , sizeof(float) );	if (host_gamma_allangle == NULL){printf("memory problem of host_gamma_allangle !!\n");system("PAUSE");exit(1);}
	host_gamma_yry_allangle = (float *)calloc( all_b_size , sizeof(float) );	if (host_gamma_yry_allangle == NULL){printf("memory problem of host_gamma_yry_allangle !!\n");system("PAUSE");exit(1);}
	// [editing] compute 'yry' and copy to *d_yry_allangle
	memset(host_yry_allangle,0,sizeof(float)*all_b_size);
	memset(host_gamma_allangle,0,sizeof(float)*all_b_size);
	memset(host_gamma_yry_allangle,0,sizeof(float)*all_b_size);


	// load all angle projection data to *host_prj_allangle
	load_prj(host_prj_allangle,h_prj_folder,h_prj_name);

	// load all angle scatter data to *host_scat_allangle
	load_prj(host_scat_allangle,h_scat_folder,h_scat_name);

	compute_yry(host_yry_allangle,host_prj_allangle,host_scat_allangle); // [test BF]
	//CUDA_SAFE_CALL(cudaMemcpy(d_yry_allangle,host_yry_allangle,all_b_size*sizeof(float),cudaMemcpyHostToDevice));
	// [editing] END: compute 'yry' and copy to *d_yry_allangle

cudaMalloc((void**) &d_angles, h_ANGLES*sizeof(float));
cudaMemset(d_angles, 0, h_ANGLES*sizeof(float));
	//CUDA_SAFE_CALL(cudaMalloc((void**) &d_angles, h_ANGLES*sizeof(float)));
	//CUDA_SAFE_CALL(cudaMemset(d_angles, 0, h_ANGLES*sizeof(float)));

cudaMalloc((void**) &d_index, h_ANGLES*sizeof(int));
cudaMemset(d_index, 0, h_ANGLES*sizeof(int));
	//CUDA_SAFE_CALL(cudaMalloc((void**) &d_index, h_ANGLES*sizeof(int)));
	//CUDA_SAFE_CALL(cudaMemset(d_index, 0, h_ANGLES*sizeof(int)));

	//Copy the continuous values of rotating h_angles to device
	//CUDA_SAFE_CALL(cudaMemcpy(d_angles, h_angles, h_ANGLES*sizeof(float), cudaMemcpyHostToDevice) );
cudaMemcpy(d_angles, h_angles, h_ANGLES*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_index, h_index, h_ANGLES*sizeof(int), cudaMemcpyHostToDevice);

	//Copy the order of h_angles (for OS-type algorithm) to device
	//CUDA_SAFE_CALL(cudaMemcpy(d_index, h_index, h_ANGLES*sizeof(int), cudaMemcpyHostToDevice) );



cudaMalloc((void**)&d_gamma_allangle, all_b_size*sizeof(float));
cudaMemset(d_gamma_allangle, 0, all_b_size*sizeof(float));
	// [editing]: here we fwd prj to get 'gamma' term, for bigger obj/prj, we may need to fwdprj subangle set.
	//CUDA_SAFE_CALL(cudaMalloc((void**)&d_gamma_allangle, all_b_size*sizeof(float)));if (d_gamma_allangle == NULL){printf("memory problem of d_gamma_allangle !!\n");system("PAUSE");exit(1);}
	//CUDA_SAFE_CALL(cudaMemset(d_gamma_allangle, 0, all_b_size*sizeof(float)));



	// calculate 'gamma' term by fwdprj all '1' obj
	angleStart = 0;
	angleEnd   = h_ANGLES;; // here is a non-intuitive thing, angleStart is the REAL C starting index, but angleEnd is '<angleEnd'; which does not follow common sense. Here total number of angles that will be projected is angleEnd-angleStart
	fprojectCB_1R_GPU_OSTR_normprj(
		d_gamma_allangle,
		d_angles,
		d_index,
		angleStart,
		angleEnd);
cudaMemcpy(host_gamma_allangle, d_gamma_allangle, all_b_size*sizeof(float), cudaMemcpyDeviceToHost);
cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) 
    				printf("Error after memcpy: %s\n", cudaGetErrorString(err));


cudaFree(d_gamma_allangle);
	//CUDA_SAFE_CALL(cudaMemcpy(host_gamma_allangle,d_gamma_allangle,all_b_size*sizeof(float),cudaMemcpyDeviceToHost));
	//CUDA_SAFE_CALL(cudaFree(d_gamma_allangle)); // We free d_gamma_allangle here, since it will not be used in the future (future we only use partial angles). Note: for limited GPU memory, this GPU-memory-free step could be very important.

	compute_gamma_yry(host_gamma_yry_allangle,host_yry_allangle,host_gamma_allangle); // [host_gamma_yry_allangle test BF]

	printf("After computing gamma_yry; Before free yry and gamma\n");
	//system("PAUSE");


	free((void *)host_yry_allangle);
	free((void *)host_gamma_allangle);

	printf("After free yry and gamma \n");
	//system("PAUSE");


	cudaMalloc((void**)&d_d, f_size*sizeof(float));
	cudaMemset(d_d, 0, f_size*sizeof(float));

	//CUDA_SAFE_CALL(cudaMalloc((void**)&d_d, f_size*sizeof(float)));if (d_d == NULL){printf("memory problem of d_d !!\n");system("PAUSE");exit(1);}
	//CUDA_SAFE_CALL(cudaMemset(d_d, 0, f_size*sizeof(float)));

	cudaMalloc((void**)&d_gamma_yry_allangle, all_b_size*sizeof(float));//CUDA_SAFE_CALL(cudaMalloc((void**)&d_gamma_yry_allangle, all_b_size*sizeof(float)));if (d_gamma_yry_allangle == NULL){printf("memory problem of d_gamma_yry_allangle !!\n");system("PAUSE");exit(1);}
	cudaMemset(d_gamma_yry_allangle, 0, all_b_size*sizeof(float));//CUDA_SAFE_CALL(cudaMemset(d_gamma_yry_allangle, 0, all_b_size*sizeof(float)));
	cudaMemcpy(d_gamma_yry_allangle, host_gamma_yry_allangle, all_b_size*sizeof(float), cudaMemcpyHostToDevice);//CUDA_SAFE_CALL(cudaMemcpy(d_gamma_yry_allangle, host_gamma_yry_allangle, all_b_size*sizeof(float), cudaMemcpyHostToDevice));
	free((void *)host_gamma_yry_allangle); // we won't need host_gamma_yry_allangle anymore
	
	printf("After free gamma_yry \n");
	//system("PAUSE");


	/*
	iter=91;
	itoa(iter,buffer,10);
	strcpy(estid,h_recon_folder);
	strcat(estid,h_recon_name);
	strcat(estid,buffer);
	save_data(
		estid,
		host_gamma_yry_allangle,
		'f',
		h_ANGLES,
		h_BINSx*h_BINSy);
	*/


	angleStart=0;
	angleEnd=h_ANGLES;
	
		printf("Before compute d \n");
	bprojectCB_GPU_SBP(
		(float *)d_d, 
		(float *)d_gamma_yry_allangle, 
		d_index, 
		d_angles, 
		angleStart,
		angleEnd); 
	//CUDA_SAFE_CALL(cudaFree(d_gamma_yry_allangle));
	cudaFree(d_gamma_yry_allangle);
	printf("After compute d_gamma_yry_allangle \n");

	host_d = (float *)calloc( f_size,sizeof(float) );if (host_d == NULL){printf("memory problem of host_d !!\n");system("PAUSE");exit(1);}
	memset(host_d,0,sizeof(float)*f_size);//[test]

	cudaMemcpy(host_d,d_d,f_size*sizeof(float),cudaMemcpyDeviceToHost);
	cudaFree(d_d);

	//CUDA_SAFE_CALL(cudaMemcpy(host_d,d_d,f_size*sizeof(float),cudaMemcpyDeviceToHost));	
	//CUDA_SAFE_CALL(cudaFree(d_d));// we free d_d since it will run update on CPU.

	printf("After free d \n");

	/////////// load in the initial guess for d_est, the recon////////////
	// Note: we allocate d_est here for GPU memory budget reason (after freeing other GPU space which won't be used from now)
	host_est = (float *)calloc( f_size,sizeof(float) );if (host_est == NULL){printf("memory problem of host_est !!\n");system("PAUSE");exit(1);}
	memset(host_est,0,sizeof(float)*f_size);//[test]

	cudaMalloc((void**)&d_est, f_size*sizeof(float));
	cudaMemset(d_est, 0, f_size*sizeof(float));

	//CUDA_SAFE_CALL(cudaMalloc((void**)&d_est, f_size*sizeof(float)));if (d_est == NULL){printf("memory problem of d_est !!\n");system("PAUSE");exit(1);} 
	//CUDA_SAFE_CALL(cudaMemset(d_est, 0, f_size*sizeof(float)));

	if (h_ini_name==NULL){
		memset(host_est,0.001,sizeof(float)*f_size);
	}
	else{
		status = load_data(h_ini_name, host_est, 'f', h_IMGSIZz, h_IMGSIZx*h_IMGSIZy);
		if(!status){
			printf("Cannot load initial guess stored at %s to do initializing, initial value will be set as 0.001!\n", h_ini_name);
			memset(host_est,0.001,sizeof(float)*f_size);
		}
	}

	cudaMemcpy(d_est,host_est,f_size*sizeof(float),cudaMemcpyHostToDevice);
	//CUDA_SAFE_CALL(cudaMemcpy(d_est,host_est,f_size*sizeof(float),cudaMemcpyHostToDevice));// Copy initial guess to device 
	/////////// END :load in the initial guess for d_est, the recon////////////



	// load all angle blankscan data to *host_blank_allangle
	host_blank_allangle = (float *)calloc( all_b_size , sizeof(float) );	if (host_blank_allangle == NULL){printf("memory problem of host_blank_allangle !!\n");system("PAUSE");exit(1);}
	memset(host_blank_allangle,0,sizeof(float)*all_b_size);
	load_prj(host_blank_allangle,h_blank_folder,h_blank_name);


	//////////////////// Important modification here///////////////////////
	// Current version OS angle order is optimized: 
	// If total angle number is 25, and we use 5 subsets, then
	// we group the subsets as following: first subset: angle 0,4,9,14,19; second subset: 1,5,10,15,20 etc.
	// 
	// Here we re-order the h_index, but do not re-order h_angles. Reason: h_index is used to fetch h_angles. e.g. if h_index is 0,1,2,3,4, and h_angles is -0.3,-0.15,0,0.15,0.3, then when later 
	// fwd prj or bkprj happens, the angles in order will be: -0.3,-0.15,0,0.15,0.3; In another case, if h_index is 0,2,1,4,3, h_angles is still -0.3,-0.15,0,0.15,0.3, but then when later 
	// fwd prj or bkprj happens, the angles in order will be: -0.3,0,-0.15,0.3,0.15

	host_allangle_tmp = (float *)calloc( all_b_size , sizeof(float) );	if (host_allangle_tmp == NULL){printf("memory problem of host_allangle_tmp !!\n");system("PAUSE");exit(1);}
	memset(host_allangle_tmp,0,sizeof(float)*all_b_size);

	regroup_prj(host_prj_allangle,host_allangle_tmp);
	memset(host_allangle_tmp,0,sizeof(float)*all_b_size);

	regroup_prj(host_scat_allangle,host_allangle_tmp);
	memset(host_allangle_tmp,0,sizeof(float)*all_b_size);

	regroup_prj(host_blank_allangle,host_allangle_tmp);
	memset(host_allangle_tmp,0,sizeof(float)*all_b_size);

	free((void *)host_allangle_tmp); // we won't need host_allangle_tmp anymore

	int *index_tmp;
	index_tmp = (int *)calloc( h_ANGLES , sizeof(int) );
	memset(index_tmp,0,sizeof(int)*h_ANGLES);
	for (i=0;i<h_ANGLES;++i){index_tmp[i]=h_index[i];}
	int flag = 0;
	for(i=0;i<h_subset_num;++i){
		for(j=0;j<ANGLES_per_sub;++j){
			h_index[flag]=index_tmp[j*h_subset_num+i];
			++flag;
		}
	}

	free((void *) index_tmp);
	cudaMemcpy(d_index, h_index, h_ANGLES*sizeof(int), cudaMemcpyHostToDevice);

	//CUDA_SAFE_CALL(cudaMemcpy(d_index, h_index, h_ANGLES*sizeof(int), cudaMemcpyHostToDevice) ); // update d_index with new order scheme
	///////////////////END of Important modification here///////////////////////////////////////////



	host_line_sub = (float *)calloc( sub_b_size , sizeof(float) );	if (host_line_sub == NULL){printf("memory problem of host_line_sub !!\n");system("PAUSE");exit(1);}
	host_prj_sub = (float *)calloc( sub_b_size , sizeof(float) );	if (host_prj_sub == NULL){printf("memory problem of host_prj_sub !!\n");system("PAUSE");exit(1);}
	host_scat_sub = (float *)calloc( sub_b_size , sizeof(float) );	if (host_scat_sub == NULL){printf("memory problem of host_scat_sub !!\n");system("PAUSE");exit(1);}
	host_blank_sub = (float *)calloc( sub_b_size , sizeof(float) );	if (host_blank_sub == NULL){printf("memory problem of host__sub !!\n");system("PAUSE");exit(1);}
	memset(host_line_sub,0,sizeof(float)*sub_b_size);
	memset(host_prj_sub,0,sizeof(float)*sub_b_size);
	memset(host_scat_sub,0,sizeof(float)*sub_b_size);
	memset(host_blank_sub,0,sizeof(float)*sub_b_size);


	// allocate CPU memo for host_h_sub;
	host_h_sub = (float *)calloc( sub_b_size , sizeof(float) );	if (host_h_sub == NULL){printf("memory problem of host_h_sub !!\n");system("PAUSE");exit(1);}
	memset(host_h_sub,0,sizeof(float)*sub_b_size);

	// allocate CPU memo for host_capL, host_RDD, host_RD;
	host_capL = (float *)calloc( f_size , sizeof(float) );	if (host_capL == NULL){printf("memory problem of host_capL !!\n");system("PAUSE");exit(1);}
	memset(host_capL,0,sizeof(float)*f_size);

	host_RDD = (float *)calloc( f_size,sizeof(float) );if (host_RDD == NULL){printf("memory problem of host_RDD !!\n");system("PAUSE");exit(1);}
	memset(host_RDD,0,sizeof(float)*f_size);//[test]

	host_RD = (float *)calloc( f_size,sizeof(float) );if (host_RD == NULL){printf("memory problem of host_RD !!\n");system("PAUSE");exit(1);}
	memset(host_RD,0,sizeof(float)*f_size);//[test]



	cudaMalloc((void**)&d_line_sub, sub_b_size*sizeof(float));
	cudaMemset(d_line_sub, 0, sub_b_size*sizeof(float));
		// Note: we allocate d_line_sub here for GPU memory budget reason (after freeing other GPU space which won't be used from now)
		//CUDA_SAFE_CALL(cudaMalloc((void**)&d_line_sub, sub_b_size*sizeof(float)));if (d_line_sub == NULL){printf("memory problem of d_line_sub !!\n");system("PAUSE");exit(1);} 
		//CUDA_SAFE_CALL(cudaMemset(d_line_sub, 0, sub_b_size*sizeof(float)));


	cudaMalloc((void**)&d_h_sub, sub_b_size*sizeof(float));
	cudaMemset(d_h_sub, 0, sub_b_size*sizeof(float));
		// allocate GPU memo for d_h_sub;
		//CUDA_SAFE_CALL(cudaMalloc((void**)&d_h_sub, sub_b_size*sizeof(float)));if (d_h_sub == NULL){printf("memory problem of d_h_sub !!\n");system("PAUSE");exit(1);}
		//CUDA_SAFE_CALL(cudaMemset(d_h_sub, 0, sub_b_size*sizeof(float)));

	cudaMalloc((void**)&d_capL, f_size*sizeof(float));
	cudaMemset(d_capL, 0, f_size*sizeof(float));
		// allocate GPU memo for d_capL;
		//CUDA_SAFE_CALL(cudaMalloc((void**)&d_capL, f_size*sizeof(float)));if (d_capL == NULL){printf("memory problem of d_capL !!\n");system("PAUSE");exit(1);}
		//CUDA_SAFE_CALL(cudaMemset(d_capL, 0, f_size*sizeof(float)));

	cudaMalloc((void**)&d_RDD, f_size*sizeof(float));
	cudaMemset(d_RDD, 0, f_size*sizeof(float));
		// allocate GPU memo for d_RDD;
		//CUDA_SAFE_CALL(cudaMalloc((void**)&d_RDD, f_size*sizeof(float)));if (d_RDD == NULL){printf("memory problem of d_RDD !!\n");system("PAUSE");exit(1);}
		//CUDA_SAFE_CALL(cudaMemset(d_RDD, 0, f_size*sizeof(float)));

	cudaMalloc((void**)&d_RD, f_size*sizeof(float));
	cudaMemset(d_RD, 0, f_size*sizeof(float));
		// allocate GPU memo for d_capL;
		//CUDA_SAFE_CALL(cudaMalloc((void**)&d_RD, f_size*sizeof(float)));if (d_RD == NULL){printf("memory problem of d_RD !!\n");system("PAUSE");exit(1);}
		//CUDA_SAFE_CALL(cudaMemset(d_RD, 0, f_size*sizeof(float)));






	printf("Before iteration loop \n");
	//////////////////// start iteration loop///////////////////
	for (iter=0;iter<h_iter_num;++iter){
		for (a=0; a<h_subset_num; ++a){
			angleStart=a*ANGLES_per_sub;
			angleEnd=(a+1)*ANGLES_per_sub;

			// fwdprj d_est to get d_line_sub
			fprojectCB_1R_GPU_OSTR_cos(
				d_est,
				d_line_sub,
				d_angles,
				d_index,
				angleStart,
				angleEnd);
			//system("PAUSE");

			cudaMemcpy(host_line_sub,d_line_sub,sub_b_size*sizeof(float),cudaMemcpyDeviceToHost);
			//CUDA_SAFE_CALL(cudaMemcpy(host_line_sub,d_line_sub,sub_b_size*sizeof(float),cudaMemcpyDeviceToHost));


			// Now copy subset data to host_scat_sub/host_blank_sub/host_prj_sub/ create h term 20:22 April 2014 
			for(i=0;i<sub_b_size;i++){
				host_prj_sub[i]=host_prj_allangle[angleStart*b_size+i];
			}
			for(i=0;i<sub_b_size;i++){
				host_scat_sub[i]=host_scat_allangle[angleStart*b_size+i];
			}
			for(i=0;i<sub_b_size;i++){
				host_blank_sub[i]=host_blank_allangle[angleStart*b_size+i];
			}


			// here we compute 'h' term on CPU, if GPU memory allows, we can further accelerate this using GPU 
			compute_h(host_h_sub,host_prj_sub,host_blank_sub,host_line_sub,host_scat_sub);
			//CUDA_SAFE_CALL(cudaMemcpy(d_h_sub,host_h_sub,sub_b_size*sizeof(float),cudaMemcpyHostToDevice));
			cudaMemcpy(d_h_sub,host_h_sub,sub_b_size*sizeof(float),cudaMemcpyHostToDevice);

			// now we bkprj 'h' to get 'capL' term, Note: we calculate d_capL without multiplying the subset number, we will do the multiplication for host_capL;
			bprojectCB_GPU_SBP(
				(float *)d_capL, 
				(float *)d_h_sub, 
				d_index, 
				d_angles, 
				angleStart,
				angleEnd); 
			//CUDA_SAFE_CALL(cudaMemcpy(host_capL,d_capL,f_size*sizeof(float),cudaMemcpyDeviceToHost));
			cudaMemcpy(host_capL,d_capL,f_size*sizeof(float),cudaMemcpyDeviceToHost);
			for(i=0;i<f_size;i++){
				host_capL[i]=h_subset_num*host_capL[i];
			}



			prior_GPU_OSTR_Q2(
				d_RDD,
				d_RD,
				d_est,
				z_xy_ratio);

			cudaMemcpy(host_RDD, d_RDD, f_size*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(host_RD,  d_RD,  f_size*sizeof(float), cudaMemcpyDeviceToHost);

				
			

			//CUDA_SAFE_CALL(cudaMemcpy(host_RDD,d_RDD,f_size*sizeof(float),cudaMemcpyDeviceToHost));
			//CUDA_SAFE_CALL(cudaMemcpy(host_RD,d_RD,f_size*sizeof(float),cudaMemcpyDeviceToHost));


			update_est(	
				host_est,
				host_capL,
				host_RD,
				host_d,
				host_RDD);

			cudaMemcpy(d_est,host_est,f_size*sizeof(float),cudaMemcpyHostToDevice);
			//CUDA_SAFE_CALL(cudaMemcpy(d_est,host_est,f_size*sizeof(float),cudaMemcpyHostToDevice));


			cudaMemset(d_line_sub, 0, sub_b_size*sizeof(float));
			cudaMemset(d_capL, 0, f_size*sizeof(float));
			cudaMemset(d_RDD, 0, f_size*sizeof(float));
			cudaMemset(d_RD, 0, f_size*sizeof(float));

//			CUDA_SAFE_CALL(cudaMemset(d_line_sub, 0, sub_b_size*sizeof(float)));
//			CUDA_SAFE_CALL(cudaMemset(d_capL, 0, f_size*sizeof(float)));
//			CUDA_SAFE_CALL(cudaMemset(d_RDD, 0, f_size*sizeof(float)));
//			CUDA_SAFE_CALL(cudaMemset(d_RD, 0, f_size*sizeof(float)));
			


			memset(host_line_sub,0,sizeof(float)*sub_b_size);
			memset(host_h_sub,0,sizeof(float)*sub_b_size);
			memset(host_prj_sub,0,sizeof(float)*sub_b_size);
			memset(host_scat_sub,0,sizeof(float)*sub_b_size);
			memset(host_blank_sub,0,sizeof(float)*sub_b_size);

			memset(host_capL,0,sizeof(float)*f_size);
			memset(host_RDD,0,sizeof(float)*f_size);
			memset(host_RD,0,sizeof(float)*f_size);

			printf("Finished: %d sub-iter, %d iter \n",a+1,iter+1); 
			//system("PAUSE");

		}
		int saveid=iter;
		sprintf(buffer,"%d", saveid);

		//itoa(saveid,buffer,10);
		strcpy(estid,h_recon_folder);
		strcat(estid,h_recon_name);
		strcat(estid,buffer);

		if ((iter+1)%h_IO_Iter==0){
			save_data(
				estid,
				host_est,
				'f',
				h_IMGSIZz,
				h_IMGSIZx*h_IMGSIZy);
		}

	}
	/////////////////////////////////END of Iteration loop///////////////////////////////////////


	printf("After iteration loop; Before free memory \n");
	//system("PAUSE");

cudaFree(d_capL);
cudaFree(d_RDD);
cudaFree(d_RD);
cudaFree(d_index);
cudaFree(d_angles);
cudaFree(d_est);
cudaFree(d_line_sub);

	//CUDA_SAFE_CALL(cudaFree(d_capL));//[test]
	//CUDA_SAFE_CALL(cudaFree(d_RDD));//[test]
//	CUDA_SAFE_CALL(cudaFree(d_RD));//[test]
//	CUDA_SAFE_CALL(cudaFree(d_index));
//	CUDA_SAFE_CALL(cudaFree(d_angles));
//	CUDA_SAFE_CALL(cudaFree(d_est));
	//CUDA_SAFE_CALL(cudaFree(d_d));
	//CUDA_SAFE_CALL(cudaFree(d_line_sub));
	//CUDA_SAFE_CALL(cudaFree(d_gamma_yry_allangle));
	//CUDA_SAFE_CALL(cudaFree(d_gamma_allangle));

	free((void *)host_prj_allangle);
	free((void *)host_scat_allangle);
	//free((void *)host_yry_allangle);
	//free((void *)host_gamma_allangle);
	//free((void *)host_gamma_yry_allangle);
	free((void *)host_d);
	free((void *)host_est);
	free((void *)host_line_sub);
	free((void *)host_prj_sub);
	free((void *)host_scat_sub);
	free((void *)host_blank_sub);
	free((void *)host_h_sub);
	free((void *)host_capL);
	free((void *)host_RDD);
	free((void *)host_RD);
	//free((void *)index_tmp);
	printf("After free memory \n");
	//system("PAUSE");

}



//Import parameter from file:
void	import_param()
{
  //float	angles_step;
  int     i, status;

  status=load_data(h_angle_filename,h_angles,'f',1,h_ANGLES);
  //for(i=0;i<h_ANGLES;i++) h_angles[i]=h_angles[i]*180/M_PI;

  if(!status){
	  printf("error in reading h_angles !\n"); exit(1);
  }

		for (i=0;i<h_ANGLES;i++)	h_index[i] = i;
		 printf("Complete reading parameters !\n");
}



//Caller to CPU code to perform backprojection.
void Reconstruction()
{
	//system("PAUSE");
	
	printf("METHODS	: %d (0:OSTR [Q2 Prior]. 1:OSTR [Lange Prior], 2:) \n", h_method);
	printf("Prj Dim %d %d \n",h_BINSx,h_BINSy);
	printf("Recon Dim %d %d %d\n",h_IMGSIZx,h_IMGSIZy,h_IMGSIZz);
	printf("Projection Data : %s%s\n", h_prj_folder,h_prj_name);
	printf("scatter Data : %s%s\n", h_scat_folder,h_scat_name);
	printf("Reconstructed image	: %s,%s\n",h_recon_folder,h_recon_name);
	printf("Iteration Number: %d \n", h_iter_num);
	printf("Export Recon Files every X Iter: %d \n", h_IO_Iter);
	printf("Beta Value: %f \n", h_beta);
	printf("Delta Value: %f \n", h_delta);
	printf("Initial Guess File	: %s \n", h_ini_name);


	import_param();


	switch (h_method)
	{
		case 0:	//Shift-and-add
			ConebeamCT_OSTR_Q2Prior_GPU ();
			break;
		case 1:	//SART
			ConebeamCT_OSTR_LangePrior_GPU ();
			break;
		case 2:	//Convex OSEM			
			break;
		case 3: //OS-SART
			break;								
		default: break;
	}

}
