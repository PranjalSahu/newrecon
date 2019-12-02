


#include	<stdarg.h>
#include    <ctype.h>
#include    <stdlib.h>
#include	<string.h>
#include	<malloc.h>

#include "stdio.h"
#include "math.h"
#include "memory.h"
#include "cuda_runtime.h"
#include "cufft.h"
//#include <cutil_inline.h>
//#include "cutil.h"

#include "Tomo_OSTR.h"
//#include <direct.h>


/* #include        "INCLUDE/gui.h" */

/*******************************************************************
 01/10/2000, by Ing-Tsung Hsiao, updated by YX for own use. 

 readINTEGERpara(char *datafile, char *Label,int NumOfPara,int *para)

 To read the data parameters (filename or numbers) for the projection
 or reconstruction programs.  The definition of the labels is located
 in the file "project.h" or the data file.

 In this case, the parameter format is "integer".

 The "Label" is referred to the parameter field "Label", and 
 "NumOfPara" the number of parameter.  For example, the "image_size" 
 label is the size of the image (object) used in the projection or 
 reconstruction, its "ParaType" is equal to
 "INTEGER" (known in advance), and has "NumOfPara" = 1.  Both  "ParaType"
 and "NumOfPara" are known and defined in datafile or their corresponding
 use.   

 readSTRINGpara(char *datafile,char *Label,int NumOfPara,char **para) : 
                                                         for STRING
 readFLOATpara(char *datafile,char *Label,int NumOfPara,float *para) : 
							for FLOAT
 readCHARpara(char *datafile,char *Label,int NumOfPara,char *para) : 
							for CHAR   
 readINTEGERpara(char *datafile,char *Label,int NumOfPara,int *para) : 
							for Integer
 The parameter datafile is added by Yuxiang.    08/2000
 *******************************************************************/

int  readINTEGERpara(char *datafile, char *Label,int NumOfPara,int *para)
{
FILE	*fp;
char    tempstr[50];
unsigned char StillSearch=1;
char    temppara[20][100];
int  i; 

        fp = fopen(datafile,"r");
	if( fp == NULL ) {
		printf("### Error: %s not found!!\n",datafile);
		exit(1);
	}
      
	while(StillSearch){
	  //*************************************************
	  // Read the parameter label; check if it finds the
	  // label.  If yes, go to read the parameters after
	  // the label.  Otherwise, keep searching........
	  //************************************************* 
	fscanf(fp,"%s",tempstr);
        
        if(strcmp(tempstr,Label)==0)
         { 
	   for(i=0;i<NumOfPara;i++)
	     {
	     fscanf(fp,"%s",temppara[i]);
             
	     //*****************************************************
	     //As a check, if it hits the comments fields, something
	     //wrong, probabily less parameters in the file datafile
	     // or the first character of the para is not a digit
	     // (0,1,...9, it should be a number)
	     //*****************************************************
	     if((strcmp(temppara[i],"#")==0)||(isalpha( *(temppara[i]) )) )
              { 
		printf("Something wrong in the %s about this label \"%s\"\n",
		       datafile,Label);
                printf("Check the parameters after this label in %s!!\n\007",
		       datafile);
                exit(1);
		}
	     }
	    StillSearch=0;
	 }

	if ((feof(fp))&(StillSearch==1)) 
        {
           StillSearch=0; 
           printf("The parameter label \"%s\" wasn't found in the file \" %s \" !! \n\007",Label,datafile);
	   printf("Check the file!! \" %s \"  \n\007",datafile);
	   //********************************
           //already at the bottom of the file 
	   //but not found the para label yet
	   //********************************
	   exit(1);
        }

	}
       fclose(fp);
       for(i=0;i<NumOfPara;i++)  para[i]=atoi(temppara[i]);
       return 1;
 
}

/*******************************************************************
 readCHARpara(char *datafile,char *Label,int NumOfPara,char para) : for CHAR   
 *******************************************************************/

int  readCHARpara(char *datafile, char *Label,int NumOfPara,char *para)
{
FILE	*fp;
char    tempstr[50],temppara;
unsigned char StillSearch=1;
int  i; 

  fp = fopen(datafile,"r");
	if( fp == NULL ) {
		printf("### Error: %s not found!!\n",datafile);
		exit(1);
	}
      
	while(StillSearch){
	   fscanf(fp,"%s",tempstr);
           if(strcmp(tempstr,Label)==0)
           { 
	     	fscanf(fp,"%c",&temppara);
	   	while(isalpha(temppara)==0)
	     	    fscanf(fp,"%c",&temppara);
	 
	   	*para=temppara;  
	   	StillSearch=0;
	   }

	   if ((feof(fp))&(StillSearch==1)) 
           {
           	StillSearch=0; 
           	printf("The parameter \"%s\" not found in the file \" %s \" !! \n\007",Label,datafile);
	   	printf("Check the file!! \" %s \"  \n\007",datafile);
            	//already at the bottom of the file
	   	exit(1);
          }

       }
       fclose(fp);
       return 1;
}


/*******************************************************************
 readSTRINGpara(char *datafile,char *Label,int NumOfPara,int *para) : for STRING
 For string parameter, the number of parameter is usually "one".
 *******************************************************************/

int  readSTRINGpara(char *datafile,char *Label,int NumOfPara,char *para)
{
FILE	*fp;
char    tempstr[300];
unsigned char StillSearch=1;
int  i; 

 if (NumOfPara>1) 
   {
    printf("You are requesting for more than one string\n");
    printf("But usually only one string is allowed.\n");
    printf("Reset it\n");
    NumOfPara=1;
   }
             
  fp = fopen(datafile,"r");
	if( fp == NULL ) {
		printf("### Error: %s not found!!\n",datafile);
		exit(1);
	}
      
	while(StillSearch){
	  //*************************************************
	  // Read the parameter label; check if it finds the
	  // label.  If yes, go to read the parameters after
	  // the label.  Otherwise, keep searching........
	  //************************************************* 
	fscanf(fp,"%s",tempstr);
        if(strcmp(tempstr,Label)==0)
         { 
	   //for(i=0;i<NumOfPara;i++)
	     {
	     fscanf(fp,"%s",para);
             
	     //*****************************************************
	     //As a check, if it hits the comments fields, something
	     //wrong, probabily less parameters in the file datafile.
	     //*****************************************************
	     /*	          
	     if(strcmp(para[i],"#")==0)
              { 
		printf("Something wrong in the %s about this label \"%s\"\n",
			datafile,Label);
                printf("Check the parameters after this label in %s !!\n\007",
			datafile);
                exit(1);
	      }
	   */
	     }
	   
	    StillSearch=0;
	 }//****while(StillSearch)

	if ((feof(fp))&&(StillSearch==1)) 
          {
           StillSearch=0; 
           printf("The parameter label \"%s\" not found in the file \" %s \" !!\n\007",Label,datafile);
	   printf("Check the file!! \" %s \"  \n\007",datafile);
	   //********************************
           //already at the bottom of the file 
	   //but not found the para label yet
	   //********************************
	   fclose(fp);
	   exit(1);
          }

	}
       fclose(fp);
       return 1;
}

/*******************************************************************
 readFLOATpara(char *datafile,char *Label,int NumOfPara,int *para) : for FLOAT
 *******************************************************************/

int  readFLOATpara(char *datafile,char *Label,int NumOfPara,float *para)
{
FILE	*fp;
char    tempstr[50];
unsigned char StillSearch=1;
char    temppara[20][100];
int  i; 

  fp = fopen(datafile,"r");
	if( fp == NULL ) {
		printf("### Error:  datafile not found!!\n");
		exit(1);
	}
      
	while(StillSearch){
	  //*************************************************
	  // Read the parameter label; check if it finds the
	  // label.  If yes, go to read the parameters after
	  // the label.  Otherwise, keep searching........
	  //************************************************* 
	fscanf(fp,"%s",tempstr);
        if(strcmp(tempstr,Label)==0)
         { 
	   for(i=0;i<NumOfPara;i++)
	     {
	     fscanf(fp,"%s",temppara[i]);
             
	     //*****************************************************
	     //As a check, if it hits the comments fields, something
	     //wrong, probabily less parameters in the file datafile
	     //*****************************************************
	     if((strcmp(temppara[i],"#")==0)||(isalpha( *(temppara[i]) )) )
              { 
		printf("Something wrong in the %s about this label \"%s\"\n",
		       datafile, Label);
                printf("Check the parameters after this label in %s!!\n\007",
			datafile);
                exit(1);
	      }
	     }
	    StillSearch=0;
	 }

	if ((feof(fp))&(StillSearch==1)) 
          {
           StillSearch=0; 
           printf("The parameter label \"%s\" not found in the file \" %s \" !!\n\007",Label,datafile);
	   printf("Check the file!! \"%s \"  \n\007",datafile);
	   //********************************
           //already at the bottom of the file 
	   //but not found the para label yet
	   //********************************
	   exit(1);
          }

	}
       fclose(fp);
       for(i=0;i<NumOfPara;i++)  para[i]=atof(temppara[i]);
       return 1;
 
}


/*******************************************************************
 readDOUBLEpara(char *datafile,char *Label,int NumOfPara,int *para) : for DOUBLE
 *******************************************************************/

int  readDOUBLEpara(char *datafile,char *Label,int NumOfPara,double *para)
{
FILE	*fp;
char    tempstr[50];
unsigned char StillSearch=1;
char    temppara[20][100];
int  i; 

  fp = fopen(datafile,"r");
	if( fp == NULL ) {
		printf("### Error:  datafile not found!!\n");
		exit(1);
	}
      
	while(StillSearch){
	  //*************************************************
	  // Read the parameter label; check if it finds the
	  // label.  If yes, go to read the parameters after
	  // the label.  Otherwise, keep searching........
	  //************************************************* 
	fscanf(fp,"%s",tempstr);
        if(strcmp(tempstr,Label)==0)
         { 
	   for(i=0;i<NumOfPara;i++)
	     {
	     fscanf(fp,"%s",temppara[i]);
             
	     //*****************************************************
	     //As a check, if it hits the comments fields, something
	     //wrong, probabily less parameters in the file datafile
	     //*****************************************************
	     if((strcmp(temppara[i],"#")==0)||(isalpha( *(temppara[i]) )) )
              { 
		printf("Something wrong in the %s about this label \"%s\"\n",
		       datafile, Label);
                printf("Check the parameters after this label in %s!!\n\007",
			datafile);
                exit(1);
	      }
	     }
	    StillSearch=0;
	 }

	if ((feof(fp))&(StillSearch==1)) 
          {
           StillSearch=0; 
           printf("The parameter label \"%s\" not found in the file \" %s \" !!\n\007",Label,datafile);
	   printf("Check the file!! \"%s \"  \n\007",datafile);
	   //********************************
           //already at the bottom of the file 
	   //but not found the para label yet
	   //********************************
	   exit(1);
          }

	}
       fclose(fp);
       for(i=0;i<NumOfPara;i++)  para[i]=(double)atof(temppara[i]);
       return 1;
 
}




/*******************************************************************
 readSTRINGpara(char *datafile,char *Label,int NumOfPara,int *para) : for STRING
 For string parameter, the number of parameter is usually "one".
 *******************************************************************/

int  readmultiSTRINGpara(char *datafile, char *Label,int NumOfPara,char **para)
{
FILE	*fp;
char    tempstr[50];
unsigned char StillSearch=1;
int  i; 

  fp = fopen(datafile,"r");
	if( fp == NULL ) {
		printf("### Error: %s not found!!\n",datafile);
		exit(1);
	}
      
	while(StillSearch){
	  //*************************************************
	  // Read the parameter label; check if it finds the
	  // label.  If yes, go to read the parameters after
	  // the label.  Otherwise, keep searching........
	  //************************************************* 
	fscanf(fp,"%s",tempstr);
        if(strcmp(tempstr,Label)==0)
        { 
	     for(i=0;i<NumOfPara;i++)
	     {
	     fscanf(fp,"%s",para[i]);
             
	     //*****************************************************
	     //As a check, if it hits the comments fields, something
	     //wrong, probabily less parameters in the file datafile
	     //*****************************************************
	    	          
	     if(strcmp(para[i],"#")==0)
              { 
		printf("Something wrong in the  %s about this label \"%s\"\n",
			datafile,Label);
                printf("Check the parameters after this label in %s !!\n\007",
			datafile);
                exit(1);
	      }
	  
	     }
	   
	    StillSearch=0;
	 } 

	if ((feof(fp))&(StillSearch==1)) 
          {
           StillSearch=0; 
           printf("The parameter label \"%s\" not found in the file \" %s \" !!\n\007",Label,datafile);
	   printf("Check the file!! \" %s \"  \n\007",datafile);
	   //********************************
           //already at the bottom of the file 
	   //but not found the para label yet
	   //********************************
	   exit(1);
          }
        }
       fclose(fp);
       return 1;
}

