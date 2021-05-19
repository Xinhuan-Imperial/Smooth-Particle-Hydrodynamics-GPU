/**
 * This software is Smooth Particle Hydrodynamics Modelling for dam breaking
 * based on GPU, writen by Xinhuan Zhou from Ultrasound Lab for Imaging and Sensing
 * http://www.bg.ic.ac.uk/research/m.tang/ulis/
 * Wall: dynamic boundary conditions
 * StepAlgorithm="Verlet"; VerletSteps=40; Kernel="Wendland"; Viscosity="Artificial"; Visco=0.100000; CaseNp=171496
 * CaseNbound=43186; CaseNfluid=128310; Dx=0.0085; H=0.014722; CoefficientH=1; CteB=162005.140625; Gamma=7.000000
 * RhopZero=1000.000000; Cs0=33.6755; CFLnumber=0.200000; DtIni=0.000437186; DtMin=2.18593e-05; MassFluid=0.000614
 * Awen (Wendland)=130920.898438; Bwen (Wendland)=-44463280.000000; TimeMax=1.6; TimePart=0.01; Gravity=(0.000000,0.000000,-9.810000)
 * RhopOutMin=700.000000;RhopOutMax=1300.000000
 * Author: Xinhuan Zhou, PhD of Imperial College London
 * Date: 15 October 2019
 * 
**/

#include <iostream>
#include <string>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

using namespace std;
#define BLOCK_X 128

string output_direc = "./dam";
string filename = "dam-breaking";
string logname;

float TIME_ALL=1.6f;
float TIME_SAVE=0.01f;
int VerletSteps=40;
float DISTX = 1.599f;
float DISTY = 0.6725f;
float DISTZ = 0.4515f;

int NUM_PARTICLES=171496;
int NUM_FLUIDS=128310;
int NUM_WALLS=43186;
int bx=1340;
float dx=0.0085f;
float RHO0=1000.f;
int NUM_THREADS=BLOCK_X*bx;
float H0=0.014722f;
float CX=floor(DISTX/2/H0)+1.f;
float CY=floor(DISTY/2/H0)+1.f;
float CZ=floor(DISTZ/2/H0)+1.f;
float CFL=0.2f;
int NUM_CELLS=(int)CX*CY*CZ;
float DtMin=2.18593e-05f;

// CUDA global constants
__constant__ int d_NUM_THREADS=171520;
__constant__ float d_H0=0.014722f;
__constant__ float d_CX=55;
__constant__ float d_CY=23;
__constant__ float d_CZ=16;
__constant__ float d_p0=162005.140625f;
__constant__ float d_RHO0=1000.f;
//__constant__ float d_coef=130932.96395f;
__constant__ float d_coef=130920.898438f;
__constant__ float d_coef1=-44463280.f;
__constant__ int d_bx=1340;
__constant__ float d_MASS=0.000614f;
__constant__ float d_CS=33.6755f;
__constant__ float d_CFL=0.2f;
__constant__ float d_RhopOutMin=700.f;
__constant__ float d_RhopOutMax=1300.f;
__constant__ float d_DtMin=2.18593e-05f;
__constant__ float flt_max=3.40282e+38f;
__constant__ int int_max=2147483647;
__constant__ float d_DISTX = 1.599f;
__constant__ float d_DISTY = 0.6725f;
__constant__ float d_DISTZ = 0.4515f;


float *DT,*h_pos,*h_veld,*h_f,*d_pos,*d_veld,*d_deltarho,*d_veld_pre,*d_f,*d_time,*h_T,*d_tmp;
int ind,*d_particle,*d_Pkey,*d_num,*d_start;
bool *h_keep,*d_keep;

void InitSPH(){
	string value;
	ifstream ifs("./input.csv",ifstream::in);
	if(ifs.is_open()){
		getline(ifs,value,'\n');
		for(int ind=0;ind<NUM_PARTICLES;ind++){
		  getline(ifs,value,',');
		  getline(ifs,value,',');
		  h_veld[ind]=stod(value);
		  getline(ifs,value,',');
		  h_veld[NUM_THREADS+ind]=stod(value);
		  getline(ifs,value,',');
		  h_veld[NUM_THREADS*2+ind]=stod(value);
		  getline(ifs,value,',');
		  h_veld[NUM_THREADS*3+ind]=stod(value);
		  getline(ifs,value,',');
		  getline(ifs,value,',');
		  getline(ifs,value,',');
		  h_pos[ind]=stod(value);
		  getline(ifs,value,',');
		  h_pos[NUM_THREADS+ind]=stod(value);
		  getline(ifs,value,'\n');
		  h_pos[NUM_THREADS*2+ind]=stod(value);
		  h_keep[ind]=true;
		}
    ifs.close();
	}
	cout<<"Initialization completed"<<endl;
}

__global__ void Density_pre(int NUM_WALLS,int NUM_PARTICLES,float*  __restrict__ d_pos,float*  __restrict__ d_veld,int*  __restrict__ d_start,int*  __restrict__ d_num,int*  __restrict__ d_Pkey){
	int ind,num,pind,start;
	float rho,rx,ry,rz,r1,r2,r3,rhoi,rhoj,r_square,r_norm,r1_norm,r2_norm,r3_norm,q,tmp,ux,uy,uz,u1,u2,u3;
	ind=threadIdx.x+blockIdx.x*BLOCK_X;
	float drho=0.f;
	
	if(ind<NUM_PARTICLES){
		rx=d_pos[ind];//pos,force
		ry=d_pos[d_NUM_THREADS+ind];
		rz=d_pos[d_NUM_THREADS*2+ind];
		ux=d_veld[ind];
		uy=d_veld[d_NUM_THREADS+ind];
		uz=d_veld[d_NUM_THREADS*2+ind];
		rhoi=d_veld[d_NUM_THREADS*3+ind];
		for(int cellz=(int)max(0.0,floor(rz/(2.0*d_H0))-1.0);cellz<=min(floor(rz/(2.0*d_H0))+1.0,d_CZ-1.0);cellz++){
			for(int celly=(int)max(0.0,floor(ry/(2.0*d_H0))-1.0);celly<=min(floor(ry/(2.0*d_H0))+1.0,d_CY-1.0);celly++){
				for(int cellx=(int)max(0.0,floor(rx/(2.0*d_H0))-1.0);cellx<=min(floor(rx/(2.0*d_H0))+1.0,d_CX-1.0);cellx++){
					pind=cellx+celly*d_CX+cellz*d_CX*d_CY;
					num=d_num[pind];
					if(num==0)continue;
					start=d_start[pind];
					
					for(int cind=start;cind<start+num;cind++){
						pind=d_Pkey[cind];//d_veld-vel,density,reuse pind
						r1=rx-d_pos[pind];//pos,force
						r2=ry-d_pos[d_NUM_THREADS+pind];
						r3=rz-d_pos[d_NUM_THREADS*2+pind];

						r_square=r1*r1+r2*r2+r3*r3;
						r_norm=sqrt(r_square);
						r1_norm=r1/r_norm;
						r2_norm=r2/r_norm;
						r3_norm=r3/r_norm;
						q=r_norm/d_H0;
						if(q>=2||q==0)continue;
						tmp=q*pow(1.0-q/2.0,3.0);

						u1=ux-d_veld[pind];
						u2=uy-d_veld[d_NUM_THREADS+pind];
						u3=uz-d_veld[d_NUM_THREADS*2+pind];
						rhoj=d_veld[d_NUM_THREADS*3+pind];
						pind>=NUM_WALLS?drho+=tmp*(r1_norm*u1+r2_norm*u2+r3_norm*u3)+0.2*tmp*d_CS*d_H0/rhoj/r_norm*(rhoi-rhoj):drho+=tmp*(r1_norm*u1+r2_norm*u2+r3_norm*u3);
					}
				}
			}
		}
		rho=rhoi+0.000437186*d_coef1*d_MASS*drho;
		if(ind<NUM_WALLS)rho=max(rho,1000.f);
		d_veld[d_NUM_THREADS*3+ind]=rho;
	}
}

__global__ void ComputeCells(int NUM_PARTICLES,float*  __restrict__ d_pos,int* __restrict__ d_particle,int* __restrict__ d_Pkey,int* __restrict__ d_num,bool* __restrict__ d_keep){
	int ind,cellidx;
	float rx,ry,rz;
	ind=threadIdx.x+blockIdx.x*BLOCK_X;
	
	if(ind<NUM_PARTICLES){
		rx=d_pos[ind];
		ry=d_pos[d_NUM_THREADS+ind];
		rz=d_pos[d_NUM_THREADS*2+ind];
		cellidx=floor(rx/(2.0f*d_H0))+floor(ry/(2.0f*d_H0))*d_CX+floor(rz/(2.0f*d_H0))*d_CY*d_CX;
		d_keep[ind]==true?d_particle[ind]=cellidx:d_particle[ind]=int_max;
		d_Pkey[ind]=ind;
		__syncthreads();
		atomicAdd(d_num+cellidx,1);
	}
}

__global__ void ParticleInteract(int NUM_WALLS,int NUM_PARTICLES,float*  __restrict__ d_pos,float*  __restrict__ d_veld,float*  __restrict__ d_f,int*  __restrict__ d_start,int*  __restrict__ d_num,float*  __restrict__ d_time,int*  __restrict__ d_Pkey,float*  __restrict__ d_deltarho,bool*  __restrict__ d_keep){
	//1.compute time step; 2.calculate density change; 3.calculate force
	int start,num,pind,ind;
	float tmp1,r1,r2,r3,u1,u2,u3,tmp,q,pi,pj,rhoi,rhoj,r_square,r_norm,rx,ry,rz,ux,uy,uz,mu,ftmp,r1_norm,r2_norm,r3_norm;
	ind=threadIdx.x+blockIdx.x*BLOCK_X;
	float drho=0.0f;
	float fx=0.0f;
	float fy=0.0f;
	float fz=0.0f;	
	float tmp2=0.0f;
	
	if(ind<NUM_PARTICLES&&d_keep[ind]==true){
		rx=d_pos[ind];//pos,force
		ry=d_pos[d_NUM_THREADS+ind];
		rz=d_pos[d_NUM_THREADS*2+ind];
		ux=d_veld[ind];
		uy=d_veld[d_NUM_THREADS+ind];
		uz=d_veld[d_NUM_THREADS*2+ind];
		rhoi=d_veld[d_NUM_THREADS*3+ind];
		pi=d_p0*powf(rhoi/d_RHO0,7.0f)-d_p0;
		
		for(int cellz=(int)max(0.0f,floor(rz/(2.0f*d_H0))-1.0f);cellz<=min(floor(rz/(2.0f*d_H0))+1.0f,d_CZ-1.0f);cellz++){
			for(int celly=(int)max(0.0f,floor(ry/(2.0f*d_H0))-1.0f);celly<=min(floor(ry/(2.0f*d_H0))+1.0f,d_CY-1.0f);celly++){
				for(int cellx=(int)max(0.0f,floor(rx/(2.0f*d_H0))-1.0f);cellx<=min(floor(rx/(2.0f*d_H0))+1.0f,d_CX-1.0f);cellx++){
					pind=cellx+celly*d_CX+cellz*d_CX*d_CY;
					num=d_num[pind];
					if(num==0)continue;
					start=d_start[pind];
					
					for(int cind=start;cind<start+num;cind++){
						pind=d_Pkey[cind];//d_veld-vel,density,reuse pind
						if(d_keep[pind]==false)continue;
						r1=rx-d_pos[pind];//pos,force
						r2=ry-d_pos[d_NUM_THREADS+pind];
						r3=rz-d_pos[d_NUM_THREADS*2+pind];

						r_square=r1*r1+r2*r2+r3*r3;
						r_norm=sqrt(r_square);
						r1_norm=r1/r_norm;
						r2_norm=r2/r_norm;
						r3_norm=r3/r_norm;
						q=r_norm/d_H0;

						if(q>=2||q==0)continue;
						tmp=q*powf(1.0f-q/2.0f,3.0f);

						u1=ux-d_veld[pind];
						u2=uy-d_veld[d_NUM_THREADS+pind];
						u3=uz-d_veld[d_NUM_THREADS*2+pind];
						rhoj=d_veld[d_NUM_THREADS*3+pind];
						pj=d_p0*powf(rhoj/d_RHO0,7.0f)-d_p0;
						
						mu=d_H0*(u1*r1+u2*r2+u3*r3)/(r_square+0.01f*d_H0*d_H0);
						if(pind>=NUM_WALLS){
							tmp2=max(tmp2,fabs(mu));
						}				
						
						drho+=tmp*(r1_norm*u1+r2_norm*u2+r3_norm*u3);
						//(ind>=NUM_WALLS&&pind<NUM_WALLS)?drho+=tmp*(r1_norm*u1+r2_norm*u2+r3_norm*u3):drho+=tmp*(r1_norm*u1+r2_norm*u2+r3_norm*u3)+0.1f*tmp*d_CS*d_H0/rhoj/r_norm*(rhoi-rhoj);
						//drho+=tmp*(r1_norm*u1+r2_norm*u2+r3_norm*u3)+0.1f*tmp*d_CS*d_H0/rhoj/r_norm*(rhoi-rhoj);
						
						if(mu>=0)tmp1=0.0f;
						else{
							//tmp1=-0.1f*(cs1+cs2)*mu/(rhoi+rhoj);
							tmp1=-0.2f*d_CS*mu/(rhoi+rhoj);
						}
						ftmp=(pi+pj)/rhoi/rhoj+tmp1;
						fx=fx-tmp*r1_norm*ftmp;
						fy=fy-tmp*r2_norm*ftmp;
						fz=fz-tmp*r3_norm*ftmp;
					}
				}
			}
		}

		fz=fz-9.81f/(d_MASS*d_coef1);
		tmp1=d_H0/sqrt(fx*fx+fy*fy+fz*fz)/(d_MASS*d_coef1);
		tmp1=sqrt(tmp1);
		tmp2=d_H0/(d_CS+tmp2);
		if(ind>=NUM_WALLS)d_time[ind]=min(tmp1,tmp2);
		d_f[ind]=fx;
		d_f[d_NUM_THREADS+ind]=fy;
		d_f[d_NUM_THREADS*2+ind]=fz;
		d_deltarho[ind]=drho;
	}
}

__global__ void SysUpdate(int NUM_WALLS,int NUM_PARTICLES,float* DT,float*  __restrict__ d_deltarho,float*  __restrict__ d_f,float*  __restrict__ d_pos,float*  __restrict__ d_veld,float*  __restrict__ d_veld_pre,float*  __restrict__ d_time,bool*  __restrict__ d_keep){
	int ind=threadIdx.x+blockIdx.x*BLOCK_X;
	float rho,rx,ry,rz,fx,fy,fz,drho,ux,uy,uz,ux_pre,uy_pre,uz_pre,rho_pre;
	float T=*DT;
	T=max(T*d_CFL,d_DtMin);
	
	if(ind<NUM_PARTICLES&&d_keep[ind]==true){
		rx=d_pos[ind];//pos,force
		ry=d_pos[d_NUM_THREADS+ind];
		rz=d_pos[d_NUM_THREADS*2+ind];
		fx=d_f[ind];
		fy=d_f[d_NUM_THREADS+ind];
		fz=d_f[d_NUM_THREADS*2+ind];	
		drho=d_deltarho[ind];
		
		ux=d_veld[ind];//vel,density//vel,density
		uy=d_veld[d_NUM_THREADS+ind];
		uz=d_veld[d_NUM_THREADS*2+ind];

		ux_pre=d_veld_pre[ind];//vel,density//vel,density
		uy_pre=d_veld_pre[d_NUM_THREADS+ind];
		uz_pre=d_veld_pre[d_NUM_THREADS*2+ind];
		rho_pre=d_veld_pre[d_NUM_THREADS*3+ind];

		rx=rx+T*ux+0.5f*T*d_MASS*d_coef1*fx*T;
		ry=ry+T*uy+0.5f*T*d_MASS*d_coef1*fy*T;
		rz=rz+T*uz+0.5f*T*d_MASS*d_coef1*fz*T;
		ux=ux_pre+2.0f*T*d_MASS*d_coef1*fx;
		uy=uy_pre+2.0f*T*d_MASS*d_coef1*fy;
		uz=uz_pre+2.0f*T*d_MASS*d_coef1*fz;
		rho=rho_pre+2.0f*T*d_coef1*d_MASS*drho;
		if(ind<NUM_WALLS)rho=max(rho,1000.f);
		d_veld_pre[d_NUM_THREADS*3+ind]=rho;
		if(ind>=NUM_WALLS){
			if(rho<=d_RhopOutMin||rho>=d_RhopOutMax||rx>d_DISTX||rx<0||ry>d_DISTY||ry<0||rz>d_DISTZ||rz<0){
				d_time[ind]=flt_max;
				d_keep[ind]=false;
			}
			
			d_veld_pre[ind]=ux;
			d_veld_pre[d_NUM_THREADS+ind]=uy;
			d_veld_pre[d_NUM_THREADS*2+ind]=uz;
			d_pos[ind]=rx;
			d_pos[d_NUM_THREADS+ind]=ry;
			d_pos[d_NUM_THREADS*2+ind]=rz;	
		}
	}
}
		
__global__ void EulerUpdate(int NUM_WALLS,int NUM_PARTICLES,float* DT,float*  __restrict__ d_deltarho,float*  __restrict__ d_f,float*  __restrict__ d_pos,float*  __restrict__ d_veld,float*  __restrict__ d_veld_pre,float*  __restrict__ d_time,bool*  __restrict__ d_keep){
	int ind=threadIdx.x+blockIdx.x*BLOCK_X;
	float rho,rx,ry,rz,fx,fy,fz,drho,ux,uy,uz;
	float T=*DT;
	T=max(T*d_CFL,d_DtMin);
	
	if(ind<NUM_PARTICLES&&d_keep[ind]==true){
		rx=d_pos[ind];//pos,force
		ry=d_pos[d_NUM_THREADS+ind];
		rz=d_pos[d_NUM_THREADS*2+ind];
		fx=d_f[ind];
		fy=d_f[d_NUM_THREADS+ind];
		fz=d_f[d_NUM_THREADS*2+ind];	
		drho=d_deltarho[ind];
		
		ux=d_veld[ind];//vel,density//vel,density
		uy=d_veld[d_NUM_THREADS+ind];
		uz=d_veld[d_NUM_THREADS*2+ind];
		rho=d_veld[d_NUM_THREADS*3+ind];		
	
		rx=rx+T*ux+0.5f*T*d_MASS*d_coef1*fx*T;
		ry=ry+T*uy+0.5f*T*d_MASS*d_coef1*fy*T;
		rz=rz+T*uz+0.5f*T*d_MASS*d_coef1*fz*T;
		ux=ux+T*d_MASS*d_coef1*fx;
		uy=uy+T*d_MASS*d_coef1*fy;
		uz=uz+T*d_MASS*d_coef1*fz;
		rho=rho+T*d_coef1*d_MASS*drho;
		if(ind<NUM_WALLS)rho=max(rho,1000.f);
		d_veld_pre[d_NUM_THREADS*3+ind]=rho;		
		
		if(ind>=NUM_WALLS){
			if(rho<=d_RhopOutMin||rho>=d_RhopOutMax||rx>d_DISTX||rx<0||ry>d_DISTY||ry<0||rz>d_DISTZ||rz<0){
				d_time[ind]=flt_max;
				d_keep[ind]=false;
			}
			d_veld_pre[ind]=ux;//vel,density. We need to swap the two pointers:d_veld and d_veld_pre
			d_veld_pre[d_NUM_THREADS+ind]=uy;
			d_veld_pre[d_NUM_THREADS*2+ind]=uz;
			d_pos[ind]=rx;
			d_pos[d_NUM_THREADS+ind]=ry;
			d_pos[d_NUM_THREADS*2+ind]=rz;	
		}
	}
}
	
void outputSave(int t) {
	string datafilename = output_direc +"/dam_" + to_string(t) + ".csv";;
	ofstream ofs(datafilename);
	ofs<<"x,y,z,vel,ux,uy,uz,rho"<<endl;
					
	for(int ind=0;ind<NUM_PARTICLES;ind++){//outside:-1,wall:1,fluid:0
		ofs<<h_pos[ind]<<','<<h_pos[NUM_THREADS+ind]<<','<<h_pos[NUM_THREADS*2+ind]<<','<<sqrt(pow(h_veld[ind],2.f)+pow(h_veld[NUM_THREADS+ind],2.f)+pow(h_veld[NUM_THREADS*2+ind],2.f))<<','<<h_veld[ind]<<','<<h_veld[NUM_THREADS+ind]<<','<<h_veld[NUM_THREADS*2+ind]<<','<<h_veld[NUM_THREADS*3+ind]<<endl;
	}
	ofs.close();
}


int main(int argc,const char **argv) {
	float milli;
	float time_bef=0;
	float time_aft=0;
	int i=0;
	
	cudaEvent_t start,stop;	
	logname=output_direc+'/'+ "CONVERGENCE.log";
	ofstream logfile(logname);
	
	dim3 dimGrid(bx);
	dim3 dimBlock(BLOCK_X);
	//allocate memory
	h_pos=(float*)malloc(sizeof(float)*NUM_THREADS*3);
	h_veld=(float*)malloc(sizeof(float)*NUM_THREADS*4);
	h_f=(float*)malloc(sizeof(float)*NUM_THREADS*3);
	h_T=(float*)malloc(sizeof(float));
	h_keep=(bool*)malloc(sizeof(bool)*NUM_THREADS);
	*h_T=0.000437186f/CFL;
	
	cudaMalloc(&d_pos,NUM_THREADS*3*sizeof(float));
	cudaMalloc(&d_veld,NUM_THREADS*4*sizeof(float));
	cudaMalloc(&d_veld_pre,NUM_THREADS*4*sizeof(float));
	cudaMalloc(&d_f,NUM_THREADS*3*sizeof(float));
	cudaMalloc(&d_particle,NUM_THREADS*sizeof(int));
	cudaMalloc(&d_start,NUM_CELLS*sizeof(int));
	cudaMalloc(&d_num,NUM_CELLS*sizeof(int));
	cudaMalloc(&d_Pkey,NUM_THREADS*sizeof(int));
	cudaMalloc(&d_time,NUM_THREADS*sizeof(float));
	cudaMalloc(&d_deltarho,NUM_THREADS*sizeof(float));
	cudaMalloc(&d_keep,NUM_THREADS*sizeof(bool));
	cudaMalloc(&DT,sizeof(float));
	cudaMemset(d_time,1.6f,NUM_THREADS*sizeof(float));
	
	thrust::device_ptr <int > dev_d_particle(d_particle);
	thrust::device_ptr<int> dev_d_start(d_start);
	thrust::device_ptr<int> dev_d_num(d_num);
	thrust::device_ptr<float> dev_d_time(d_time);
	thrust::device_ptr<float> dev_DT(DT);
	thrust::device_ptr<int> dev_d_Pkey(d_Pkey);
	
	cudaDeviceSynchronize();
	InitSPH();
	outputSave(0);
	
	cudaMemcpy(d_keep,h_keep,NUM_THREADS*sizeof(bool),cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos,h_pos,NUM_THREADS*3*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_veld,h_veld,NUM_THREADS*4*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_veld_pre,h_veld,NUM_THREADS*4*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(DT,h_T,sizeof(float),cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	
	cudaEventRecord(start,0);
	

	while(time_bef<TIME_ALL){
		cudaMemset(d_num,0,NUM_CELLS*sizeof(int));
		cudaDeviceSynchronize();

		ComputeCells<<<dimGrid,dimBlock>>>(NUM_PARTICLES,d_pos,d_particle,d_Pkey,d_num,d_keep);
		cudaDeviceSynchronize();
		
		thrust::sort_by_key(thrust::device,dev_d_particle,dev_d_particle+NUM_PARTICLES,dev_d_Pkey);
		cudaDeviceSynchronize();
		
		thrust::exclusive_scan(thrust::device,dev_d_num, dev_d_num + NUM_CELLS, dev_d_start); // in-place scan
		cudaDeviceSynchronize();
		
		ParticleInteract<<<dimGrid,dimBlock>>>(NUM_WALLS,NUM_PARTICLES,d_pos,d_veld,d_f,d_start,d_num,d_time,d_Pkey,d_deltarho,d_keep);
		cudaDeviceSynchronize();
		if(i!=0){
			dev_DT=thrust::min_element(thrust::device,dev_d_time+NUM_WALLS, dev_d_time+NUM_PARTICLES);
			DT = thrust::raw_pointer_cast(dev_DT);
			cudaDeviceSynchronize();
		}
		
		
		if (i%VerletSteps==0){
			EulerUpdate<<<dimGrid,dimBlock>>>(NUM_WALLS,NUM_PARTICLES,DT,d_deltarho,d_f,d_pos,d_veld,d_veld_pre,d_time,d_keep);
		}
		else{
			SysUpdate<<<dimGrid,dimBlock>>>(NUM_WALLS,NUM_PARTICLES,DT,d_deltarho,d_f,d_pos,d_veld,d_veld_pre,d_time,d_keep);
		}
		d_tmp=d_veld;
		d_veld = d_veld_pre;
		d_veld_pre=d_tmp;	
		cudaDeviceSynchronize();

		cudaMemcpy(h_T,DT,sizeof(float),cudaMemcpyDeviceToHost);
		*h_T=max(*h_T*CFL,DtMin);
		time_aft+=*h_T;

		if((int)(time_aft*1000.f)/(int)(TIME_SAVE*1000.f)>(int)(time_bef*1000.f)/(int)(TIME_SAVE*1000.f)){
			cout << "ITERATION # " << i<<" ; TIME "<<time_aft <<endl;
			logfile<< "ITERATION # " << i<<" ; TIME "<<time_aft <<endl;
			cudaMemcpy(h_pos,d_pos,NUM_THREADS*3*sizeof(float),cudaMemcpyDeviceToHost);
			cudaMemcpy(h_veld,d_veld,NUM_THREADS*4*sizeof(float),cudaMemcpyDeviceToHost);
			outputSave(i);
		}
		cudaDeviceSynchronize();
		time_bef=time_aft;
		i++;
		cout<<'#'<<i<<','<<time_aft<<endl;
	}
	free(h_pos);
	free(h_veld);
	free(h_f);
	free(h_keep);
	cudaFree(d_pos);
	cudaFree(d_veld);
	cudaFree(d_veld_pre);
	cudaFree(d_deltarho);
	cudaFree(d_f);
	cudaFree(d_time);
	cudaFree(d_particle);
	cudaFree(d_start);
	cudaFree(d_Pkey);
	cudaFree(d_num);
	cudaFree(d_keep);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli,start,stop);
	cout << "TOTAL RUNNING TIME: " << milli << " MILLI SECONDS" << endl;
	logfile<< "ITERATION # " << i<<" ; RUNNING TIME(ms) "<<milli <<endl;
	logfile.close();
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceReset();
	return 0;
}
