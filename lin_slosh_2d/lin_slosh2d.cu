/**
 * This software is Smooth Particle Hydrodynamics Modelling for 2D linear sloshing tank
 * based on GPU, writen by Xinhuan Zhou from Ultrasound Lab for Imaging and Sensing
 * http://www.bg.ic.ac.uk/research/m.tang/ulis/
 * Wall: dynamic boundary conditions
 * The tank is 1m-1m, filled with water to a level 0.1m. The rectilinear sinusoidal excitation in x direction
 * is fx=0.5*(2pi*0.1462)^2sin(2pi*0.1462*t). Total simulation time is 41.05s.
 * StepAlgorithm="Verlet"; VerletSteps=40; Kernel="Wendland"; Viscosity="Artificial"; Visco=0.100000; CaseNp=20801
 * CaseNbound=801; CaseNfluid=20000; Dx=0.01; H=0.014142;  CteB=1115537.125; Gamma=7.000000
 * RhopZero=1000.000000; Cs0=88.3672; CFLnumber=0.200000; DtIni=0.000160038; DtMin=8.00192e-06; MassFluid=0.1
 * Awen (Wendland)=2784.999756; Bwen (Wendland)=-984716.8125; TimeMax=2; TimePart=0.01; Gravity=(0.000000,0.000000,-9.810000)
 * RhopOutMin=700.000000;RhopOutMax=1300.000000
 * Author: Xinhuan Zhou, PhD of Imperial College London
 * Date: 28 October 2019
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

string output_direc = "./output";
string filename = "slosh";
string logname;

float TIME_ALL=41.05f;
float TIME_SAVE=0.1f;
int VerletSteps=40;
float DISTX_MIN = -0.5f;
float DISTX_MAX = 0.5f;
float DISTZ = 1.f;

int NUM_PARTICLES=17160;
int NUM_FLUIDS=15560;
int NUM_WALLS=1600;
int bx=135;
float dx=0.0025f;
float RHO0=1000.f;
int NUM_THREADS=BLOCK_X*bx;
float H0=0.00325f;
float CX=floor((DISTX_MAX-DISTX_MIN)/2.f/H0)+1.f;
float CZ=floor(DISTZ/2.f/H0)+1.f;
float CFL=0.2f;
int NUM_CELLS=(int)CX*CZ;
float DtMin=8.00192e-06f;

// CUDA global constants
__constant__ int d_NUM_THREADS=17280;
__constant__ float d_H0=0.00325f;
__constant__ float d_CX=154;
__constant__ float d_CZ=154;
__constant__ float d_p0=53254.285156f;
__constant__ float d_RHO0=1000.f;
//__constant__ float d_coef=130932.96395f;
__constant__ float d_coef=52733.589844f;
__constant__ float d_coef1=-81134320.f;
__constant__ float d_coef2=-507089.5f;
__constant__ int d_bx=135;
__constant__ float d_MASS=0.00625f;
__constant__ float d_CS=19.3075f;
__constant__ float d_CFL=0.2f;
__constant__ float d_RhopOutMin=700.f;
__constant__ float d_RhopOutMax=1300.f;
__constant__ float d_DtMin=8.41642e-6f;
__constant__ float flt_max=3.40282e+38f;
__constant__ int int_max=2147483647;
__constant__ float d_DISTX_MIN = -0.5f;
__constant__ float d_DISTX_MAX = 0.5f;
__constant__ float d_DISTZ = 1.f;
__constant__ float d_VISCO=1e-6f;

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
		  getline(ifs,value,',');
		  h_veld[NUM_THREADS+ind]=stod(value);

		  getline(ifs,value,',');
		  h_veld[NUM_THREADS*2+ind]=stod(value);
		  getline(ifs,value,',');
		  getline(ifs,value,',');
		  getline(ifs,value,',');
		  h_pos[ind]=stod(value);
		  getline(ifs,value,',');
		  getline(ifs,value,'\n');
		  h_pos[NUM_THREADS*1+ind]=stod(value);
		  h_keep[ind]=true;
		}
    ifs.close();
	}
	cout<<"Initialization completed"<<CX<<','<<CZ<<endl;
}

__global__ void ComputeCells(int NUM_PARTICLES,float*  __restrict__ d_pos,int* __restrict__ d_particle,int* __restrict__ d_Pkey,int* __restrict__ d_num,bool* __restrict__ d_keep){
	int ind,cellidx;
	float rx,rz;
	ind=threadIdx.x+blockIdx.x*BLOCK_X;
	
	if(ind<NUM_PARTICLES){
		rx=d_pos[ind];
		rz=d_pos[d_NUM_THREADS+ind];
		cellidx=floor((rx-d_DISTX_MIN)/(2.0f*d_H0))+floor(rz/(2.0f*d_H0))*d_CX;
		//d_keep[ind]==true?d_particle[ind]=cellidx:d_particle[ind]=int_max;
		d_particle[ind]=cellidx;
		d_Pkey[ind]=ind;
		__syncthreads();
		atomicAdd(d_num+cellidx,1);
	}
}

__global__ void ParticleInteract(float time_bef,int NUM_WALLS,int NUM_PARTICLES,float*  __restrict__ d_pos,float*  __restrict__ d_veld,float*  __restrict__ d_f,int*  __restrict__ d_start,int*  __restrict__ d_num,float*  __restrict__ d_time,int*  __restrict__ d_Pkey,float*  __restrict__ d_deltarho,bool*  __restrict__ d_keep){
	//1.compute time step; 2.calculate density change; 3.calculate force
	int start,num,pind,ind;
	float tmp1,r1,r3,u1,u3,tmp,q,pi,pj,rhoi,rhoj,r_square,r_norm,rx,rz,ux,uz,mu,ftmp,r1_norm,r3_norm;
	ind=threadIdx.x+blockIdx.x*BLOCK_X;
	float drho=0.0f;
	float fx=0.0f;
	float fz=0.0f;	
	float tmp2=0.0f;
	
	if(ind<NUM_PARTICLES&&d_keep[ind]==true){
		rx=d_pos[ind];//pos,force
		rz=d_pos[d_NUM_THREADS+ind];
		ux=d_veld[ind];
		uz=d_veld[d_NUM_THREADS+ind];
		rhoi=d_veld[d_NUM_THREADS*2+ind];
		pi=d_p0*powf(rhoi/d_RHO0,7.0f)-d_p0;
		
		for(int cellz=(int)max(0.0f,floor(rz/(2.0f*d_H0))-1.0f);cellz<=min(floor(rz/(2.0f*d_H0))+1.0f,d_CZ-1.0f);cellz++){
			for(int cellx=(int)max(0.0f,floor((rx-d_DISTX_MIN)/(2.0f*d_H0))-1.0f);cellx<=min(floor((rx-d_DISTX_MIN)/(2.0f*d_H0))+1.0f,d_CX-1.0f);cellx++){
				pind=cellx+cellz*d_CX;
				num=d_num[pind];
				if(num==0)continue;
				start=d_start[pind];
				
				for(int cind=start;cind<start+num;cind++){
					pind=d_Pkey[cind];//d_veld-vel,density,reuse pind
					if(d_keep[pind]==false)continue;
					r1=rx-d_pos[pind];//pos,force
					r3=rz-d_pos[d_NUM_THREADS+pind];

					r_square=r1*r1+r3*r3;
					r_norm=sqrt(r_square);
					r1_norm=r1/r_norm;
					r3_norm=r3/r_norm;
					q=r_norm/d_H0;

					if(q>=2||q==0)continue;
					tmp=q*powf(1.0f-q/2.0f,3.0f);

					u1=ux-d_veld[pind];
					u3=uz-d_veld[d_NUM_THREADS+pind];
					rhoj=d_veld[d_NUM_THREADS*2+pind];
					pj=d_p0*powf(rhoj/d_RHO0,7.0f)-d_p0;
					
					mu=d_H0*(u1*r1+u3*r3)/(r_square+0.01f*d_H0*d_H0);
					if(pind>=NUM_WALLS){
						tmp2=max(tmp2,fabs(mu));
					}				
					
					drho+=tmp*(r1_norm*u1+r3_norm*u3);
					//drho+=tmp*(r1_norm*u1+r3_norm*u3)+0.2f*d_H0*d_CS*(1.f-rhoi/rhoj)*tmp/r_norm;

					if(mu>=0)tmp1=0.0f;
					else{
						tmp1=-0.2f*d_CS*mu/(rhoi+rhoj);
					}
					ftmp=(pi+pj)/rhoi/rhoj+tmp1;
					fx=fx-tmp*r1_norm*ftmp;
					fz=fz-tmp*r3_norm*ftmp;
				}
			}
		}
		ftmp=0.421914534189641f*sinf(0.918601691909655f*time_bef);
		fx=fx+ftmp/d_coef2;
		fz=fz-9.81f/d_coef2;
		tmp1=d_H0/sqrt(fx*fx+fz*fz)/d_coef2;
		tmp1=sqrt(tmp1);
		if(isnan(tmp1))tmp1=flt_max;
		tmp2=d_H0/(d_CS+tmp2);
		if(ind>=NUM_WALLS)d_time[ind]=min(tmp1,tmp2);
		d_f[ind]=fx;
		d_f[d_NUM_THREADS+ind]=fz;
		d_deltarho[ind]=drho;
	}
}

__global__ void SysUpdate(int NUM_WALLS,int NUM_PARTICLES,float* DT,float*  __restrict__ d_deltarho,float*  __restrict__ d_f,float*  __restrict__ d_pos,float*  __restrict__ d_veld,float*  __restrict__ d_veld_pre,float*  __restrict__ d_time,bool*  __restrict__ d_keep){
	int ind=threadIdx.x+blockIdx.x*BLOCK_X;
	float rho,rx,rz,fx,fz,drho,ux,uz,ux_pre,uz_pre,rho_pre;
	float T=*DT;
	T=max(T*d_CFL,d_DtMin);
	
	if(ind<NUM_PARTICLES&&d_keep[ind]==true){
		rx=d_pos[ind];//pos,force
		rz=d_pos[d_NUM_THREADS+ind];
		fx=d_f[ind];
		fz=d_f[d_NUM_THREADS+ind];	
		drho=d_deltarho[ind];
		
		ux=d_veld[ind];//vel,density//vel,density
		uz=d_veld[d_NUM_THREADS+ind];

		ux_pre=d_veld_pre[ind];//vel,density//vel,density
		uz_pre=d_veld_pre[d_NUM_THREADS+ind];
		rho_pre=d_veld_pre[d_NUM_THREADS*2+ind];

		rx=rx+T*ux+0.5f*T*d_coef2*fx*T;
		rz=rz+T*uz+0.5f*T*d_coef2*fz*T;
		ux=ux_pre+2.0f*T*d_coef2*fx;
		uz=uz_pre+2.0f*T*d_coef2*fz;
		rho=rho_pre+2.0f*T*d_coef2*drho;
		if(ind<NUM_WALLS)rho=max(rho,1000.f);
		d_veld_pre[d_NUM_THREADS*2+ind]=rho;
		if(ind>=NUM_WALLS){
			if(rho<=d_RhopOutMin||rho>=d_RhopOutMax||rx>=d_DISTX_MAX||rx<=d_DISTX_MIN||rz>=d_DISTZ||rz<=0){
				d_time[ind]=flt_max;
				d_keep[ind]=false;
			}
			
			d_veld_pre[ind]=ux;
			d_veld_pre[d_NUM_THREADS+ind]=uz;
			d_pos[ind]=rx;
			d_pos[d_NUM_THREADS+ind]=rz;	
		}
	}
}
		
__global__ void EulerUpdate(int NUM_WALLS,int NUM_PARTICLES,float* DT,float*  __restrict__ d_deltarho,float*  __restrict__ d_f,float*  __restrict__ d_pos,float*  __restrict__ d_veld,float*  __restrict__ d_veld_pre,float*  __restrict__ d_time,bool*  __restrict__ d_keep){
	int ind=threadIdx.x+blockIdx.x*BLOCK_X;
	float rho,rx,rz,fx,fz,drho,ux,uz;
	float T=*DT;
	T=max(T*d_CFL,d_DtMin);
	
	if(ind<NUM_PARTICLES&&d_keep[ind]==true){
		rx=d_pos[ind];//pos,force
		rz=d_pos[d_NUM_THREADS+ind];
		fx=d_f[ind];
		fz=d_f[d_NUM_THREADS+ind];	
		drho=d_deltarho[ind];
		
		ux=d_veld[ind];//vel,density//vel,density
		uz=d_veld[d_NUM_THREADS+ind];
		rho=d_veld[d_NUM_THREADS*2+ind];		
	
		rx=rx+T*ux+0.5f*T*d_coef2*fx*T;
		rz=rz+T*uz+0.5f*T*d_coef2*fz*T;
		ux=ux+T*d_coef2*fx;
		uz=uz+T*d_coef2*fz;
		rho=rho+T*d_coef2*drho;
		if(ind<NUM_WALLS)rho=max(rho,1000.f);
		d_veld_pre[d_NUM_THREADS*2+ind]=rho;		
		
		if(ind>=NUM_WALLS){
			if(rho<=d_RhopOutMin||rho>=d_RhopOutMax||rx>=d_DISTX_MAX||rx<=d_DISTX_MIN||rz>=d_DISTZ||rz<=0){
				d_time[ind]=flt_max;
				d_keep[ind]=false;
			}
			d_veld_pre[ind]=ux;//vel,density. We need to swap the two pointers:d_veld and d_veld_pre
			d_veld_pre[d_NUM_THREADS+ind]=uz;
			d_pos[ind]=rx;
			d_pos[d_NUM_THREADS+ind]=rz;	
		}
	}
}
	
void outputSave(int t) {
	string datafilename = output_direc +"/slosh_" + to_string(t) + ".csv";;
	ofstream ofs(datafilename);
	ofs<<"x,y,z,vel,ux,uz,rho"<<endl;
					
	for(int ind=0;ind<NUM_PARTICLES;ind++){//outside:-1,wall:1,fluid:0
		if(h_keep[ind]==true){
			ofs<<h_pos[ind]<<','<<0<<','<<h_pos[NUM_THREADS+ind]<<','<<sqrt(pow(h_veld[ind],2.f)+pow(h_veld[NUM_THREADS+ind],2.f))<<','<<h_veld[ind]<<','<<h_veld[NUM_THREADS+ind]<<','<<h_veld[NUM_THREADS*2+ind]<<endl;
		}
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
	h_pos=(float*)malloc(sizeof(float)*NUM_THREADS*2);
	h_veld=(float*)malloc(sizeof(float)*NUM_THREADS*3);
	h_f=(float*)malloc(sizeof(float)*NUM_THREADS*2);
	h_T=(float*)malloc(sizeof(float));
	h_keep=(bool*)malloc(sizeof(bool)*NUM_THREADS);
	*h_T=0.000168328f/CFL;
	
	cudaMalloc(&d_pos,NUM_THREADS*2*sizeof(float));
	cudaMalloc(&d_veld,NUM_THREADS*3*sizeof(float));
	cudaMalloc(&d_veld_pre,NUM_THREADS*3*sizeof(float));
	cudaMalloc(&d_f,NUM_THREADS*2*sizeof(float));
	cudaMalloc(&d_particle,NUM_THREADS*sizeof(int));
	cudaMalloc(&d_start,NUM_CELLS*sizeof(int));
	cudaMalloc(&d_num,NUM_CELLS*sizeof(int));
	cudaMalloc(&d_Pkey,NUM_THREADS*sizeof(int));
	cudaMalloc(&d_time,NUM_THREADS*sizeof(float));
	cudaMalloc(&d_deltarho,NUM_THREADS*sizeof(float));
	cudaMalloc(&d_keep,NUM_THREADS*sizeof(bool));
	cudaMalloc(&DT,sizeof(float));
	cudaMemset(d_time,2.f,NUM_THREADS*sizeof(float));
	
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
	cudaMemcpy(d_pos,h_pos,NUM_THREADS*2*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_veld,h_veld,NUM_THREADS*3*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_veld_pre,h_veld,NUM_THREADS*3*sizeof(float),cudaMemcpyHostToDevice);
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
		
		ParticleInteract<<<dimGrid,dimBlock>>>(time_bef,NUM_WALLS,NUM_PARTICLES,d_pos,d_veld,d_f,d_start,d_num,d_time,d_Pkey,d_deltarho,d_keep);
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
		//if(i>30860&&i<30890){
			cout << "ITERATION # " << i<<" ; TIME "<<time_aft <<endl;
			logfile<< "ITERATION # " << i<<" ; TIME "<<time_aft <<endl;
			cudaMemcpy(h_pos,d_pos,NUM_THREADS*2*sizeof(float),cudaMemcpyDeviceToHost);
			cudaMemcpy(h_veld,d_veld,NUM_THREADS*3*sizeof(float),cudaMemcpyDeviceToHost);
			cudaMemcpy(h_keep,d_keep,NUM_THREADS*sizeof(bool),cudaMemcpyDeviceToHost);
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
